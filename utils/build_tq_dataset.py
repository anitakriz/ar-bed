import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import random
import re
import sys
from threading import Event, RLock, Thread
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import os
from dotenv import load_dotenv
from tqdm.auto import tqdm
import torch
from openai import AzureOpenAI
from datasets import Dataset, load_dataset
import unicodedata

load_dotenv()

from .misc import set_seed, resolve_repo_path
from .api_models import ApiUsageTracker
from .tq_prompting import tq_question_guidance


def load_tq_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}


TQ_FEEDBACK_CACHES = {}
TQ_FEEDBACK_LOCK = RLock()
JSONL_WRITE_LOCK = RLock()


def get_tq_cache(cache_path):
    cache_path = str(Path(cache_path).expanduser().resolve())

    if cache_path not in TQ_FEEDBACK_CACHES:
        TQ_FEEDBACK_CACHES[cache_path] = load_tq_cache(cache_path)

    return TQ_FEEDBACK_CACHES[cache_path]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "tq"
DEFAULT_TQ_THETA_SET_PATH = "data/tq/animals_200.json"
DEFAULT_TQ_TRAIN_SAMPLED_STATES = "train_sampled_states_200animals.jsonl"
DEFAULT_TQ_EVAL_SAMPLED_STATES = "eval_sampled_states_200animals.jsonl"

ANSWER_LABELS = ["yes", "no", "maybe"]
ACTIVE_THRESHOLD = 0.2
QUESTIONER_SYSTEM_PROMPT = """You are playing Twenty Questions.

You must return your strategy and exactly one question using the required XML format.
Do not include anything outside the <think> and <question> tags.
""".strip()

BELIEF_SYSTEM_PROMPT = """You are playing a game of 20 Questions. Using all of the questions and answers so far, generate a diverse set of candidate hypotheses that satisfy the current history of questions and answers."""

LIKELIHOOD_SYSTEM_PROMPT = """You are an impartial judge for the game of 20 Questions. Given a question and a list of guesses, output the likelihood of the answers "yes", "no", and "maybe" in a JSON format."""


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_questioner_prompt(
    history: Optional[Sequence[Any]],
    category: str,
    turn: int,
    max_turns: int = 20,
    candidates: Optional[Sequence[str]] = None,
    question_guidance: Optional[str] = None,
) -> List[Dict[str, str]]:

    candidates = [
        candidate for candidate in (candidates or []) if str(candidate).strip()
    ]
    candidates = list(dict.fromkeys(candidates))

    use_conditional_generation = len(candidates) >= 2

    if use_conditional_generation:
        generation_instructions = f"""
Possible Candidates:
{json.dumps(candidates, ensure_ascii=False)}

You may ask a question that is useful for distinguishing between the candidate
hypotheses. 

The secret {category} is not guaranteed to appear in this candidate list, so do not assume the list is complete.

STRICT output format:
<think>
State your reasoning.
</think>

<question>
State your question.
</question>

""".strip()

    else:
        generation_instructions = f"""

STRICT output format:
<think>
State your reasoning.
</think>

<question>
State your question.
</question>

""".strip()

    user_msg = f"""
Goal: Identify the secret {category} in at most {max_turns} turns.

RULES:
- Only ask questions that can be answered with "yes", "no", or "maybe".

Game state:
Turn: {turn}/{max_turns}

Previous turns:
{format_history(history)}

{generation_instructions}

""".strip()

    system_prompt = QUESTIONER_SYSTEM_PROMPT
    if question_guidance:
        system_prompt = (
            f"{system_prompt}\n\nDomain guidance:\n{question_guidance.strip()}"
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]


def extract_question(text: Any) -> Optional[str]:
    if isinstance(text, list):
        text = text[0]["content"]

    text = str(text)
    match = re.search(
        r"<question>\s*(.*?)\s*</question>",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        question = match.group(1).strip()
        return question or None
    return None


def extract_likelihoods_prompt(questions, guesses):
    questions = list(questions)
    guesses = list(guesses)

    prompt = f"""
Guesses:
{json.dumps(guesses, ensure_ascii=False)}

Questions:
{json.dumps(questions, ensure_ascii=False)}

For each question and each guess, estimate:

p(answer | guess is the true hidden target, question)

where answer is one of:
- "yes"
- "no"
- "maybe"

Rules:
- For each question/guess pair, the probabilities for "yes", "no", and "maybe" must sum to 1.
- Use "maybe" when the question is ambiguous, underspecified, or depends on interpretation.

In addition, return a "solved" field for each question/guess pair.
"solved" should be true if the question directly guesses that exact hidden target.
For example:
- guess: "tiger", question: "Is the secret animal a tiger?" -> solved: true
- guess: "lion", question: "Is the secret animal a tiger?" -> solved: false
- guess: "tiger", question: "Is it a mammal?" -> solved: false

STRICT output format:
{{
  "<question>": {{
    "<guess>": {{"yes": 0.0, "no": 0.0, "maybe": 0.0, "solved": false}}
  }}
}}
""".strip()

    return [
        {"role": "system", "content": LIKELIHOOD_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _repair_common_json_commas(text):
    """
    Conservatively repair only comma-placement errors outside JSON strings.

    Repairs:
      1. duplicate commas:
             {"a": 1, , "b": 2}
         ->  {"a": 1, "b": 2}

      2. trailing commas before a closing object/array:
             {"a": 1,}
             [1, 2,]
         ->  {"a": 1}
             [1, 2]

    String contents are never modified.
    """
    chars = list(str(text))
    keep = [True] * len(chars)

    in_string = False
    escaped = False
    last_significant = None

    i = 0
    while i < len(chars):
        ch = chars[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            last_significant = ch
            i += 1
            continue

        if ch.isspace():
            i += 1
            continue

        if ch == ",":
            # Duplicate comma: keep the first one, remove subsequent commas
            # until another real JSON token appears.
            if last_significant == ",":
                keep[i] = False
                i += 1
                continue

            # Trailing comma: look ahead to the next non-whitespace token.
            j = i + 1
            while j < len(chars) and chars[j].isspace():
                j += 1

            if j < len(chars) and chars[j] in "}]":
                keep[i] = False
                i += 1
                continue

            last_significant = ","
            i += 1
            continue

        last_significant = ch
        i += 1

    return "".join(ch for ch, should_keep in zip(chars, keep) if should_keep)


def extract_json_object(text):
    text = str(text).strip()

    # Handles accidental ```json ... ``` wrapping.
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    # Handles accidental extra text around JSON.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response:\n{text}")

    json_text = text[start : end + 1]

    # Fast path: trust valid JSON exactly as returned.
    original_error = None
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        original_error = exc

    # Conservative recovery for the common malformed-comma cases we have
    # actually observed from the oracle. We do NOT attempt arbitrary JSON
    # reconstruction or regex-salvage individual likelihood entries.
    repaired = _repair_common_json_commas(json_text)

    if repaired != json_text:
        try:
            parsed = json.loads(repaired)
            print(
                "Recovered malformed likelihood JSON by repairing "
                "duplicate/trailing commas.",
                flush=True,
            )
            return parsed
        except json.JSONDecodeError:
            pass

    # Preserve the original error location/message when repair is insufficient.
    if original_error is not None:
        raise original_error
    raise ValueError(f"Could not parse JSON object from response:\n{text}")


def normalize_likelihood_entry(probs):
    """
    Validate and normalize one LLM likelihood entry.

    Returns None when the LLM omitted the entry or returned malformed
    probabilities.
    """
    if not isinstance(probs, dict):
        return None

    try:
        yes = float(probs["yes"])
        no = float(probs["no"])
        maybe = float(probs["maybe"])
    except (KeyError, TypeError, ValueError):
        return None

    values = np.asarray([yes, no, maybe], dtype=float)

    if not np.all(np.isfinite(values)):
        return None

    if np.any(values < 0):
        return None

    total = float(values.sum())

    if total <= 0:
        return None

    raw_solved = probs.get("solved", False)

    if isinstance(raw_solved, str):
        solved = raw_solved.strip().lower() == "true"
    else:
        solved = bool(raw_solved)

    return {
        "yes": yes / total,
        "no": no / total,
        "maybe": maybe / total,
        "solved": solved,
    }


def normalize_lookup_key(value):
    """Normalize text for accent-insensitive dictionary lookup."""
    text = str(value).strip().casefold()
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )


def _matching_normalized_keys(mapping, requested_key):
    """
    Return all keys equivalent to requested_key under normalize_lookup_key().

    The exact requested spelling is always checked first. This matters when a
    cache/response accidentally contains both a canonical spelling such as
    ``vicuña`` and an alias such as ``vicuñá``/``vicuna``: a malformed alias
    must not shadow a valid canonical entry.
    """
    if not isinstance(mapping, dict):
        return []

    requested_key = str(requested_key)
    normalized_requested = normalize_lookup_key(requested_key)

    matches = []

    if requested_key in mapping:
        matches.append(requested_key)

    for key in mapping:
        if key == requested_key:
            continue
        if normalize_lookup_key(key) == normalized_requested:
            matches.append(key)

    return matches


def _lookup_normalized_likelihood(mapping, requested_key):
    """
    Find the first *valid* likelihood entry for requested_key.

    Exact spelling is preferred, then accent/case-equivalent aliases.
    Returns (matched_key, normalized_probs), or (None, None).
    """
    for key in _matching_normalized_keys(mapping, requested_key):
        normalized = normalize_likelihood_entry(mapping.get(key))
        if normalized is not None:
            return key, normalized

    return None, None


def _lookup_normalized_mapping_value(mapping, requested_key):
    """Lookup a non-likelihood mapping value with exact-key preference."""
    if not isinstance(mapping, dict):
        return None

    requested_key = str(requested_key)

    if requested_key in mapping:
        return mapping[requested_key]

    normalized_requested = normalize_lookup_key(requested_key)

    for key, value in mapping.items():
        if normalize_lookup_key(key) == normalized_requested:
            return value

    return None


def parse_likelihoods_response(text, questions, guesses):
    try:
        data = extract_json_object(text)
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        print(f"Could not parse likelihood response: {e}")
        print(f"Raw response:\n{text}")

        # Important: an invalid JSON response means NONE of its apparent
        # entries are trusted. tq_feedback() will retry the unresolved pairs
        # in smaller batches.
        return {question: {} for question in questions}

    if not isinstance(data, dict):
        print(
            f"Likelihood response was not a dictionary. Received: {type(data).__name__}"
        )

        return {question: {} for question in questions}

    out = {}

    for question in questions:
        # Prefer the exact question string, but tolerate case/accent variants.
        raw_probs = _lookup_normalized_mapping_value(
            data,
            question,
        )

        if not isinstance(raw_probs, dict):
            print(f"Skipping missing likelihood question: {question!r}")
            out[question] = {}
            continue

        out[question] = {}

        for guess in guesses:
            _, probs = _lookup_normalized_likelihood(
                raw_probs,
                guess,
            )

            if probs is None:
                print(
                    "Skipping missing or malformed likelihood entry: "
                    f"question={question!r}, guess={guess!r}"
                )
                continue

            # Always return the caller's canonical spelling.
            out[question][guess] = probs

    return out


def _query_likelihoods_with_recovery(
    question,
    words,
    usage_tracker=None,
    *,
    split_depth=0,
    max_split_depth=3,
    singleton_retry=1,
):
    """
    Query likelihoods and recover from malformed/incomplete model JSON.

    Normal path:
        one request for the full requested batch.

    Recovery path:
        - keep every valid returned entry;
        - retry only unresolved words;
        - if several words are unresolved, split them into smaller batches;
        - retry a singleton once.

    This avoids throwing away valid-looking entries such as ``sloth`` merely
    because a different line made the overall JSON invalid. We never regex-
    salvage values from invalid JSON; instead we ask the oracle again with a
    smaller output request.
    """
    print(
        "Starting query_likelihoods_with_recovery: "
        f"question={question!r}, num_words={len(words)}"
    )
    words = list(dict.fromkeys(str(word) for word in words))

    if not words:
        return {}

    messages = extract_likelihoods_prompt(
        [question],
        words,
    )

    text = call_policy(
        messages,
        provider="azure-openai",
        usage_tracker=usage_tracker,
    )

    data = parse_likelihoods_response(
        text,
        [question],
        words,
    )

    found = dict(data.get(question, {}))

    unresolved = [word for word in words if word not in found]

    if not unresolved:
        return found

    # A single omitted/malformed entry gets one clean retry.
    if len(unresolved) == 1:
        word = unresolved[0]

        # If the original request already contained only this singleton,
        # limit retries explicitly.
        if len(words) == 1:
            if singleton_retry <= 0:
                return found

            retry = _query_likelihoods_with_recovery(
                question,
                [word],
                usage_tracker=usage_tracker,
                split_depth=split_depth,
                max_split_depth=max_split_depth,
                singleton_retry=singleton_retry - 1,
            )
            found.update(retry)
            return found

        retry = _query_likelihoods_with_recovery(
            question,
            [word],
            usage_tracker=usage_tracker,
            split_depth=split_depth + 1,
            max_split_depth=max_split_depth,
            singleton_retry=singleton_retry,
        )
        found.update(retry)
        return found

    # If a large response was malformed, parse_likelihoods_response returned
    # no trusted entries. Retry the unresolved subset in smaller batches.
    if split_depth >= max_split_depth:
        print(
            "Giving up on unresolved likelihood entries after recovery "
            f"depth={split_depth}: question={question!r}, "
            f"num_unresolved={len(unresolved)}",
            flush=True,
        )
        return found

    midpoint = max(1, len(unresolved) // 2)

    for subset in (
        unresolved[:midpoint],
        unresolved[midpoint:],
    ):
        if not subset:
            continue

        recovered = _query_likelihoods_with_recovery(
            question,
            subset,
            usage_tracker=usage_tracker,
            split_depth=split_depth + 1,
            max_split_depth=max_split_depth,
            singleton_retry=singleton_retry,
        )
        found.update(recovered)

    return found


def save_tq_cache(cache, cache_path):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # temporary_path = cache_path.with_suffix(
    #     cache_path.suffix + ".tmp"
    # )

    with cache_path.open("w") as f:
        json.dump(
            cache,
            f,
            indent=2,
            sort_keys=True,
        )

    # os.replace(
    #     temporary_path,
    #     cache_path,
    # )


def _tq_feedback_unlocked(
    questions,
    ground_truths,
    cache_path,
    usage_tracker=None,
    track_cache_status=False,
):

    questions = list(questions)
    ground_truths = list(ground_truths)

    cache = get_tq_cache(cache_path)

    cached_answers = {question: {} for question in questions}

    missing_words_by_question = {}
    cache_hits = 0
    cache_misses = 0

    for question in questions:
        if question not in cache:
            cache[question] = {}

        missing_words = []

        for i, word in enumerate(ground_truths):
            word = str(word)

            matching_keys = _matching_normalized_keys(
                cache[question],
                word,
            )

            cached_key, normalized_value = _lookup_normalized_likelihood(
                cache[question],
                word,
            )

            if normalized_value is not None:
                # Canonicalize ALL accent/case aliases to the spelling the
                # caller requested. This prevents a stale malformed alias from
                # later shadowing a valid canonical entry.
                for alias_key in matching_keys:
                    if alias_key != word:
                        cache[question].pop(alias_key, None)

                cache[question][word] = normalized_value
                cached_answers[question][i] = normalized_value
                cache_hits += 1
            else:
                # Remove every malformed equivalent spelling, not just the
                # exact requested key.
                for alias_key in matching_keys:
                    cache[question].pop(alias_key, None)

                missing_words.append(word)
                cache_misses += 1

        if missing_words:
            missing_words_by_question[question] = missing_words

    if (
        track_cache_status
        and usage_tracker is not None
        and hasattr(usage_tracker, "record_oracle_lookup")
    ):
        usage_tracker.record_oracle_lookup(
            num_questions=len(questions),
            num_ground_truths=len(ground_truths),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

    if missing_words_by_question:
        missing_questions = list(missing_words_by_question.keys())

        # Preserve the historical fast path: ask for the union of missing
        # words once. If this parses cleanly and includes every requested
        # pair, there are NO extra API calls.
        missing_words = sorted(
            {word for words in missing_words_by_question.values() for word in words}
        )

        messages = extract_likelihoods_prompt(
            missing_questions,
            missing_words,
        )

        text = call_policy(
            messages,
            provider="azure-openai",
            usage_tracker=usage_tracker,
        )

        initial_data = parse_likelihoods_response(
            text,
            missing_questions,
            missing_words,
        )

        unresolved_by_question = {}

        for question in missing_questions:
            question_probs = initial_data.get(
                question,
                {},
            )

            unresolved = []

            for i, word in enumerate(ground_truths):
                word = str(word)

                if word not in missing_words_by_question[question]:
                    continue

                _, probs = _lookup_normalized_likelihood(
                    question_probs,
                    word,
                )

                if probs is None:
                    unresolved.append(word)
                    continue

                for alias_key in _matching_normalized_keys(
                    cache[question],
                    word,
                ):
                    if alias_key != word:
                        cache[question].pop(alias_key, None)

                cache[question][word] = probs
                cached_answers[question][i] = probs

            if unresolved:
                unresolved_by_question[question] = unresolved

        # Recovery is invoked ONLY for pairs missing after the normal call.
        # A globally malformed JSON response therefore causes smaller retries;
        # a valid response that omitted just "vicuña" retries only "vicuña".
        for question, unresolved in unresolved_by_question.items():
            recovered = _query_likelihoods_with_recovery(
                question,
                unresolved,
                usage_tracker=usage_tracker,
            )

            for i, word in enumerate(ground_truths):
                word = str(word)

                if word not in unresolved:
                    continue

                _, probs = _lookup_normalized_likelihood(
                    recovered,
                    word,
                )

                # Still unresolved after bounded recovery: leave uncached and
                # return None for this position, preserving previous semantics.
                if probs is None:
                    continue

                for alias_key in _matching_normalized_keys(
                    cache[question],
                    word,
                ):
                    if alias_key != word:
                        cache[question].pop(alias_key, None)

                cache[question][word] = probs
                cached_answers[question][i] = probs

        save_tq_cache(
            cache,
            cache_path,
        )

    return [
        [cached_answers[question].get(i) for i in range(len(ground_truths))]
        for question in questions
    ]


def tq_feedback(
    questions,
    ground_truths,
    cache_path,
    usage_tracker=None,
    track_cache_status=False,
):
    with TQ_FEEDBACK_LOCK:
        return _tq_feedback_unlocked(
            questions=questions,
            ground_truths=ground_truths,
            cache_path=cache_path,
            usage_tracker=usage_tracker,
            track_cache_status=track_cache_status,
        )


def format_history(history):
    if history is None or len(history) == 0:
        return "No history available. This is your first turn."

    formatted_history = ""
    for item in history:
        if isinstance(item, dict):
            question = item["question"]
            answer = item["answer"]
        else:
            question, answer = item
        formatted_history += f"Question: {question}\nAnswer: {answer}\n"
    return formatted_history


def history_questions(history):
    return [item["question"] if isinstance(item, dict) else item[0] for item in history]


def history_answers(history):
    return [item["answer"] if isinstance(item, dict) else item[1] for item in history]


def extract_beliefs_prompt(
    history, active_candidates, category="animals", num_new_hypotheses=5
):
    prompt = f"""
    Generate up to {num_new_hypotheses} additional candidate entities that are {category}.

    Each new candidate must satisfy every clue in the history.
    Do not repeat any entity from the already retained candidates.
    Do not repeat entities within your output.
    Produce a varied set by identifying features not implied by the clues and diversifying along them.

    Already retained candidates:
    {active_candidates}

    History:
    {format_history(history)}

    Return only valid JSON.

    STRICT output format:
    {{"candidates": ["entity1", "entity2", "entity3"]}}
    """.strip()

    return [
        {"role": "system", "content": BELIEF_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def extract_beliefs(
    history,
    active_candidates=None,
    num_needed=15,
    category="animals",
    usage_tracker=None,
):
    if active_candidates is None:
        active_candidates = []

    candidates = list(
        dict.fromkeys(
            [
                str(candidate).strip().lower()
                for candidate in active_candidates
                if str(candidate).strip()
            ]
        )
    )

    messages = extract_beliefs_prompt(
        history=history,
        active_candidates=candidates,
        category=category,
        num_new_hypotheses=num_needed,
    )

    text = call_policy(
        messages,
        provider="azure-openai",
        usage_tracker=usage_tracker,
        usage_kind="oracle",
    )

    new_candidates = extract_json_object(text)

    filtered = []
    for candidate in new_candidates["candidates"]:
        candidate = str(candidate).strip().lower()

        if candidate and candidate not in candidates:
            filtered.append(candidate)

    return filtered


def filter_compatible_candidates(
    likelihoods_by_question: Sequence[Dict[str, Dict[str, float]]],
    history: Sequence[Any],
    threshold: float = ACTIVE_THRESHOLD,
) -> List[str]:
    if not likelihoods_by_question:
        return []

    if not history:
        return sorted(likelihoods_by_question[0].keys())

    observed_answers = []
    for item in history:
        if isinstance(item, dict):
            observed_answers.append(str(item["answer"]).strip().lower())
        else:
            observed_answers.append(str(item[1]).strip().lower())

    candidate_entities = set(likelihoods_by_question[0].keys())
    for likelihoods in likelihoods_by_question[1:]:
        candidate_entities &= set(likelihoods.keys())

    compatible = []

    for entity in sorted(candidate_entities):
        keep = True

        for likelihoods, observed_answer in zip(
            likelihoods_by_question, observed_answers
        ):
            item = likelihoods.get(entity, {})
            probs = get_probs(item)

            prob_observed = probs.get(observed_answer)

            if prob_observed < threshold:
                keep = False
                break

        if keep:
            compatible.append(entity)

    return compatible


def update_C(
    C: np.ndarray,
    question: str,
    true_word: str,
    theta_set: Sequence[str],
    filter_threshold: float,
    cache_path,
    usage_tracker=None,
):
    active_indices = np.where(C > filter_threshold)[0]
    active_candidates = [theta_set[i] for i in active_indices]

    print(len(active_candidates))

    if not active_candidates:
        return C, None, False

    query_words = list(active_candidates)

    # The oracle response for the true word is required to determine
    # the observed answer.
    if true_word not in query_words:
        query_words.append(true_word)

    likelihoods_last = tq_feedback(
        questions=[question],
        ground_truths=query_words,
        cache_path=cache_path,
        usage_tracker=usage_tracker,
        track_cache_status=True,
    )[0]

    likelihoods_by_word = {
        word: probs
        for word, probs in zip(
            query_words,
            likelihoods_last,
        )
        if probs is not None
    }

    true_probs = likelihoods_by_word.get(true_word)

    if true_probs is None:
        raise ValueError(
            "Missing likelihood probabilities for the true word: "
            f"question={question!r}, true_word={true_word!r}"
        )

    observed_answer = max(
        ANSWER_LABELS,
        key=lambda answer: true_probs.get(answer),
    )

    valid_indices = []
    likelihood_values = []

    for index, theta in zip(
        active_indices,
        active_candidates,
    ):
        probs = likelihoods_by_word.get(theta)

        if probs is None:
            print(
                "Skipping C update for missing candidate likelihood: "
                f"question={question!r}, candidate={theta!r}"
            )
            continue

        likelihood = probs.get(observed_answer)

        try:
            likelihood = float(likelihood)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(likelihood) or likelihood < 0:
            continue

        valid_indices.append(index)
        likelihood_values.append(likelihood)

    if not valid_indices:
        raise ValueError(
            "No valid active-candidate likelihoods were returned for "
            f"question={question!r}"
        )

    valid_indices = np.asarray(
        valid_indices,
        dtype=int,
    )

    likelihood_vec = np.asarray(
        likelihood_values,
        dtype=float,
    )

    C[valid_indices] *= likelihood_vec

    solved = bool(true_probs.get("solved"))

    return C, observed_answer, solved


def as_likelihood_dict(candidates, likelihood_rows):
    return {
        candidate: probs
        for candidate, probs in zip(candidates, likelihood_rows)
        if probs is not None
    }


def get_probs(item):
    # Supports both old format:
    # {"yes": ..., "no": ..., "maybe": ...}
    #
    # and new format:
    # {"probs": {"yes": ..., ...}, "solved": false}
    if item is None:
        return {}

    if "probs" in item:
        return item["probs"]

    return item


def refill_sampled_candidate_set(
    history: Sequence[Any],
    previous_candidates: Optional[Sequence[str]],
    num_hypotheses: int,
    filter_threshold: float,
    refill_rounds: int,
    likelihood_cache_path,
    category: str = "animals",
    usage_tracker=None,
) -> List[str]:

    previous_candidates = list(previous_candidates or [])

    if history and previous_candidates:
        questions = history_questions(history)
        last_question = questions[-1]
        last_history = history[-1]

        last_likelihood_rows = tq_feedback(
            questions=[last_question],
            ground_truths=previous_candidates,
            cache_path=likelihood_cache_path,
            usage_tracker=usage_tracker,
            track_cache_status=True,
        )[0]

        last_likelihoods = as_likelihood_dict(
            previous_candidates,
            last_likelihood_rows,
        )

        active_candidates = filter_compatible_candidates(
            likelihoods_by_question=[last_likelihoods],
            history=[last_history],
            threshold=filter_threshold,
        )

    else:
        active_candidates = list(previous_candidates)

    for _ in range(int(refill_rounds)):
        num_needed = int(num_hypotheses) - len(active_candidates)
        if num_needed <= 0:
            break

        new_candidates = extract_beliefs(
            history=history,
            active_candidates=active_candidates,
            num_needed=num_needed,
            category=category,
            usage_tracker=usage_tracker,
        )

        if not new_candidates:
            continue

        if history:
            questions = history_questions(history)

            likelihood_rows_by_question = tq_feedback(
                questions=questions,
                ground_truths=new_candidates,
                cache_path=likelihood_cache_path,
                usage_tracker=usage_tracker,
                track_cache_status=True,
            )

            likelihoods_by_question = [
                as_likelihood_dict(new_candidates, likelihood_rows)
                for likelihood_rows in likelihood_rows_by_question
            ]

            compatible_new = filter_compatible_candidates(
                likelihoods_by_question=likelihoods_by_question,
                history=history,
                threshold=filter_threshold,
            )

        else:
            compatible_new = new_candidates

        for candidate in compatible_new:
            if candidate not in active_candidates:
                active_candidates.append(candidate)

    return active_candidates


from openai import (
    AzureOpenAI,
    BadRequestError,
    InternalServerError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
)


def call_azure_openai_policy(
    messages,
    max_retries=8,
    usage_tracker=None,
    usage_kind="oracle",
):
    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        max_retries=0,  # We handle retries below.
        timeout=180.0,
    )

    deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT"]

    retryable_errors = (
        InternalServerError,
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                reasoning_effort="none",
            )

            if usage_tracker is not None:
                if usage_kind == "questioner":
                    usage_tracker.record_questioner_response(response)
                else:
                    usage_tracker.record_oracle_response(response)

            return response.choices[0].message.content

        except retryable_errors as error:
            if attempt == max_retries - 1:
                raise

            delay = min(
                60.0,
                5.0 * (2**attempt),
            )
            delay += random.uniform(0.0, 2.0)

            print(
                f"Azure request failed: {error}\n"
                f"Retrying in {delay:.1f}s "
                f"({attempt + 1}/{max_retries})...",
                flush=True,
            )

            time.sleep(delay)


def call_policy(
    prompt,
    provider,
    model_name=None,
    temperature=0.7,
    max_output_tokens=160,
    usage_tracker=None,
    usage_kind="oracle",
):
    if provider == "azure-openai":
        return call_azure_openai_policy(
            messages=prompt,
            usage_tracker=usage_tracker,
            usage_kind=usage_kind,
        )
    raise ValueError(f"Unknown API rollout provider: {provider}")


def build_llm_rollout(
    true_word,
    category,
    theta_set,
    provider,
    model_name,
    max_turns,
    num_sampled_hypotheses,
    num_samples_per_trajectory,
    refill_rounds,
    filter_threshold,
    likelihood_cache_path,
    question_guidance=None,
    usage_tracker=None,
    temperature=0.7,
    max_output_tokens=160,
    include_sampled_candidates=True,
):
    """
    Generate one 20q trajectory using an external API model.

    The model only sees the prompt/history and proposes a guess.
    A second "oracle" LLM computes the true TQ feedback.

    If Azure rejects a prompt for a usage-policy violation, retain completed
    turns with stop_reason="api_policy_rejection", failed_turn, and error details.
    Other BadRequestError failures propagate to the caller.
    """
    history = []
    steps = []
    stop_reason = "max_turns"
    failed_turn = None
    sampled_candidates: List[str] = []
    rollout_error = None

    C = np.ones(len(theta_set), dtype=np.float32)
    print(true_word)

    for turn in range(1, int(max_turns) + 1):
        try:
            if include_sampled_candidates:
                sampled_candidates = refill_sampled_candidate_set(
                    history=history,
                    previous_candidates=sampled_candidates,
                    num_hypotheses=num_sampled_hypotheses,
                    filter_threshold=filter_threshold,
                    refill_rounds=refill_rounds,
                    likelihood_cache_path=likelihood_cache_path,
                    category=category,
                    usage_tracker=usage_tracker,
                )

            question_prompt = make_questioner_prompt(
                history,
                category,
                turn,
                max_turns=max_turns,
                question_guidance=question_guidance,
            )
            completion = call_policy(
                prompt=question_prompt,
                provider=provider,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                usage_tracker=usage_tracker,
                usage_kind="questioner",
            )

            question = extract_question(completion)
            print(question)
            if question is None:
                stop_reason = "no_answer_tags"
                break

            C, observed_answer, solved = update_C(
                C=C,
                question=question,
                true_word=true_word,
                theta_set=theta_set,
                filter_threshold=filter_threshold,
                cache_path=likelihood_cache_path,
                usage_tracker=usage_tracker,
            )
            print(observed_answer)
            if observed_answer not in ANSWER_LABELS:
                stop_reason = "no_active_candidates"
                break

            step = {
                "turn": turn,
                "question": question,
                "answer": observed_answer,
                "completion": completion,
                "C": json.dumps(C.tolist()),
                "sampled_candidates": list(sampled_candidates)
                if include_sampled_candidates
                else None,
                "solved": solved,
            }
            steps.append(step)

            history.append(
                {
                    "question": question,
                    "answer": observed_answer,
                }
            )

            if solved is True:
                stop_reason = "solved"
                break
        except BadRequestError as error:
            error_text = str(error)
            if "invalid_prompt" not in error_text and "usage policy" not in error_text:
                raise
            stop_reason = "api_policy_rejection"
            failed_turn = turn
            rollout_error = {
                "type": type(error).__name__,
                "turn": turn,
                "message": error_text,
            }
            print(
                f"Salvaging {len(steps)} completed turns after policy rejection "
                f"on turn {turn}.",
                flush=True,
            )
            break

    return {
        "true_word": true_word,
        "provider": provider,
        "steps": steps,
        "stop_reason": stop_reason,
        "error": rollout_error,
        "failed_turn": failed_turn,
    }


def append_jsonl(path, row):
    if path is None:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with JSONL_WRITE_LOCK:
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")


def build_llm_rollout_states(
    true_word,
    category,
    theta_set,
    epsilon,
    provider,
    model_name,
    max_turns,
    temperature,
    max_output_tokens,
    num_samples_per_trajectory,
    num_sampled_hypotheses,
    filter_threshold,
    refill_rounds,
    likelihood_cache_path,
    question_guidance=None,
    rollout_jsonl_path=None,
    usage_tracker=None,
    include_sampled_candidates=True,
):
    rollout = build_llm_rollout(
        true_word=true_word,
        category=category,
        theta_set=theta_set,
        provider=provider,
        model_name=model_name,
        max_turns=max_turns,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        num_samples_per_trajectory=num_samples_per_trajectory,
        num_sampled_hypotheses=num_sampled_hypotheses,
        refill_rounds=refill_rounds,
        filter_threshold=filter_threshold,
        likelihood_cache_path=likelihood_cache_path,
        question_guidance=question_guidance,
        usage_tracker=usage_tracker,
        include_sampled_candidates=include_sampled_candidates,
    )

    append_jsonl(rollout_jsonl_path, rollout)

    steps = rollout["steps"]

    if not steps:
        return []

    # Prefix length k means: use first k completed steps as history,
    # then train/evaluate the question asked at step k.
    max_prefix_len = min(len(steps) - 1, int(max_turns) - 1)
    valid_prefix_lens = list(range(max_prefix_len + 1))

    sampled_prefix_lens = random.sample(
        valid_prefix_lens,
        k=min(int(num_samples_per_trajectory), len(valid_prefix_lens)),
    )

    rows = []

    for num_prefix_turns in sampled_prefix_lens:
        history = [
            {
                "question": step["question"],
                "answer": step["answer"],
            }
            for step in steps[:num_prefix_turns]
        ]

        turn = len(history) + 1

        if num_prefix_turns == 0:
            C_current = np.ones(len(theta_set), dtype=np.float32)
        else:
            C_current = np.array(
                json.loads(steps[num_prefix_turns - 1]["C"]),
                dtype=np.float32,
            )

        target_step = steps[num_prefix_turns]

        rows.append(
            {
                "prompt": make_questioner_prompt(
                    history,
                    category,
                    turn,
                    max_turns=max_turns,
                    question_guidance=question_guidance,
                ),
                "history_json": json.dumps(history),
                "turn": turn,
                "true_word": true_word,
                "contrastive_set_json": json.dumps(theta_set),
                "C_json": json.dumps(C_current.tolist()),
                # Keep the field for backward-compatible dataset schemas, but
                # avoid all sampled-hypothesis generation when disabled.
                "sampled_candidates_json": json.dumps(
                    target_step["sampled_candidates"]
                    if target_step.get("sampled_candidates") is not None
                    else []
                ),
                "target_question": target_step["question"],
                "target_answer": target_step["answer"],
                "target_solved": target_step["solved"],
                "rollout_stop_reason": rollout["stop_reason"],
                "prefix_len": num_prefix_turns,
            }
        )

    return rows


# def get_trajectory_paths(config):
#     data_config = config.get("data")
#     dataset_dir = resolve_repo_path(data_config.get("dataset_dir"))
#     return dataset_dir / "train.jsonl", dataset_dir / "eval.jsonl"


def build_dataset(
    category,
    num_states,
    answer_words,
    theta_set,
    epsilon,
    max_turns,
    temperature,
    max_output_tokens,
    api_provider,
    api_model,
    output_path,
    rollout_jsonl_path,
    num_sampled_hypotheses,
    num_samples_per_trajectory,
    filter_threshold,
    refill_rounds,
    likelihood_cache_path,
    question_guidance=None,
    rollout_batch_size=1,
    usage_tracker=None,
    api_usage_path=None,
    max_failures=100,
    include_sampled_candidates=True,
):
    answer_words = list(answer_words)
    theta_set = list(theta_set)

    rows = []
    failures = 0
    rollouts_completed = 0
    rollouts_submitted = 0
    started_at = time.time()

    f = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            with output_path.open("r", encoding="utf-8") as existing_file:
                for line_number, line in enumerate(existing_file, start=1):
                    if not line.strip():
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid JSONL in {output_path} at line {line_number}"
                        ) from error
            if len(rows) > int(num_states):
                raise ValueError(
                    f"Existing dataset {output_path} has {len(rows)} rows, "
                    f"exceeding requested target {num_states}."
                )
            if rows:
                print(
                    f"[resume] loaded_states={len(rows)}/{int(num_states)} "
                    f"from={output_path}",
                    flush=True,
                )
        f = output_path.open("a")

    initial_num_rows = len(rows)

    pbar = tqdm(
        total=int(num_states),
        initial=initial_num_rows,
        desc="Generating API rollout states",
        disable=not sys.stderr.isatty(),
    )

    rollout_batch_size = max(1, int(rollout_batch_size))

    def run_rollout(true_word):
        return build_llm_rollout_states(
            true_word=true_word,
            category=category,
            theta_set=theta_set,
            epsilon=epsilon,
            provider=api_provider,
            model_name=api_model,
            max_turns=max_turns,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            num_samples_per_trajectory=num_samples_per_trajectory,
            rollout_jsonl_path=rollout_jsonl_path,
            num_sampled_hypotheses=num_sampled_hypotheses,
            filter_threshold=filter_threshold,
            refill_rounds=refill_rounds,
            likelihood_cache_path=likelihood_cache_path,
            question_guidance=question_guidance,
            usage_tracker=usage_tracker,
            include_sampled_candidates=include_sampled_candidates,
        )

    try:
        with ThreadPoolExecutor(max_workers=rollout_batch_size) as executor:
            while len(rows) < int(num_states):
                remaining = int(num_states) - len(rows)
                estimated_rollouts = max(
                    1,
                    math.ceil(remaining / max(1, int(num_samples_per_trajectory))),
                )
                current_batch_size = min(rollout_batch_size, estimated_rollouts)
                true_words = [
                    random.choice(answer_words) for _ in range(current_batch_size)
                ]
                rollouts_submitted += current_batch_size
                print(
                    "[batch] "
                    f"states={len(rows)}/{int(num_states)} "
                    f"launching_rollouts={current_batch_size} "
                    f"rollouts_submitted={rollouts_submitted} "
                    f"parallelism={rollout_batch_size}",
                    flush=True,
                )
                futures = [executor.submit(run_rollout, word) for word in true_words]

                for future in as_completed(futures):
                    rollouts_completed += 1
                    try:
                        new_rows = future.result()
                    except Exception as error:
                        failures += 1
                        print(
                            f"Skipping failed rollout {failures}/{max_failures}: "
                            f"{error!r}",
                            flush=True,
                        )
                        if failures >= max_failures:
                            raise RuntimeError(
                                f"Too many failed rollouts: {failures}"
                            ) from error
                        continue

                    if not new_rows or len(rows) >= int(num_states):
                        new_rows = []
                    else:
                        remaining = int(num_states) - len(rows)
                        new_rows = new_rows[:remaining]
                        rows.extend(new_rows)

                        if f is not None:
                            for row in new_rows:
                                f.write(json.dumps(row) + "\n")
                            f.flush()

                        pbar.update(len(new_rows))

                    usage = usage_tracker.snapshot() if usage_tracker is not None else {}
                    elapsed = max(time.time() - started_at, 1.0e-9)
                    new_state_count = len(rows) - initial_num_rows
                    state_rate = new_state_count / elapsed
                    states_remaining = max(int(num_states) - len(rows), 0)
                    eta_seconds = (
                        states_remaining / state_rate if state_rate > 0 else None
                    )
                    eta_text = (
                        f"{eta_seconds / 60.0:.1f}m"
                        if eta_seconds is not None
                        else "unknown"
                    )
                    print(
                        "[progress] "
                        f"states={len(rows)}/{int(num_states)} "
                        f"rollouts_completed={rollouts_completed} "
                        f"failures={failures} "
                        f"elapsed={elapsed / 60.0:.1f}m "
                        f"eta={eta_text} "
                        f"questioner_calls={usage.get('questioner_api_calls', 0)} "
                        f"oracle_calls={usage.get('oracle_api_calls', 0)} "
                        f"estimated_cost=${usage.get('estimated_total_cost_usd', 0.0):.6f}",
                        flush=True,
                    )

    finally:
        pbar.close()
        if f is not None:
            f.close()

    return Dataset.from_list(rows)


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary_path, path)


def build_training_datasets(config, words):
    game_config = config.get("game")
    data_config = config.get("data")

    category = game_config["category"]
    max_turns = int(game_config["max_turns"])
    epsilon = float(game_config["likelihood_epsilon"])
    likelihood_cache_path = game_config["likelihood_cache_path"]
    question_guidance = tq_question_guidance(config)

    api_provider = data_config.get("api_provider")
    api_model = data_config.get("api_model")
    temperature = data_config.get("temperature")
    max_output_tokens = data_config.get("max_output_tokens")
    include_sampled_candidates = bool(
        data_config.get("include_sampled_candidates", True)
    )
    rollout_batch_size = int(data_config.get("rollout_batch_size", 1))
    api_usage_path = data_config.get("api_usage_path")
    initial_usage = None
    if api_usage_path is not None and Path(api_usage_path).exists():
        with Path(api_usage_path).open("r", encoding="utf-8") as handle:
            initial_usage = json.load(handle)
    usage_tracker = ApiUsageTracker(initial_counts=initial_usage)
    usage_snapshot_stop = Event()
    usage_snapshot_thread = None

    if api_usage_path is not None:
        def persist_usage_snapshots():
            while not usage_snapshot_stop.wait(5.0):
                write_json_atomic(api_usage_path, usage_tracker.snapshot())

        usage_snapshot_thread = Thread(
            target=persist_usage_snapshots,
            name="tq-api-usage-snapshot",
            daemon=True,
        )
        usage_snapshot_thread.start()

    if category == "animals":
        default_sampled_train_path = f"data/tq/{DEFAULT_TQ_TRAIN_SAMPLED_STATES}"
        default_sampled_eval_path = f"data/tq/{DEFAULT_TQ_EVAL_SAMPLED_STATES}"
        default_full_train_path = "data/tq/train_full_rollouts.jsonl"
        default_full_eval_path = "data/tq/eval_full_rollouts.jsonl"
    else:
        default_sampled_train_path = f"data/tq/train_sampled_states_200{category}.jsonl"
        default_sampled_eval_path = f"data/tq/eval_sampled_states_200{category}.jsonl"
        default_full_train_path = f"data/tq/train_full_rollouts_{category}.jsonl"
        default_full_eval_path = f"data/tq/eval_full_rollouts_{category}.jsonl"

    sampled_train_path = data_config.get(
        "sampled_train_path",
        default_sampled_train_path,
    )
    sampled_eval_path = data_config.get(
        "sampled_eval_path",
        default_sampled_eval_path,
    )
    full_train_path = data_config.get(
        "full_train_path", default_full_train_path
    )
    full_eval_path = data_config.get(
        "full_eval_path", default_full_eval_path
    )

    try:
        train_dataset = build_dataset(
            category=category,
            num_states=int(data_config.get("train_states")),
            answer_words=words,
            theta_set=words,
            epsilon=epsilon,
            max_turns=max_turns,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            api_provider=api_provider,
            api_model=api_model,
            output_path=sampled_train_path,
            rollout_jsonl_path=full_train_path,
            num_sampled_hypotheses=int(data_config.get("num_sampled_hypotheses")),
            num_samples_per_trajectory=int(data_config.get("num_samples_per_trajectory")),
            filter_threshold=data_config.get("filter_threshold"),
            refill_rounds=data_config.get("refill_rounds"),
            likelihood_cache_path=likelihood_cache_path,
            question_guidance=question_guidance,
            rollout_batch_size=rollout_batch_size,
            usage_tracker=usage_tracker,
            api_usage_path=api_usage_path,
            include_sampled_candidates=include_sampled_candidates,
        )

        eval_dataset = build_dataset(
            category=category,
            num_states=int(data_config.get("eval_states")),
            answer_words=words,
            theta_set=words,
            epsilon=epsilon,
            max_turns=max_turns,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            api_provider=api_provider,
            api_model=api_model,
            output_path=sampled_eval_path,
            rollout_jsonl_path=full_eval_path,
            num_sampled_hypotheses=int(data_config.get("num_sampled_hypotheses")),
            num_samples_per_trajectory=int(data_config.get("num_samples_per_trajectory")),
            filter_threshold=data_config.get("filter_threshold"),
            refill_rounds=data_config.get("refill_rounds"),
            likelihood_cache_path=likelihood_cache_path,
            question_guidance=question_guidance,
            rollout_batch_size=rollout_batch_size,
            usage_tracker=usage_tracker,
            api_usage_path=api_usage_path,
            include_sampled_candidates=include_sampled_candidates,
        )
    finally:
        usage_snapshot_stop.set()
        if usage_snapshot_thread is not None:
            usage_snapshot_thread.join(timeout=10.0)
        usage = usage_tracker.snapshot()
        if api_usage_path is not None:
            write_json_atomic(api_usage_path, usage)

    return train_dataset, eval_dataset, usage


def save_training_datasets(config):
    set_seed(config["game"]["seed"])

    words = resolve_tq_theta_set(config)
    train_dataset, eval_dataset, usage = build_training_datasets(config, words)

    return {
        "num_train_rows": len(train_dataset),
        "num_eval_rows": len(eval_dataset),
        "api_usage": usage,
    }


def get_dataset_paths(config):
    data_config = config.get("data", {})
    dataset_dir = resolve_repo_path(data_config.get("dataset_dir") or "data/tq")
    train_path = data_config.get("sampled_train_path")
    eval_path = data_config.get("sampled_eval_path")

    return (
        resolve_repo_path(train_path)
        if train_path
        else dataset_dir / DEFAULT_TQ_TRAIN_SAMPLED_STATES,
        resolve_repo_path(eval_path)
        if eval_path
        else dataset_dir / DEFAULT_TQ_EVAL_SAMPLED_STATES,
    )


def load_tq_words_file(path):
    path = resolve_repo_path(path)
    with Path(path).open("r", encoding="utf-8") as handle:
        words = json.load(handle)

    if not isinstance(words, list):
        raise ValueError(f"TQ word list must be a JSON list: {path}")

    cleaned = []
    seen = set()
    for word in words:
        word = str(word).strip()
        if not word or word in seen:
            continue
        cleaned.append(word)
        seen.add(word)

    if not cleaned:
        raise ValueError(f"No TQ words loaded from {path}")

    return cleaned


def resolve_tq_theta_set(config):
    data_config = config.get("data", {})
    theta_set_path = data_config.get(
        "theta_set_path",
        DEFAULT_TQ_THETA_SET_PATH,
    )
    return load_tq_words_file(theta_set_path)


def load_training_datasets(config):
    train_path, eval_path = get_dataset_paths(config)

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train dataset: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval dataset: {eval_path}")

    train_dataset = load_dataset("json", data_files=str(train_path), split="train")
    eval_dataset = load_dataset("json", data_files=str(eval_path), split="train")

    return train_dataset, eval_dataset


def remove_empty_sampled_candidates(input_path, output_path):
    from pathlib import Path
    import json

    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.resolve() == output_path.resolve():
        raise ValueError(
            "Input and output paths must be different. Refusing to overwrite in place."
        )

    kept = 0
    removed = 0

    with input_path.open("r") as f_in, output_path.open("w") as f_out:
        for line in f_in:
            if not line.strip():
                continue

            row = json.loads(line)

            try:
                sampled_candidates = json.loads(
                    row.get("sampled_candidates_json", "[]")
                )
            except Exception:
                sampled_candidates = []

            if len(sampled_candidates) == 0:
                removed += 1
                continue

            f_out.write(json.dumps(row) + "\n")
            kept += 1

    print(f"Input:   {input_path}")
    print(f"Output:  {output_path}")
    print(f"Kept:    {kept}")
    print(f"Removed: {removed}")

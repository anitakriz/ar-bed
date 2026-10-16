"""Evaluate BED-LLM full-game baselines.

Example:

    python -m scripts.evaluate_bed_llm_baseline \
      --config config/tq_base.yaml \
      --theta-words data/tq/animals_200.json \
      --eval-words data/tq/animals_test.json \
      --output-dir outputs/bed_llm_base_animals \
      --likelihood-cache outputs/bed_llm_base_animals/cache.json \
      --seed 42 \
      --category animals \
      --max-eval-words 50 \
      --active-threshold 0.01 \
      --zero-inactive-hypotheses

Pass ``--checkpoint`` to evaluate BED-LLM with a trained adapter.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml
from transformers import set_seed

from utils.api_models import ApiUsageTracker
from utils.build_tq_dataset import (
    ANSWER_LABELS,
    extract_beliefs_prompt,
    extract_json_object,
    format_history,
    load_tq_words_file,
    tq_feedback,
)
from utils.llm_backend import (
    load_model_and_tokenizer,
    load_model_and_tokenizer_from_checkpoint,
)
from utils.tq_environment import (
    ask_question as ask_question_schema,
    make_multiturn_prompt,
    make_tq_environment_factory,
)
from utils.tq_full_game_eval import _question_generation_eos_token_ids
from utils.tq_prompting import tq_chat_template_kwargs


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        out[key] = (
            _merge(out[key], value)
            if isinstance(value, dict) and isinstance(out.get(key), dict)
            else copy.deepcopy(value)
        )
    return out


def load_yaml(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    defaults = raw.pop("defaults", []) or []
    config: Dict[str, Any] = {}
    for entry in defaults:
        if entry == "_self_":
            config = _merge(config, raw)
        else:
            base_path = path.parent / str(entry)
            if base_path.suffix not in {".yaml", ".yml"}:
                base_path = base_path.with_suffix(".yaml")
            config = _merge(config, load_yaml(base_path))

    if "_self_" not in defaults:
        config = _merge(config, raw)
    return config


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def normalize_entity(text: Optional[str]) -> str:
    """Case-insensitive exact-match normalization used for belief accuracy."""
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text)).strip().casefold()
    value = value.strip("\"'`")
    return " ".join(value.split())


def entropy(probabilities: Sequence[float], eps: float = 1e-12) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or len(p) == 0:
        return 0.0
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        return 0.0
    total = float(p.sum())
    if total <= 0:
        return 0.0
    p = p / total
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def _normalise_oracle_distribution(dist: Any) -> Optional[Dict[str, Any]]:
    """Normalize one tq_feedback distribution while preserving `solved`."""
    if not isinstance(dist, dict):
        return None
    try:
        values = np.asarray(
            [float(dist[label]) for label in ANSWER_LABELS], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        return None
    total = float(values.sum())
    if total <= 0:
        return None
    values /= total
    return {
        **{label: float(value) for label, value in zip(ANSWER_LABELS, values)},
        "solved": bool(dist.get("solved", False)),
    }


def _blau_increment_from_state(
    C_prev: np.ndarray,
    C_new: np.ndarray,
    true_likelihood: float,
    eps: float = 1e-12,
) -> float:
    """Mirror the evaluator environment's Blau diagnostic increment."""
    return float(
        math.log(max(float(true_likelihood), eps))
        - math.log(max(float(C_new.sum()), eps))
        + math.log(max(float(C_prev.sum()), eps))
    )


def compute_eig(
    distributions_by_hypothesis: Dict[str, Dict[str, float]],
    hypotheses: Sequence[str],
) -> float:
    """Uniform-prior Rao-Blackwellized BED EIG."""
    valid = [theta for theta in hypotheses if theta in distributions_by_hypothesis]
    if len(valid) < 2:
        return 0.0

    matrix = np.asarray(
        [
            [
                float(distributions_by_hypothesis[theta].get(label, 0.0))
                for label in ANSWER_LABELS
            ]
            for theta in valid
        ],
        dtype=np.float64,
    )

    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        return 0.0

    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = matrix[row_sums[:, 0] > 0]
    if len(matrix) < 2:
        return 0.0
    matrix = matrix / matrix.sum(axis=1, keepdims=True)

    conditional_entropy = float(np.mean([entropy(row) for row in matrix]))
    marginal_entropy = entropy(matrix.mean(axis=0))
    return float(marginal_entropy - conditional_entropy)


def _extract_single_tool_call(parsed: Any):
    if not isinstance(parsed, dict):
        return None, None

    calls = parsed.get("tool_calls") or []
    if len(calls) != 1:
        return None, None

    function = calls[0].get("function", {})
    name = function.get("name")
    arguments = function.get("arguments", {})

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None, None

    if not isinstance(arguments, dict):
        return None, None

    return name, arguments


def _ensure_post_tool_generation_marker(rendered: str, messages) -> str:
    """Mirror the continuation handling in current tq_full_game_eval.py."""
    if not messages or messages[-1].get("role") != "tool":
        return rendered

    stripped = rendered.rstrip()
    if stripped.endswith("<|turn>model") or stripped.endswith("<|channel>thought"):
        return rendered

    return rendered + "<|turn>model\n"


def parse_candidate_list(text: str, existing: Sequence[str]) -> List[str]:
    try:
        data = extract_json_object(text)
    except Exception:
        return []

    if not isinstance(data, dict):
        return []

    raw = data.get("candidates", [])
    if not isinstance(raw, list):
        return []

    seen = {normalize_entity(x) for x in existing if normalize_entity(x)}
    output = []
    for candidate in raw:
        candidate = str(candidate).strip()
        key = normalize_entity(candidate)
        if key and key not in seen:
            output.append(candidate)
            seen.add(key)
    return output


def parse_belief_guess(text: str) -> Optional[str]:
    """Parse the evaluation-only greedy belief probe."""
    try:
        data = extract_json_object(text)
        if isinstance(data, dict):
            guess = data.get("guess")
            if guess is not None and str(guess).strip():
                return str(guess).strip()
    except Exception:
        pass

    # Bounded fallback for an otherwise well-behaved greedy decoder.
    stripped = str(text).strip()
    stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
    stripped = re.sub(r"```$", "", stripped).strip()
    if not stripped:
        return None

    # Avoid turning a long explanation into a spurious entity guess.
    first_line = stripped.splitlines()[0].strip()
    first_line = first_line.strip("\"'`")
    if first_line.startswith("<|") or first_line.endswith("|>"):
        return None
    if 0 < len(first_line) <= 120:
        return first_line
    return None


class BEDLLMEvaluator:
    def __init__(
        self,
        *,
        config: Dict[str, Any],
        model,
        tokenizer,
        theta_set: Sequence[str],
        eval_words: Sequence[str],
        output_dir: Path,
    ):
        self.config = copy.deepcopy(config)
        self.model = model
        self.processor = tokenizer
        self.text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

        game = self.config["game"]
        training = self.config["training"]
        evaluation = self.config["evaluation"]
        bed = evaluation["bed_llm"]

        self.category = str(game["category"])
        self.max_turns = int(game["max_turns"])
        self.active_threshold = float(game["active_threshold"])
        self.theta_set = list(theta_set)
        self.eval_words = list(eval_words)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.bed_num_hypotheses = int(bed["num_hypotheses"])
        self.bed_num_questions = int(bed["num_questions"])
        self.bed_temperature = float(bed["temperature"])
        self.bed_consistency_threshold = float(bed["consistency_threshold"])
        self.bed_refill_rounds = int(bed["refill_rounds"])
        self.bed_question_retries = int(bed["question_retries"])
        self.belief_max_new_tokens = int(bed["belief_max_new_tokens"])
        self.guess_max_new_tokens = int(bed["guess_max_new_tokens"])
        self.likelihood_batch_size = int(bed["likelihood_batch_size"])
        self.run_belief_probe = bool(bed.get("run_belief_probe", True))
        self.store_raw_candidate_completions = bool(
            bed["store_raw_candidate_completions"]
        )

        self.question_max_new_tokens = int(training["max_completion_length"])
        self.max_seq_length = int(bed.get("max_seq_length", 32768))
        self.seed = int(evaluation["seed"])
        self.chat_template_kwargs = tq_chat_template_kwargs(self.config)

        if self.likelihood_batch_size < 1:
            raise ValueError("likelihood_batch_size must be >= 1.")
        if self.max_seq_length < 1:
            raise ValueError("evaluation.bed_llm.max_seq_length must be >= 1.")

        if self.text_tokenizer.pad_token_id is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        self.text_tokenizer.padding_side = "left"

        # Reuse the exact current environment implementation. The current factory
        # is Blau-specific, but here Blau is *only* an environment diagnostic;
        # BED-LLM question selection never sees C or the hidden theta universe.
        env_config = copy.deepcopy(self.config)
        env_game = env_config.setdefault("game", {})
        env_game["reward_type"] = "blau"
        env_game["rollout_horizon"] = self.max_turns
        env_game["rollout_gamma"] = 1.0

        if not env_game.get("likelihood_cache_path"):
            raise ValueError(
                "game.likelihood_cache_path is required because the current "
                "TQ environment uses tq_feedback as the synthetic answerer."
            )

        self.likelihood_cache_path = str(env_game["likelihood_cache_path"])
        self.usage_tracker = ApiUsageTracker()
        self.environment_factory = make_tq_environment_factory(
            env_config,
            usage_tracker=self.usage_tracker,
        )
        self.timing = Counter()
        self.counts = Counter()

    # ------------------------------------------------------------------
    # Generic local-model generation
    # ------------------------------------------------------------------

    def _model_context_limit(self) -> Optional[int]:
        model_config = getattr(self.model, "config", None)
        context_limit = getattr(model_config, "max_position_embeddings", None)
        if context_limit is None:
            text_config = getattr(model_config, "text_config", None)
            context_limit = getattr(text_config, "max_position_embeddings", None)
        return int(context_limit) if context_limit is not None else None

    def _check_sequence_length(
        self,
        *,
        prompt_tokens: int,
        extra_tokens: int,
        operation: str,
    ) -> None:
        if prompt_tokens > self.max_seq_length:
            raise RuntimeError(
                f"{operation} prompt has {prompt_tokens} tokens, exceeding the "
                f"configured BED-LLM prompt limit of {self.max_seq_length}. "
                "No prompt was truncated."
            )

        context_limit = self._model_context_limit()
        required_tokens = prompt_tokens + extra_tokens
        if context_limit is not None and required_tokens > context_limit:
            raise RuntimeError(
                f"{operation} requires up to {required_tokens} tokens "
                f"({prompt_tokens} prompt + {extra_tokens} additional), exceeding "
                f"the model context limit of {context_limit}. No prompt was truncated."
            )

    def _render_messages(
        self,
        messages,
        *,
        with_question_tool: bool = False,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        template_kwargs = dict(self.chat_template_kwargs)
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking

        kwargs = dict(tokenize=False, add_generation_prompt=True, **template_kwargs)
        if with_question_tool:
            kwargs["tools"] = [ask_question_schema]

        rendered = self.text_tokenizer.apply_chat_template(messages, **kwargs)
        if with_question_tool:
            rendered = _ensure_post_tool_generation_marker(rendered, messages)
        return rendered

    @torch.no_grad()
    def _generate_rendered(
        self,
        rendered_prompts: Sequence[str],
        *,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        max_new_tokens: int,
        stop_at_question_tool: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        old_padding = self.text_tokenizer.padding_side
        self.text_tokenizer.padding_side = "left"

        try:
            inputs = self.text_tokenizer(
                list(rendered_prompts),
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            attention_mask = inputs.get("attention_mask")
            max_prompt_tokens = (
                int(attention_mask.sum(dim=1).max().item())
                if attention_mask is not None
                else int(inputs["input_ids"].shape[-1])
            )
            self._check_sequence_length(
                prompt_tokens=max_prompt_tokens,
                extra_tokens=int(max_new_tokens),
                operation="BED-LLM generation",
            )
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)

            kwargs = {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": bool(do_sample),
                "num_return_sequences": int(num_return_sequences),
                "pad_token_id": self.text_tokenizer.pad_token_id,
                "eos_token_id": (
                    _question_generation_eos_token_ids(self.text_tokenizer)
                    if stop_at_question_tool
                    else self.text_tokenizer.eos_token_id
                ),
            }
            if do_sample:
                if temperature <= 0:
                    raise ValueError("Sampling temperature must be > 0.")
                kwargs["temperature"] = float(temperature)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = self.model.generate(**inputs, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            self.timing["local_generation_seconds"] += elapsed

            prompt_width = inputs["input_ids"].shape[-1]
            generated = outputs[:, prompt_width:]
            decoded = self.text_tokenizer.batch_decode(
                generated,
                skip_special_tokens=False,
            )

            eos = kwargs["eos_token_id"]
            eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
            rows = []
            for text, ids in zip(decoded, generated):
                token_ids = ids.tolist()
                first_eos = next(
                    (i for i, token_id in enumerate(token_ids) if token_id in eos_ids),
                    None,
                )
                generated_tokens = (
                    len(token_ids) if first_eos is None else first_eos + 1
                )
                rows.append(
                    {
                        "text": text,
                        "generated_tokens": int(generated_tokens),
                        "hit_max_new_tokens": bool(
                            first_eos is None
                            and generated_tokens >= int(max_new_tokens)
                        ),
                    }
                )

            n = int(num_return_sequences)
            groups = [rows[i * n : (i + 1) * n] for i in range(len(rendered_prompts))]
            return groups
        finally:
            self.text_tokenizer.padding_side = old_padding

    def _generate_messages(
        self,
        messages,
        *,
        with_question_tool: bool,
        num_return_sequences: int,
        do_sample: bool,
        temperature: float,
        max_new_tokens: int,
        enable_thinking: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        rendered = self._render_messages(
            messages,
            with_question_tool=with_question_tool,
            enable_thinking=enable_thinking,
        )
        groups = self._generate_rendered(
            [rendered],
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stop_at_question_tool=with_question_tool,
        )
        return groups[0]

    def _filter_candidates(
        self,
        candidates: Sequence[str],
        history: Sequence[Dict[str, str]],
    ) -> List[str]:
        candidates = list(candidates)
        if not candidates:
            return []
        if not history:
            return candidates
        likelihoods_by_question = self._get_model_likelihoods(
            [str(item["question"]) for item in history],
            candidates,
        )
        compatible = []
        for candidate in candidates:
            keep = True
            for item, likelihoods in zip(history, likelihoods_by_question):
                dist = likelihoods.get(candidate)
                observed = str(item["answer"]).strip().lower()
                if (
                    dist is None
                    or float(dist.get(observed, 0.0)) < self.bed_consistency_threshold
                ):
                    keep = False
                    break
            if keep:
                compatible.append(candidate)

        return compatible

    def _refill_bed_candidates(
        self,
        history: Sequence[Dict[str, str]],
        previous_candidates: Sequence[str],
    ):
        start = time.perf_counter()
        candidates = list(dict.fromkeys(previous_candidates))
        stats = {
            "belief_generations": 0,
            "proposed_candidates": 0,
            "accepted_candidates": 0,
            "hit_max_new_tokens": False,
        }

        # Retained hypotheses from previous turns only need to be checked against the newest question/answer.
        if history and candidates:
            candidates = self._filter_candidates(candidates, [history[-1]])

        for _ in range(self.bed_refill_rounds):
            needed = self.bed_num_hypotheses - len(candidates)
            if needed <= 0:
                break

            messages = extract_beliefs_prompt(
                history=history,
                active_candidates=candidates,
                category=self.category,
                num_new_hypotheses=needed,
            )
            rows = self._generate_messages(
                messages,
                with_question_tool=False,
                num_return_sequences=1,
                do_sample=True,
                temperature=self.bed_temperature,
                max_new_tokens=self.belief_max_new_tokens,
                enable_thinking=False,
            )
            stats["belief_generations"] += 1
            if not rows:
                continue

            row = rows[0]
            stats["hit_max_new_tokens"] |= bool(row["hit_max_new_tokens"])
            proposed = parse_candidate_list(row["text"], existing=candidates)
            stats["proposed_candidates"] += len(proposed)

            compatible = (
                self._filter_candidates(proposed, history) if history else proposed
            )
            for candidate in compatible:
                if normalize_entity(candidate) not in {
                    normalize_entity(x) for x in candidates
                }:
                    candidates.append(candidate)
                    stats["accepted_candidates"] += 1
                if len(candidates) >= self.bed_num_hypotheses:
                    break

        self.timing["bed_belief_refill_seconds"] += time.perf_counter() - start
        self.counts["belief_generations"] += stats["belief_generations"]
        self.counts["proposed_hypotheses"] += stats["proposed_candidates"]
        self.counts["accepted_hypotheses"] += stats["accepted_candidates"]
        return candidates[: self.bed_num_hypotheses], stats

    # ------------------------------------------------------------------
    # B. Candidate question generation
    # ------------------------------------------------------------------

    def _make_bed_question_messages(
        self,
        history: Sequence[Dict[str, str]],
        turn: int,
        candidates: Sequence[str],
    ):
        messages = make_multiturn_prompt(
            history=history,
            category=self.category,
            turn=turn,
            max_turns=self.max_turns,
            config=self.config,
        )

        # Keep the current trainer/environment message structure, but expose only
        # the *model-sampled BED hypothesis set* to the question generator.
        if candidates:
            candidate_text = json.dumps(list(candidates), ensure_ascii=False)
            messages[1]["content"] += (
                "\n\nCurrent BED candidate hypotheses sampled from your belief:\n"
                f"{candidate_text}\n\n"
                "Use these candidates to propose a question that is useful for "
                "distinguishing among plausible targets. The true target is not "
                "guaranteed to be in this finite candidate set."
            )

        return messages

    def _sample_candidate_questions(
        self,
        history: Sequence[Dict[str, str]],
        turn: int,
        candidates: Sequence[str],
    ) -> List[Dict[str, Any]]:
        parsed_questions: List[Dict[str, Any]] = []
        seen = set()

        for _ in range(self.bed_question_retries):
            needed = self.bed_num_questions - len(parsed_questions)
            if needed <= 0:
                break

            messages = self._make_bed_question_messages(history, turn, candidates)

            rows = self._generate_messages(
                messages,
                with_question_tool=True,
                num_return_sequences=needed,
                do_sample=True,
                temperature=self.bed_temperature,
                max_new_tokens=self.question_max_new_tokens,
            )
            self.counts["question_sequences_generated"] += len(rows)

            for row in rows:
                try:
                    parsed = self.text_tokenizer.parse_response(row["text"])
                except (TypeError, ValueError):
                    self.counts["question_parse_failures"] += 1
                    continue
                tool_name, tool_args = _extract_single_tool_call(parsed)
                if tool_name != "ask_question":
                    continue
                question = (
                    tool_args.get("question") if isinstance(tool_args, dict) else None
                )
                if not isinstance(question, str) or not question.strip():
                    continue
                question = question.strip()
                key = " ".join(question.casefold().split())
                if key in seen:
                    continue
                seen.add(key)
                item = {
                    "question": question,
                    "source": "sampled",
                    "generated_tokens": row["generated_tokens"],
                    "hit_max_new_tokens": row["hit_max_new_tokens"],
                }
                if self.store_raw_candidate_completions:
                    item["raw_completion"] = row["text"]
                    item["parsed"] = parsed
                parsed_questions.append(item)
                if len(parsed_questions) >= self.bed_num_questions:
                    break

        self.counts["valid_unique_candidate_questions"] += len(parsed_questions)
        return parsed_questions

    # ------------------------------------------------------------------
    # C. Local-model p(y | theta, x)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_model_likelihoods(
        self,
        questions: Sequence[str],
        candidates: Sequence[str],
    ) -> List[Dict[str, Dict[str, float]]]:
        """Score yes/no/maybe using the evaluated model's own token probabilities."""
        questions = list(questions)
        candidates = list(candidates)
        if not questions or not candidates:
            return [{} for _ in questions]

        answer_labels = [str(label).strip().lower() for label in ANSWER_LABELS]
        log_scores = np.full(
            (len(questions), len(candidates), len(answer_labels)),
            -np.inf,
            dtype=np.float64,
        )

        scoring_rows = []
        for q_idx, question in enumerate(questions):
            for c_idx, candidate in enumerate(candidates):
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Given the ground truth, answer the question. "
                            "Reply with exactly one of: yes, no, maybe. "
                            "Use maybe if the answer could reasonably be either yes or no."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (f"Ground Truth: {candidate}\nQuestion: {question}"),
                    },
                ]
                prompt_text = self.text_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.chat_template_kwargs,
                )
                prompt_ids = self.text_tokenizer.encode(
                    prompt_text,
                    add_special_tokens=False,
                )

                for label_idx, label in enumerate(answer_labels):
                    label_ids = self.text_tokenizer.encode(
                        label, add_special_tokens=False
                    )
                    if not label_ids:
                        raise ValueError(f"Could not tokenize answer label {label!r}.")
                    self._check_sequence_length(
                        prompt_tokens=len(prompt_ids) + len(label_ids),
                        extra_tokens=0,
                        operation="BED-LLM likelihood scoring",
                    )
                    scoring_rows.append(
                        {
                            "question_index": q_idx,
                            "candidate_index": c_idx,
                            "label_index": label_idx,
                            "input_ids": prompt_ids + label_ids,
                            "label_start": len(prompt_ids),
                            "label_length": len(label_ids),
                        }
                    )

        device = next(self.model.parameters()).device
        start_time = time.perf_counter()

        for start in range(0, len(scoring_rows), self.likelihood_batch_size):
            batch_rows = scoring_rows[start : start + self.likelihood_batch_size]
            max_length = max(len(row["input_ids"]) for row in batch_rows)

            input_ids = torch.full(
                (len(batch_rows), max_length),
                fill_value=self.text_tokenizer.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros(
                (len(batch_rows), max_length),
                dtype=torch.long,
                device=device,
            )
            padded_label_starts = []

            for row_idx, row in enumerate(batch_rows):
                row_ids = torch.tensor(
                    row["input_ids"], dtype=torch.long, device=device
                )
                pad_len = max_length - len(row["input_ids"])
                input_ids[row_idx, pad_len:] = row_ids
                attention_mask[row_idx, pad_len:] = 1
                padded_label_starts.append(pad_len + row["label_start"])

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

            for row_idx, row in enumerate(batch_rows):
                label_start = padded_label_starts[row_idx]
                label_end = label_start + row["label_length"]
                if label_start <= 0:
                    raise ValueError("Answer label has no preceding token to score it.")

                prediction_positions = torch.arange(
                    label_start - 1,
                    label_end - 1,
                    device=outputs.logits.device,
                )
                target_ids = input_ids[row_idx, label_start:label_end]
                answer_logits = outputs.logits[
                    row_idx,
                    prediction_positions,
                    :,
                ].float()
                log_probs = torch.log_softmax(answer_logits, dim=-1)
                token_log_probs = log_probs.gather(
                    dim=-1,
                    index=target_ids.unsqueeze(-1),
                ).squeeze(-1)
                score = float(token_log_probs.sum().item())

                log_scores[
                    row["question_index"],
                    row["candidate_index"],
                    row["label_index"],
                ] = score

            del outputs, input_ids, attention_mask

        self.timing["bed_model_likelihood_seconds"] += time.perf_counter() - start_time

        output: List[Dict[str, Dict[str, float]]] = []
        for q_idx in range(len(questions)):
            question_likelihoods = {}
            for c_idx, candidate in enumerate(candidates):
                scores = log_scores[q_idx, c_idx]
                if not np.all(np.isfinite(scores)):
                    probs = np.full(len(answer_labels), 1.0 / len(answer_labels))
                else:
                    shifted = scores - np.max(scores)
                    probs = np.exp(shifted)
                    probs = probs / probs.sum()
                question_likelihoods[candidate] = {
                    label: float(probability)
                    for label, probability in zip(answer_labels, probs)
                }
            output.append(question_likelihoods)

        return output

    # ------------------------------------------------------------------
    # D. EIG selection
    # ------------------------------------------------------------------

    def _select_bed_question(
        self,
        history: Sequence[Dict[str, str]],
        turn: int,
        candidates: Sequence[str],
    ) -> Dict[str, Any]:
        candidates = list(candidates)

        # EIG is not meaningful with fewer than two hypotheses. In that case,
        # fall back to free question generation from the observed history only.
        # Passing an empty candidate set deliberately prevents the collapsed BED
        # hypothesis set from being appended to the generation prompt.
        if len(candidates) < 2:
            sampled = self._sample_candidate_questions(history, turn, [])
            if not sampled:
                raise RuntimeError(
                    "BED has fewer than two hypotheses and fallback question generation failed."
                )
            chosen = sampled[0]
            return {
                "question": chosen["question"],
                "source": "insufficient_hypotheses_fallback",
                "selected_eig": None,
                "candidate_questions": sampled,
                "fallback": "fewer_than_two_bed_hypotheses",
            }

        question_rows = self._sample_candidate_questions(history, turn, candidates)

        if not question_rows:
            raise RuntimeError("No valid BED candidate questions were generated.")

        start = time.perf_counter()
        likelihoods_by_question = self._get_model_likelihoods(
            [item["question"] for item in question_rows],
            candidates,
        )
        self.timing["bed_eig_total_seconds"] += time.perf_counter() - start

        scored = []
        for item, likelihoods in zip(question_rows, likelihoods_by_question):
            scored.append(
                {
                    **item,
                    "eig": float(compute_eig(likelihoods, candidates)),
                }
            )

        selected = max(scored, key=lambda row: row["eig"])
        return {
            "question": selected["question"],
            "source": selected["source"],
            "selected_eig": float(selected["eig"]),
            "candidate_questions": scored,
            "fallback": None,
        }

    def _belief_probe_messages(self, history: Sequence[Dict[str, str]]):
        return [
            {
                "role": "system",
                "content": (
                    "Based on all questions and answers so far, what is your single best guess for the hidden animal? Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Category: {self.category}\n\n"
                    "Observed history:\n"
                    f"{format_history(history)}\n\n"
                    'STRICT output format: {"guess": "single entity"}'
                ),
            },
        ]

    def _greedy_belief_probe(
        self,
        history: Sequence[Dict[str, str]],
        true_word: str,
        turn: int,
    ) -> Dict[str, Any]:
        messages = self._belief_probe_messages(history)
        start = time.perf_counter()
        rows = self._generate_messages(
            messages,
            with_question_tool=False,
            num_return_sequences=1,
            do_sample=False,
            temperature=1.0,
            max_new_tokens=self.guess_max_new_tokens,
            enable_thinking=False,
        )
        self.timing["belief_probe_seconds"] += time.perf_counter() - start

        raw = rows[0]["text"] if rows else ""
        guess = parse_belief_guess(raw)
        correct = normalize_entity(guess) == normalize_entity(true_word)
        self.counts["belief_probes"] += 1
        self.counts["belief_probe_correct"] += int(correct)

        return {
            "turn": int(turn),
            "guess": guess,
            "correct": bool(correct),
            "raw_completion": raw,
        }

    def _execute_question_with_split_timing(
        self,
        env,
        question: str,
    ):
        """
        Execute the selected question while separating real interaction cost from
        hidden theta-set diagnostics.

        `oracle_answer_seconds` is the cost of obtaining the answer for the true
        target only. `theta_diagnostic_seconds` is evaluation-only work used to
        update C / active-count / Blau diagnostics over the hidden theta set.
        """
        if env.solved:
            raise ValueError("Game already solved; no further question is allowed.")
        if len(env.step_rewards) >= env.horizon:
            env.invalid = True
            raise ValueError("Rollout horizon already reached.")

        question = str(question).strip()
        if not question:
            env.invalid = True
            raise ValueError("Question must be non-empty.")

        # --------------------------------------------------------------
        # 1) Actual interaction: answer the selected question for theta*.
        #    This is part of the BED-LLM interaction runtime.
        # --------------------------------------------------------------
        answer_start = time.perf_counter()
        try:
            rows = tq_feedback(
                questions=[question],
                ground_truths=[env.true_word],
                cache_path=env.likelihood_cache_path,
                usage_tracker=self.usage_tracker,
                track_cache_status=True,
            )
            raw_true_dist = rows[0][0] if rows and rows[0] else None
            true_dist = _normalise_oracle_distribution(raw_true_dist)
            if true_dist is None:
                raise ValueError(
                    f"Missing/malformed true-word likelihood for "
                    f"question={question!r}, true_word={env.true_word!r}."
                )
        except Exception:
            env.invalid = True
            raise
        oracle_answer_seconds = time.perf_counter() - answer_start
        self.timing["oracle_answer_seconds"] += oracle_answer_seconds

        env.last_solved = bool(true_dist.get("solved", False))
        env.solved = env.solved or env.last_solved
        observed_answer = max(
            ANSWER_LABELS,
            key=lambda label: float(true_dist[label]),
        )
        true_likelihood = float(true_dist[observed_answer])

        # --------------------------------------------------------------
        # 2) Hidden evaluator diagnostics over theta_set.
        #    This is NOT BED-LLM inference and is timed separately.
        # --------------------------------------------------------------
        diagnostic_start = time.perf_counter()
        try:
            C_prev = np.asarray(env.C, dtype=float).copy()
            active_indices = np.flatnonzero(C_prev > env.active_threshold)
            active_candidates = [env.contrastive_set[index] for index in active_indices]

            diagnostic_likelihoods: Dict[str, Dict[str, Any]] = {}

            # Reuse the already-computed true-target distribution when theta* is
            # also in the hidden evaluator set, so we do not pay for it twice.
            if env.true_word in active_candidates:
                diagnostic_likelihoods[env.true_word] = true_dist

            remaining_candidates = [
                candidate
                for candidate in active_candidates
                if candidate != env.true_word
            ]
            if remaining_candidates:
                diagnostic_rows = tq_feedback(
                    questions=[question],
                    ground_truths=remaining_candidates,
                    cache_path=env.likelihood_cache_path,
                    usage_tracker=self.usage_tracker,
                    track_cache_status=True,
                )
                raw_row = diagnostic_rows[0] if diagnostic_rows else []
                for candidate, raw_dist in zip(remaining_candidates, raw_row):
                    dist = _normalise_oracle_distribution(raw_dist)
                    if dist is not None:
                        diagnostic_likelihoods[candidate] = dist

            if not active_candidates:
                step_reward = 0.0
            else:
                valid_indices = []
                likelihood_values = []
                for index, candidate in zip(active_indices, active_candidates):
                    dist = diagnostic_likelihoods.get(candidate)
                    if dist is None:
                        continue
                    value = float(dist.get(observed_answer, np.nan))
                    if np.isfinite(value) and value >= 0:
                        valid_indices.append(int(index))
                        likelihood_values.append(value)

                if not valid_indices:
                    step_reward = 0.0
                else:
                    C_new = C_prev.copy()
                    C_new[np.asarray(valid_indices, dtype=int)] *= np.asarray(
                        likelihood_values,
                        dtype=float,
                    )
                    if bool(
                        self.config["game"].get(
                            "zero_inactive_hypotheses",
                            False,
                        )
                    ):
                        C_new[C_new <= env.active_threshold] = 0.0
                    step_reward = _blau_increment_from_state(
                        C_prev=C_prev,
                        C_new=C_new,
                        true_likelihood=true_likelihood,
                    )
                    env.C = C_new
        except Exception:
            env.invalid = True
            raise
        theta_diagnostic_seconds = time.perf_counter() - diagnostic_start
        self.timing["theta_diagnostic_seconds"] += theta_diagnostic_seconds

        # Mirror env.ask_question bookkeeping exactly enough for the existing
        # metrics / reward accessors, without folding diagnostic cost into the
        # actual answer timer.
        env.history.append({"question": question, "answer": observed_answer})
        env.step_rewards.append(float(step_reward))
        debug = {
            "question": question,
            "answer": observed_answer,
            "reward": float(step_reward),
            "active_count": int(np.sum(np.asarray(env.C) > env.active_threshold)),
            "sum_C": float(np.sum(env.C)),
            "solved": bool(env.last_solved),
        }
        env.step_debug.append(debug)

        status = "solved" if env.last_solved else "continue"
        tool_result = f"Answer: {observed_answer}\nStatus: {status}"
        return (
            tool_result,
            debug,
            {
                "oracle_answer_seconds": float(oracle_answer_seconds),
                "theta_diagnostic_seconds": float(theta_diagnostic_seconds),
            },
        )

    # ------------------------------------------------------------------
    # Environment and game loop
    # ------------------------------------------------------------------

    def _new_game(self, true_word: str) -> Dict[str, Any]:
        env = self.environment_factory()
        env.horizon = int(self.max_turns)
        env.reset(
            true_word=str(true_word),
            history_json="[]",
            contrastive_set_json=json.dumps(self.theta_set),
            C_json=json.dumps([1.0] * len(self.theta_set)),
        )

        return {
            "true_word": str(true_word),
            "env": env,
            "history": [],
            "bed_candidates": [],
            "steps": [],
            "belief_probes": [],
            "stop_reason": "max_turns",
            "finished": False,
        }

    def _run_game(self, true_word: str) -> Dict[str, Any]:
        game = self._new_game(true_word)
        env = game["env"]
        # t=0 probe: greedy decoding of belief state
        if self.run_belief_probe:
            game["belief_probes"].append(
                self._greedy_belief_probe([], true_word=true_word, turn=0)
            )
        for turn in range(1, self.max_turns + 1):
            if game["finished"]:
                break

            turn_start = time.perf_counter()
            history = list(game["history"])

            # BED-LLM policy wall time: hypothesis update/refill + question
            # proposal + local-model likelihood scoring + EIG selection.
            # This intentionally excludes the actual answer, theta diagnostics,
            # and the evaluation-only greedy belief probe.
            bed_policy_start = time.perf_counter()

            # refill up to N candidates , retries up to R rounds, filtering by history
            candidates, refill_stats = self._refill_bed_candidates(
                history=history,
                previous_candidates=game["bed_candidates"],
            )
            game["bed_candidates"] = candidates

            try:
                selection = self._select_bed_question(
                    history=history,
                    turn=turn,
                    candidates=candidates,
                )
            except Exception as exc:
                bed_policy_seconds = time.perf_counter() - bed_policy_start
                self.timing["bed_policy_seconds"] += bed_policy_seconds
                game["stop_reason"] = "bed_question_selection_failure"
                game["finished"] = True
                game["steps"].append(
                    {
                        "turn": turn,
                        "question": None,
                        "answer": None,
                        "bed_candidates": list(candidates),
                        "refill_stats": refill_stats,
                        "bed_policy_seconds": float(bed_policy_seconds),
                        "failure_reason": repr(exc),
                    }
                )
                break

            bed_policy_seconds = time.perf_counter() - bed_policy_start
            self.timing["bed_policy_seconds"] += bed_policy_seconds
            question = selection["question"]

            # Execute the selected question with the real answer separated from
            # the hidden full-theta-set diagnostic update.
            try:
                tool_result, debug, split_times = (
                    self._execute_question_with_split_timing(
                        env,
                        question,
                    )
                )
            except Exception as exc:
                game["stop_reason"] = "environment_failure"
                game["finished"] = True
                game["steps"].append(
                    {
                        "turn": turn,
                        "question": question,
                        "answer": None,
                        "bed_candidates": list(candidates),
                        "refill_stats": refill_stats,
                        **selection,
                        "bed_policy_seconds": float(bed_policy_seconds),
                        "failure_reason": repr(exc),
                    }
                )
                break

            oracle_answer_seconds = float(split_times["oracle_answer_seconds"])
            theta_diagnostic_seconds = float(split_times["theta_diagnostic_seconds"])
            bed_llm_runtime_seconds = bed_policy_seconds + oracle_answer_seconds
            self.timing["bed_llm_runtime_seconds"] += bed_llm_runtime_seconds
            answer = str(debug["answer"])
            solved = bool(debug.get("solved", False))

            game["history"].append({"question": question, "answer": answer})

            step = {
                "turn": turn,
                "question": question,
                "answer": answer,
                "tool_response": str(tool_result),
                "solved": solved,
                "bed_candidates": list(candidates),
                "num_bed_candidates": len(candidates),
                "refill_stats": refill_stats,
                "selection_source": selection["source"],
                "selected_eig": selection["selected_eig"],
                "candidate_questions": selection["candidate_questions"],
                "fallback": selection["fallback"],
                # Hidden-evaluator diagnostics only; never shown to BED policy.
                "active_count": int(debug["active_count"]),
                "sum_C": float(debug["sum_C"]),
                "environment_blau_reward": float(debug["reward"]),
                # Clean, non-overlapping high-level timing components.
                "bed_policy_seconds": float(bed_policy_seconds),
                "oracle_answer_seconds": float(oracle_answer_seconds),
                "theta_diagnostic_seconds": float(theta_diagnostic_seconds),
                "bed_llm_runtime_seconds": float(bed_llm_runtime_seconds),
                "turn_wall_seconds_before_belief_probe": float(
                    time.perf_counter() - turn_start
                ),
            }
            game["steps"].append(step)

            if self.run_belief_probe:
                # Evaluation-only greedy belief extraction after observing y_t.
                belief_probe_wall_start = time.perf_counter()
                game["belief_probes"].append(
                    self._greedy_belief_probe(
                        game["history"],
                        true_word=true_word,
                        turn=turn,
                    )
                )
                step["belief_probe_wall_seconds"] = float(
                    time.perf_counter() - belief_probe_wall_start
                )
            else:
                step["belief_probe_wall_seconds"] = 0.0
            step["turn_wall_seconds"] = float(time.perf_counter() - turn_start)

            belief_text = ""
            if self.run_belief_probe and game["belief_probes"]:
                belief_text = (
                    f" belief={game['belief_probes'][-1]['guess']!r}"
                    f" belief_correct={game['belief_probes'][-1]['correct']}"
                )
            print(
                f"[{true_word}] turn={turn:02d} "
                f"N={len(candidates):02d} "
                f"EIG={selection['selected_eig']} "
                f"Q={question!r} A={answer!r} solved={solved}"
                f"{belief_text}",
                flush=True,
            )

            if solved:
                game["stop_reason"] = "solved"
                game["finished"] = True
                break

        final_active = int(np.sum(np.asarray(env.C) > self.active_threshold))
        valid_steps = [step for step in game["steps"] if step.get("question")]

        return {
            "true_word": game["true_word"],
            "solved": game["stop_reason"] == "solved",
            # A trajectory is valid if the game ended normally: either it solved
            # the target or simply exhausted the turn budget. BED-selection and
            # environment failures terminate the current game and are invalid.
            "valid_trajectory": game["stop_reason"] in {"solved", "max_turns"},
            "stop_reason": game["stop_reason"],
            "num_turns": len(valid_steps),
            "final_active_hypotheses": final_active,
            "trajectory_reward": float(env._get_blau_reward(gamma_override=1.0)),
            "belief_probes": game["belief_probes"],
            "steps": game["steps"],
        }

    # ------------------------------------------------------------------
    # Metrics and outputs
    # ------------------------------------------------------------------

    def _trajectory_metrics(self, results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute the trajectory-derived metrics for an arbitrary subset of games."""
        n = len(results)
        if n == 0:
            return {
                "num_games": 0,
                "solve_rate": 0.0,
                "avg_turns_solved": 0.0,
                "median_turns_solved": 0.0,
                "avg_num_turns": 0.0,
                "avg_final_active_hypotheses": 0.0,
                "normalized_active_auc": 0.0,
                "repeated_question_rate": 0.0,
                "final_belief_accuracy": 0.0,
                "belief_accuracy_by_turn": {},
                "stop_reason_counts": {},
            }

        solved = [r for r in results if r["solved"]]
        solved_turns = [r["num_turns"] for r in solved]

        # Belief accuracy among games for which a probe exists at that turn.
        by_turn: Dict[int, List[bool]] = {}
        for result in results:
            for probe in result["belief_probes"]:
                by_turn.setdefault(int(probe["turn"]), []).append(
                    bool(probe["correct"])
                )

        belief_accuracy_by_turn = {
            str(turn): {
                "accuracy": float(np.mean(values)),
                "num_games": int(len(values)),
            }
            for turn, values in sorted(by_turn.items())
        }

        final_belief_correct = [
            bool(result["belief_probes"][-1]["correct"])
            for result in results
            if result["belief_probes"]
        ]

        repeated = 0
        total_questions = 0
        auc_values = []
        theta_size = max(1, len(self.theta_set))

        for result in results:
            seen = set()
            last_active = float(theta_size)
            trajectory = [last_active]
            steps_by_turn = {int(step["turn"]): step for step in result["steps"]}
            solved_turn = result["num_turns"] if result["solved"] else None

            for turn in range(1, self.max_turns + 1):
                step = steps_by_turn.get(turn)
                if step is not None and step.get("question"):
                    key = " ".join(step["question"].casefold().split())
                    total_questions += 1
                    if key in seen:
                        repeated += 1
                    seen.add(key)
                    if step.get("active_count") is not None:
                        last_active = float(step["active_count"])

                value = (
                    0.0
                    if solved_turn is not None and turn > solved_turn
                    else last_active
                )
                trajectory.append(value)

            auc_values.append(float(np.mean(trajectory) / theta_size))

        return {
            "num_games": n,
            "solve_rate": float(len(solved) / n),
            "avg_turns_solved": float(np.mean(solved_turns)) if solved_turns else 0.0,
            "median_turns_solved": float(np.median(solved_turns))
            if solved_turns
            else 0.0,
            "avg_num_turns": float(np.mean([r["num_turns"] for r in results])),
            "avg_final_active_hypotheses": float(
                np.mean([r["final_active_hypotheses"] for r in results])
            ),
            "normalized_active_auc": float(np.mean(auc_values)),
            "repeated_question_rate": (
                float(repeated / total_questions) if total_questions else 0.0
            ),
            "final_belief_accuracy": (
                float(np.mean(final_belief_correct)) if final_belief_correct else 0.0
            ),
            "belief_accuracy_by_turn": belief_accuracy_by_turn,
            "stop_reason_counts": dict(Counter(r["stop_reason"] for r in results)),
        }

    def _split_timing_summary(
        self,
        results: Sequence[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Aggregate the clean per-step timing components for a game subset."""
        n = len(results)
        bed_policy = 0.0
        oracle_answer = 0.0
        theta_diagnostic = 0.0
        belief_probe_turns = 0.0
        turn_wall = 0.0

        for result in results:
            for step in result.get("steps", []):
                bed_policy += float(step.get("bed_policy_seconds", 0.0) or 0.0)
                oracle_answer += float(step.get("oracle_answer_seconds", 0.0) or 0.0)
                theta_diagnostic += float(
                    step.get("theta_diagnostic_seconds", 0.0) or 0.0
                )
                belief_probe_turns += float(
                    step.get("belief_probe_wall_seconds", 0.0) or 0.0
                )
                turn_wall += float(step.get("turn_wall_seconds", 0.0) or 0.0)

        bed_llm_runtime = bed_policy + oracle_answer
        denom = max(n, 1)
        return {
            "bed_policy_seconds": float(bed_policy),
            "oracle_answer_seconds": float(oracle_answer),
            # Primary number for the requested comparison: BED policy inference
            # plus obtaining the actual answer, excluding hidden diagnostics and
            # evaluation-only greedy belief probes.
            "bed_llm_runtime_seconds": float(bed_llm_runtime),
            "theta_diagnostic_seconds": float(theta_diagnostic),
            "belief_probe_turn_seconds": float(belief_probe_turns),
            "recorded_turn_wall_seconds": float(turn_wall),
            "bed_policy_seconds_per_game": float(bed_policy / denom),
            "oracle_answer_seconds_per_game": float(oracle_answer / denom),
            "bed_llm_runtime_seconds_per_game": float(bed_llm_runtime / denom),
            "theta_diagnostic_seconds_per_game": float(theta_diagnostic / denom),
        }

    def _summarize(self, results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(results)
        if n == 0:
            return {
                "num_games": 0,
                "api_usage": self.usage_tracker.snapshot(),
            }

        # Preserve every existing all-trajectory metric exactly as before.
        overall = self._trajectory_metrics(results)

        # Normal game endings are valid. The two handled failure modes stop the
        # current game immediately and are excluded from these valid-only metrics.
        valid_results = [
            r
            for r in results
            if bool(
                r.get(
                    "valid_trajectory",
                    r.get("stop_reason") in {"solved", "max_turns"},
                )
            )
        ]
        invalid_results = [r for r in results if r not in valid_results]
        valid = self._trajectory_metrics(valid_results)
        inference_timing = self._split_timing_summary(results)
        valid_inference_timing = self._split_timing_summary(valid_results)

        summary = dict(overall)
        summary.update(
            {
                "num_valid_games": int(len(valid_results)),
                "num_invalid_games": int(len(invalid_results)),
                "valid_trajectory_rate": float(len(valid_results) / n),
                "invalid_trajectory_rate": float(len(invalid_results) / n),
                # Same trajectory metrics, recomputed after dropping invalid games.
                "valid_solve_rate": valid["solve_rate"],
                "valid_avg_turns_solved": valid["avg_turns_solved"],
                "valid_median_turns_solved": valid["median_turns_solved"],
                "valid_avg_num_turns": valid["avg_num_turns"],
                "valid_avg_final_active_hypotheses": valid[
                    "avg_final_active_hypotheses"
                ],
                "valid_normalized_active_auc": valid["normalized_active_auc"],
                "valid_repeated_question_rate": valid["repeated_question_rate"],
                "valid_final_belief_accuracy": valid["final_belief_accuracy"],
                "valid_belief_accuracy_by_turn": valid["belief_accuracy_by_turn"],
                "valid_stop_reason_counts": valid["stop_reason_counts"],
                "invalid_stop_reason_counts": dict(
                    Counter(r["stop_reason"] for r in invalid_results)
                ),
                # Clean wall-clock decomposition. `bed_llm_runtime_seconds` is the
                # reportable BED-LLM interaction time requested here: policy +
                # actual answer only. It excludes theta diagnostics and belief probes.
                "inference_timing": inference_timing,
                "valid_inference_timing": valid_inference_timing,
                # These legacy counters/timers are accumulated globally during evaluation,
                # so they cannot be recomputed for valid games only without changing
                # how the evaluator records per-game diagnostics.
                "timing": {key: float(value) for key, value in self.timing.items()},
                "counts": {key: int(value) for key, value in self.counts.items()},
                "api_usage": self.usage_tracker.snapshot(),
            }
        )
        return summary

    @staticmethod
    def _write_json_atomic(path: Path, payload: Any) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        temporary_path.replace(path)

    def _save(self, results: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
        trajectories_path = self.output_dir / "trajectories.json"
        summary_path = self.output_dir / "summary.json"
        api_usage_path = self.output_dir / "api_usage.json"
        steps_path = self.output_dir / "steps.jsonl"
        belief_csv_path = self.output_dir / "belief_accuracy_by_turn.csv"

        self._write_json_atomic(
            trajectories_path,
            {
                "method": "bed_llm",
                "category": self.category,
                "max_turns": self.max_turns,
                "theta_size": len(self.theta_set),
                "eval_seed": self.seed,
                "active_threshold": self.active_threshold,
                "information_reward_mode": "realized",
                "zero_inactive_hypotheses": bool(
                    self.config["game"].get("zero_inactive_hypotheses", False)
                ),
                "eval_words": self.eval_words,
                "api_usage": self.usage_tracker.snapshot(),
                "games": list(results),
            },
        )
        self._write_json_atomic(summary_path, summary)
        self._write_json_atomic(api_usage_path, self.usage_tracker.snapshot())

        with steps_path.open("w", encoding="utf-8") as handle:
            for game in results:
                for step in game["steps"]:
                    row = {"true_word": game["true_word"], **step}
                    handle.write(
                        json.dumps(row, ensure_ascii=False, default=str) + "\n"
                    )

        with belief_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["turn", "accuracy", "num_games"]
            )
            writer.writeheader()
            for turn, row in summary.get("belief_accuracy_by_turn", {}).items():
                writer.writerow(
                    {
                        "turn": int(turn),
                        "accuracy": row["accuracy"],
                        "num_games": row["num_games"],
                    }
                )

    def _load_resume_results(self) -> List[Dict[str, Any]]:
        trajectories_path = self.output_dir / "trajectories.json"
        if not trajectories_path.exists():
            print(
                "BED-LLM resume enabled, but no trajectories.json was found; "
                "starting from game 1.",
                flush=True,
            )
            return []

        payload = json.loads(trajectories_path.read_text(encoding="utf-8"))
        if payload.get("eval_words") != self.eval_words:
            raise ValueError(
                "Cannot resume: trajectories.json uses a different eval word list."
            )

        results = list(payload.get("games", []))
        completed_words = [result.get("true_word") for result in results]
        if completed_words != self.eval_words[: len(completed_words)]:
            raise ValueError(
                "Cannot resume: completed games are not a prefix of the eval word list."
            )

        print(
            "BED-LLM resume enabled: loaded "
            f"{len(results)}/{len(self.eval_words)} completed games from "
            f"{trajectories_path}; {len(self.eval_words) - len(results)} remaining.",
            flush=True,
        )
        return results

    @torch.no_grad()
    def evaluate(self, *, resume: bool = False) -> Dict[str, Any]:
        was_training = self.model.training
        self.model.eval()
        self.timing.clear()
        self.counts.clear()

        set_all_seeds(self.seed)
        start = time.perf_counter()
        results = self._load_resume_results() if resume else []

        try:
            for index, true_word in enumerate(
                self.eval_words[len(results) :], start=len(results) + 1
            ):
                print(
                    f"\n=== BED-LLM game {index}/{len(self.eval_words)}: {true_word} ===",
                    flush=True,
                )

                result = self._run_game(true_word)
                results.append(result)

                # Crash-safe checkpoint: persist all completed games immediately.
                partial_summary = self._summarize(results)
                self._save(results, partial_summary)
                api_usage = self.usage_tracker.snapshot()
                print(
                    "BED-LLM cumulative API usage: "
                    f"hits={api_usage.get('oracle_cache_hits', 0)}, "
                    f"misses={api_usage.get('oracle_cache_misses', 0)}, "
                    f"api_calls={api_usage.get('oracle_api_calls', 0)}, "
                    "estimated_cost="
                    f"${float(api_usage.get('estimated_total_cost_usd', 0.0)):.6f}",
                    flush=True,
                )
        finally:
            if was_training:
                self.model.train()

        self.timing["total_wall_seconds"] = time.perf_counter() - start
        self.timing["seconds_per_game"] = self.timing["total_wall_seconds"] / max(
            len(results), 1
        )

        summary = self._summarize(results)
        self._save(results, summary)
        return {"summary": summary, "results": results}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone faithful inference-time BED-LLM Twenty Questions baseline."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--theta-words", type=Path, required=True)
    parser.add_argument("--eval-words", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--question-max-new-tokens", type=int, default=None)
    parser.add_argument("--likelihood-cache", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--max-eval-words", type=int, default=None)
    parser.add_argument("--active-threshold", type=float, default=0.01)
    parser.add_argument(
        "--zero-inactive-hypotheses",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load completed games from OUTPUT_DIR/trajectories.json and continue.",
    )
    parser.add_argument(
        "--belief-probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run evaluation-only greedy belief probes; use --no-belief-probe to skip them.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    if args.question_max_new_tokens is not None:
        config["training"]["max_completion_length"] = int(args.question_max_new_tokens)
    if args.likelihood_cache is not None:
        config["game"]["likelihood_cache_path"] = str(args.likelihood_cache)
    if args.seed is not None:
        config["evaluation"]["seed"] = int(args.seed)
    if args.category is not None:
        config["game"]["category"] = str(args.category)
    if args.max_eval_words is not None:
        config["evaluation"]["max_eval_words"] = int(args.max_eval_words)

    # Keep BED-LLM's hidden evaluator diagnostics identical to the direct-policy
    # full-game evaluation protocol. These settings do not affect BED question
    # selection; they only define the common realized information metrics.
    config["game"]["information_reward_mode"] = "realized"
    config["game"]["active_threshold"] = float(args.active_threshold)
    config["game"]["zero_inactive_hypotheses"] = bool(
        args.zero_inactive_hypotheses
    )
    config["evaluation"]["bed_llm"]["run_belief_probe"] = bool(args.belief_probe)

    theta_set = load_tq_words_file(args.theta_words)
    eval_words = load_tq_words_file(args.eval_words)
    max_eval_words = config["evaluation"]["max_eval_words"]
    if max_eval_words is not None:
        eval_words = eval_words[-int(max_eval_words) :]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint is None:
        model, tokenizer = load_model_and_tokenizer(config, for_inference=True)
        model_source = config["model"]["name"]
    else:
        model, tokenizer = load_model_and_tokenizer_from_checkpoint(
            config,
            checkpoint_path=args.checkpoint,
            for_inference=True,
        )
        model_source = str(args.checkpoint)

    evaluator = BEDLLMEvaluator(
        config=config,
        model=model,
        tokenizer=tokenizer,
        theta_set=theta_set,
        eval_words=eval_words,
        output_dir=args.output_dir,
    )

    protocol = {
        "method": "bed_llm",
        "config": str(args.config),
        "model_source": str(model_source),
        "theta_words": str(args.theta_words),
        "eval_words_file": str(args.eval_words),
        "eval_words": eval_words,
        "evaluation": config["evaluation"],
        "question_max_new_tokens": config["training"]["max_completion_length"],
        "max_seq_length": config["evaluation"]["bed_llm"].get(
            "max_seq_length",
            32768,
        ),
        "resume": args.resume,
    }
    with (args.output_dir / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False)

    record = evaluator.evaluate(resume=args.resume)
    print("\n=== BED-LLM SUMMARY ===")
    print(json.dumps(record["summary"], indent=2, sort_keys=True))
    print(f"\nSaved outputs to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

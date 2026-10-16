from __future__ import annotations

import copy
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
from tqdm.auto import tqdm

from .api_models import AzureChatBatchClient
from .build_tq_dataset import (
    ANSWER_LABELS,
    _query_likelihoods_with_recovery,
    extract_likelihoods_prompt,
    get_tq_cache,
    normalize_likelihood_entry,
    normalize_lookup_key,
    parse_likelihoods_response,
    save_tq_cache,
)
from .tq_environment import _blau_step_update_and_reward
from .tq_full_game_eval import (
    TQFullGameEvalCallback,
    _ensure_post_tool_generation_marker,
    _extract_single_tool_call,
    _question_has_literal_line_escape,
    _question_generation_eos_token_ids,
)


@dataclass
class _FullGameTransition:
    game: dict
    step_record: dict
    parsed: dict
    tool_name: str
    question: str
    C_prev: np.ndarray
    active_indices: np.ndarray
    active_candidates: list[str]
    query_candidates: list[str]


class TQBatchedFullGameEvalCallback(TQFullGameEvalCallback):
    """Full-game TQ eval with batched Gemma turns and batched GPT oracle calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        game_config = self.config.get("game", {})
        eval_config = self.config.get("eval", {})
        bed_eval_config = self.config.get("evaluation", {}).get("bed_llm", {})
        self.question_retries = max(
            1,
            int(
                eval_config.get(
                    "full_game_question_retries",
                    bed_eval_config.get("question_retries", 1),
                )
            ),
        )

        self.likelihood_cache_path = eval_config.get(
            "full_game_likelihood_cache_path"
        ) or game_config.get("likelihood_cache_path")
        if not self.likelihood_cache_path:
            raise ValueError(
                "Batched full-game eval requires game.likelihood_cache_path "
                "or eval.full_game_likelihood_cache_path."
            )
        self.likelihood_cache_path = str(self.likelihood_cache_path)

        self.api_batch_size = max(
            1,
            int(
                eval_config.get(
                    "full_game_api_batch_size",
                    eval_config.get(
                        "full_game_oracle_batch_size",
                        game_config.get("oracle_parallel_batch_size", 32),
                    ),
                )
            ),
        )
        self.oracle_max_output_tokens = int(
            eval_config.get("full_game_oracle_max_output_tokens", 16000)
        )
        self.oracle_max_retries = int(
            eval_config.get("full_game_oracle_max_retries", 8)
        )
        self.oracle_reasoning_effort = eval_config.get(
            "full_game_oracle_reasoning_effort",
            "none",
        )
        if self.oracle_reasoning_effort is not None:
            self.oracle_reasoning_effort = (
                str(self.oracle_reasoning_effort).strip() or None
            )

        missing = [
            name
            for name in (
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_VERSION",
                "AZURE_OPENAI_DEPLOYMENT",
            )
            if not os.environ.get(name)
        ]
        if missing:
            raise RuntimeError(
                "Batched full-game eval requires Azure OpenAI environment "
                "variables: " + ", ".join(missing)
            )

        self.batch_client = AzureChatBatchClient()

    def _generation_kwargs(self):
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.text_tokenizer.pad_token_id,
            "eos_token_id": _question_generation_eos_token_ids(
                self.text_tokenizer
            ),
        }
        if self.do_sample:
            if self.temperature <= 0:
                raise ValueError(
                    "eval.full_game_temperature must be > 0 "
                    "when eval.full_game_do_sample=true."
                )
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                }
            )
        else:
            generation_kwargs["do_sample"] = False
        return generation_kwargs

    def _render_prompt(self, game):
        rendered = self.text_tokenizer.apply_chat_template(
            game["messages"],
            tools=[game["env"].ask_question],
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        return _ensure_post_tool_generation_marker(rendered, game["messages"])

    def _generate_active_games(self, model, active_games, turn, global_step):
        pending_games = list(active_games)
        attempts_by_game = {id(game): [] for game in active_games}
        completed_by_game = {}
        generation_kwargs = self._generation_kwargs()
        device = next(model.parameters()).device

        for attempt in range(1, self.question_retries + 1):
            if not pending_games:
                break

            retry_games = []
            chunk_iter = range(0, len(pending_games), self.batch_size)
            for chunk_start in tqdm(
                chunk_iter,
                total=(len(pending_games) + self.batch_size - 1) // self.batch_size,
                desc=(
                    f"step {global_step} turn {turn} Gemma attempt "
                    f"{attempt}/{self.question_retries}"
                ),
                leave=False,
            ):
                chunk_games = pending_games[
                    chunk_start : chunk_start + self.batch_size
                ]
                rendered_prompts = [self._render_prompt(game) for game in chunk_games]
                inputs = self.text_tokenizer(
                    rendered_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                self._check_prompt_lengths(
                    model,
                    inputs,
                    max_new_tokens=self.max_new_tokens,
                    prompt_kind="Question-generation",
                )
                inputs = inputs.to(device)

                self._synchronize_generation_device()
                inference_start = time.perf_counter()
                outputs = model.generate(**inputs, **generation_kwargs)
                self._synchronize_generation_device()
                per_game_seconds = (
                    time.perf_counter() - inference_start
                ) / len(chunk_games)

                prompt_width = inputs["input_ids"].shape[-1]
                generated_batch = outputs[:, prompt_width:]
                for game, rendered, generated_ids in zip(
                    chunk_games,
                    rendered_prompts,
                    generated_batch,
                    strict=True,
                ):
                    raw_completion = self.text_tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=False,
                    )
                    parse_error = None
                    try:
                        parsed = self.text_tokenizer.parse_response(raw_completion)
                    except (TypeError, ValueError) as exc:
                        parsed = None
                        parse_error = f"{type(exc).__name__}: {exc}"
                    attempt_record = {
                        "attempt": attempt,
                        "raw_completion": raw_completion,
                        "parsed": copy.deepcopy(parsed),
                        "model_inference_seconds": float(per_game_seconds),
                    }
                    if parse_error is not None:
                        attempt_record["parse_error"] = parse_error
                    attempts = attempts_by_game[id(game)]
                    attempts.append(attempt_record)

                    tool_name, tool_args = _extract_single_tool_call(parsed)
                    question = (
                        str(tool_args.get("question", "")).strip()
                        if isinstance(tool_args, dict)
                        else ""
                    )
                    malformed_question_escape = (
                        _question_has_literal_line_escape(question)
                    )
                    if malformed_question_escape:
                        attempt_record["validation_error"] = (
                            "literal_line_escape_in_question"
                        )
                    valid = (
                        tool_name == "ask_question"
                        and hasattr(game["env"], tool_name)
                        and isinstance(tool_args, dict)
                        and bool(question)
                        and not malformed_question_escape
                    )
                    if not valid and attempt < self.question_retries:
                        retry_games.append(game)
                        continue

                    step_record: Dict[str, Any] = {
                        "turn": turn,
                        "raw_completion": raw_completion,
                        "parsed": copy.deepcopy(parsed),
                        "model_inference_seconds": float(
                            sum(row["model_inference_seconds"] for row in attempts)
                        ),
                        "question_generation_num_attempts": len(attempts),
                        "question_generation_attempts": copy.deepcopy(attempts),
                    }
                    if self.store_rendered_prompts:
                        step_record["rendered_prompt"] = rendered
                    if self.store_structured_context:
                        step_record["structured_context_before"] = copy.deepcopy(
                            game["messages"]
                        )
                    completed_by_game[id(game)] = {
                        "game": game,
                        "rendered": rendered,
                        "parsed": parsed,
                        "step_record": step_record,
                    }

            pending_games = retry_games

        return [completed_by_game[id(game)] for game in active_games]

    @staticmethod
    def _mark_invalid_tool_call(game, step_record):
        step_record.update(
            {
                "question": None,
                "answer": None,
                "tool_response": None,
                "valid_tool_call": False,
            }
        )
        game["steps"].append(step_record)
        game["finished"] = True
        game["stop_reason"] = "invalid_tool_call"

    def _make_transition(self, item) -> _FullGameTransition | None:
        game = item["game"]
        env = game["env"]
        parsed = item["parsed"]
        step_record = item["step_record"]

        tool_name, tool_args = _extract_single_tool_call(parsed)
        if tool_name is None or not hasattr(env, tool_name):
            self._mark_invalid_tool_call(game, step_record)
            return None

        if tool_name != "ask_question":
            self._mark_invalid_tool_call(game, step_record)
            return None

        question = str(tool_args.get("question", "")).strip()
        if not question or _question_has_literal_line_escape(question):
            self._mark_invalid_tool_call(game, step_record)
            return None

        if len(env.step_rewards) >= env.horizon:
            env.invalid = True
            raise ValueError("Rollout horizon already reached.")

        C_prev = env.C.copy()
        active_indices = np.flatnonzero(C_prev > env.active_threshold)
        active_candidates = [env.contrastive_set[index] for index in active_indices]

        query_candidates = list(active_candidates)
        if env.true_word not in query_candidates:
            query_candidates.append(env.true_word)

        return _FullGameTransition(
            game=game,
            step_record=step_record,
            parsed=parsed,
            tool_name=tool_name,
            question=question,
            C_prev=C_prev,
            active_indices=active_indices,
            active_candidates=active_candidates,
            query_candidates=query_candidates,
        )

    def _fetch_likelihoods_batched(
        self,
        transitions: list[_FullGameTransition],
        *,
        global_step: int,
        turn: int,
    ) -> list[dict[str, dict[str, float]]]:
        cache = get_tq_cache(self.likelihood_cache_path)
        rows_by_transition: list[list[Any]] = [
            [None] * len(transition.query_candidates) for transition in transitions
        ]
        missing_words_by_question: dict[str, set[str]] = defaultdict(set)
        cache_hits = 0
        cache_misses = 0
        ground_truth_slots = 0

        for transition_index, transition in enumerate(transitions):
            question = transition.question
            if question not in cache:
                cache[question] = {}

            normalized_cache_keys = {
                normalize_lookup_key(key): key for key in cache[question]
            }

            for candidate_index, word in enumerate(transition.query_candidates):
                ground_truth_slots += 1
                cached_key = normalized_cache_keys.get(normalize_lookup_key(word))
                cached_value = (
                    cache[question].get(cached_key) if cached_key is not None else None
                )
                normalized_value = normalize_likelihood_entry(cached_value)

                if normalized_value is not None:
                    if cached_key is not None and cached_key != word:
                        cache[question].pop(cached_key, None)
                    cache[question][word] = normalized_value
                    rows_by_transition[transition_index][candidate_index] = (
                        normalized_value
                    )
                    cache_hits += 1
                else:
                    cache[question].pop(word, None)
                    missing_words_by_question[question].add(word)
                    cache_misses += 1

        if self.usage_tracker is not None and hasattr(
            self.usage_tracker, "record_oracle_lookup_slots"
        ):
            self.usage_tracker.record_oracle_lookup_slots(
                num_questions=len(transitions),
                num_ground_truth_slots=ground_truth_slots,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
            )

        request_items = [
            (question, sorted(missing_words))
            for question, missing_words in missing_words_by_question.items()
            if missing_words
        ]

        if request_items:
            print(
                f"Full-game oracle @ step {global_step} turn {turn}: "
                f"requests={len(request_items)} cache_hits={cache_hits} "
                f"cache_misses={cache_misses}",
                flush=True,
            )
            messages_batch = [
                extract_likelihoods_prompt([question], missing_words)
                for question, missing_words in request_items
            ]

            results = []
            for start in tqdm(
                range(0, len(messages_batch), self.api_batch_size),
                total=(len(messages_batch) + self.api_batch_size - 1)
                // self.api_batch_size,
                desc=f"step {global_step} turn {turn} GPT oracle batches",
                leave=False,
            ):
                chunk_results = self.batch_client.complete_many(
                    messages_batch[start : start + self.api_batch_size],
                    batch_size=self.api_batch_size,
                    max_output_tokens=self.oracle_max_output_tokens,
                    max_retries=self.oracle_max_retries,
                    reasoning_effort=self.oracle_reasoning_effort,
                )
                results.extend(chunk_results)

            if self.usage_tracker is not None and hasattr(
                self.usage_tracker, "record_oracle_results"
            ):
                self.usage_tracker.record_oracle_results(results)

            for (question, missing_words), result in zip(
                request_items,
                results,
                strict=True,
            ):
                data = parse_likelihoods_response(
                    result.text,
                    [question],
                    missing_words,
                )
                question_probs = data.get(question, {})
                for word in missing_words:
                    probs = question_probs.get(word)
                    if probs is not None:
                        cache[question][word] = probs

                unresolved = [
                    word
                    for word in missing_words
                    if normalize_likelihood_entry(cache[question].get(word)) is None
                ]
                if unresolved:
                    recovered = _query_likelihoods_with_recovery(
                        question,
                        unresolved,
                        usage_tracker=self.usage_tracker,
                    )
                    for word, probs in recovered.items():
                        normalized = normalize_likelihood_entry(probs)
                        if normalized is not None:
                            cache[question][word] = normalized

            save_tq_cache(cache, self.likelihood_cache_path)
        else:
            print(
                f"Full-game oracle @ step {global_step} turn {turn}: "
                f"all cached cache_hits={cache_hits}",
                flush=True,
            )

        for transition_index, transition in enumerate(transitions):
            question = transition.question
            for candidate_index, word in enumerate(transition.query_candidates):
                if rows_by_transition[transition_index][candidate_index] is None:
                    rows_by_transition[transition_index][candidate_index] = (
                        normalize_likelihood_entry(cache[question].get(word))
                    )

        likelihoods_by_transition = []
        for transition, rows in zip(transitions, rows_by_transition, strict=True):
            likelihoods = {
                candidate: row
                for candidate, row in zip(
                    transition.query_candidates,
                    rows,
                    strict=True,
                )
                if row is not None
            }
            likelihoods_by_transition.append(likelihoods)

        return likelihoods_by_transition

    def _apply_transition(
        self,
        transition: _FullGameTransition,
        likelihoods: dict[str, dict[str, float]],
    ) -> str:
        env = transition.game["env"]
        true_dist = likelihoods.get(env.true_word)
        if true_dist is None:
            raise ValueError(
                "Missing true-word likelihood: "
                f"question={transition.question!r}, true_word={env.true_word!r}."
            )

        env.last_solved = bool(true_dist.get("solved", False))
        observed_answer = max(
            ANSWER_LABELS,
            key=lambda label: true_dist[label],
        )
        true_likelihood = float(true_dist[observed_answer])
        step_reward = 0.0

        if transition.active_candidates:
            valid_indices = []
            likelihood_values = []

            for index, candidate in zip(
                transition.active_indices,
                transition.active_candidates,
                strict=True,
            ):
                dist = likelihoods.get(candidate)
                if dist is None:
                    continue

                value = float(dist.get(observed_answer, np.nan))
                if np.isfinite(value) and value >= 0:
                    valid_indices.append(int(index))
                    likelihood_values.append(value)

            if valid_indices:
                # Use the same update/reward helper as the sequential environment.
                # During eval, zero_inactive_hypotheses=True means any C that
                # crosses the active threshold is permanently set to 0 before
                # the realized sPCE denominator is evaluated.
                C_new, step_reward, updated = _blau_step_update_and_reward(
                    C_prev=transition.C_prev,
                    active_indices=np.asarray(valid_indices, dtype=int),
                    active_candidates=[
                        transition.active_candidates[
                            list(transition.active_indices).index(index)
                        ]
                        for index in valid_indices
                    ],
                    likelihoods=likelihoods,
                    observed_answer=observed_answer,
                    true_likelihood=true_likelihood,
                    information_reward_mode=env.information_reward_mode,
                    active_threshold=env.active_threshold,
                    zero_inactive_hypotheses=getattr(
                        env, "zero_inactive_hypotheses", False
                    ),
                )
                if updated:
                    env.C = C_new

        env.history.append(
            {
                "question": transition.question,
                "answer": observed_answer,
            }
        )
        env.step_rewards.append(float(step_reward))
        env.step_debug.append(
            {
                "question": transition.question,
                "answer": observed_answer,
                "reward": float(step_reward),
                "active_count": int(np.sum(env.C > env.active_threshold)),
                "sum_C": float(np.sum(env.C)),
                "solved": bool(env.last_solved),
            }
        )

        status = "solved" if env.last_solved else "continue"
        return f"Answer: {observed_answer}\nStatus: {status}"

    def _apply_successful_step(
        self,
        transition: _FullGameTransition,
        tool_result: str,
    ) -> None:
        game = transition.game
        env = game["env"]
        debug = dict(env.step_debug[-1])
        step_record = transition.step_record

        step_record.update(
            {
                "question": debug["question"],
                "answer": debug["answer"],
                "tool_response": str(tool_result),
                "reward": float(debug["reward"]),
                "active_count": int(debug["active_count"]),
                "sum_C": float(debug["sum_C"]),
                "solved": bool(debug.get("solved", False)),
                "valid_tool_call": True,
            }
        )

        game["steps"].append(step_record)

        # EXACT history augmentation from the verified notebook.
        game["messages"].append(copy.deepcopy(transition.parsed))
        game["messages"].append(
            {
                "role": "tool",
                "name": transition.tool_name,
                "content": str(tool_result),
            }
        )

        if debug.get("solved", False):
            game["finished"] = True
            game["stop_reason"] = "solved"

    def _execute_batched_tool_steps(
        self,
        generated_items,
        *,
        global_step: int,
        turn: int,
    ) -> None:
        transitions = []
        for item in generated_items:
            transition = self._make_transition(item)
            if transition is not None:
                transitions.append(transition)

        if not transitions:
            return

        likelihoods_by_transition = self._fetch_likelihoods_batched(
            transitions,
            global_step=global_step,
            turn=turn,
        )

        for transition, likelihoods in zip(
            transitions,
            likelihoods_by_transition,
            strict=True,
        ):
            tool_result = self._apply_transition(transition, likelihoods)
            self._apply_successful_step(transition, tool_result)

    @torch.no_grad()
    def evaluate(self, global_step: int) -> Dict[str, float]:
        model = self.trainer.model
        was_training = model.training

        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.random.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )

        eval_seed = self.eval_seed
        random.seed(eval_seed)
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)

        model.eval()
        results = []

        try:
            games = [
                self._new_game(true_word)
                for true_word in tqdm(
                    self.eval_words,
                    desc=f"step {global_step} initializing games",
                )
            ]
            self._run_belief_probes(model, games, turn=0)

            turn_bar = tqdm(
                range(1, self.max_turns + 1),
                desc=f"step {global_step} full-game turns",
            )
            for turn in turn_bar:
                active_games = [game for game in games if not game["finished"]]
                turn_bar.set_postfix(
                    {
                        "active": len(active_games),
                        "solved": sum(
                            1 for game in games if game["stop_reason"] == "solved"
                        ),
                        "invalid": sum(
                            1
                            for game in games
                            if game["stop_reason"] == "invalid_tool_call"
                        ),
                    }
                )

                if not active_games:
                    break

                print(
                    f"\nFull-game batched eval @ step {global_step}: "
                    f"turn={turn}/{self.max_turns} "
                    f"active={len(active_games)}/{len(games)} "
                    f"gemma_bs={self.batch_size} api_bs={self.api_batch_size}",
                    flush=True,
                )

                generated_items = self._generate_active_games(
                    model,
                    active_games,
                    turn,
                    global_step,
                )
                self._execute_batched_tool_steps(
                    generated_items,
                    global_step=global_step,
                    turn=turn,
                )
                probed_games = [
                    game
                    for game in active_games
                    if game["steps"]
                    and game["steps"][-1].get("turn") == turn
                    and game["steps"][-1].get("valid_tool_call", False)
                ]
                self._run_belief_probes(model, probed_games, turn=turn)

            results.extend(self._finish_game(game) for game in games)

        finally:
            if was_training:
                model.train()

            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.random.set_rng_state(torch_rng_state)
            if cuda_rng_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_rng_state)

        self._save(
            results=results,
            global_step=global_step,
            eval_seed=eval_seed,
        )

        return self._summarize(results)

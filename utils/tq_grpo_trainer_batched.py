"""Batched TQ environment execution for the reasoning-aware GRPO trainer."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

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
from .tq_grpo_trainer import TQGRPOTrainer


@dataclass
class _Action:
    batch_idx: int
    name: str
    arguments: dict[str, Any]
    question: str


@dataclass
class _Transition:
    batch_idx: int
    question: str
    C_prev: np.ndarray
    active_indices: np.ndarray
    active_candidates: list[str]
    query_candidates: list[str]


class TQBatchedGRPOTrainer(TQGRPOTrainer):
    """TQGRPOTrainer with batched oracle/cache execution for ask_question."""

    def __init__(
        self,
        *args,
        likelihood_cache_path: str,
        usage_tracker=None,
        oracle_batch_size: int = 32,
        oracle_max_output_tokens: int = 16000,
        oracle_max_retries: int = 8,
        oracle_reasoning_effort: str | None = None,
        **kwargs,
    ):

        self.likelihood_cache_path = str(likelihood_cache_path)

        self.usage_tracker = usage_tracker
        self.oracle_batch_size = max(1, int(oracle_batch_size))
        self.oracle_max_output_tokens = int(oracle_max_output_tokens)
        self.oracle_max_retries = int(oracle_max_retries)
        self.oracle_reasoning_effort = (
            str(oracle_reasoning_effort).strip()
            if oracle_reasoning_effort is not None
            else None
        )
        if not self.oracle_reasoning_effort:
            self.oracle_reasoning_effort = None

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
                "Batched TQ oracle execution requires Azure OpenAI environment "
                "variables: "
                + ", ".join(missing)
            )

        self.tq_batch_client = AzureChatBatchClient()
        super().__init__(*args, **kwargs)

    def _make_transition(self, action: _Action, env=None) -> _Transition:
        if env is None:
            if self.environments is None:
                raise RuntimeError("TQBatchedGRPOTrainer requires environments.")
            env = self.environments[action.batch_idx]

        if bool(getattr(env, "solved", False)):
            raise ValueError("Game already solved; no further question is allowed.")
        if len(env.step_rewards) >= env.horizon:
            env.invalid = True
            raise ValueError("Rollout horizon already reached.")

        question = str(action.question).strip()
        if not question:
            env.invalid = True
            raise ValueError("Question must be non-empty.")

        C_prev = env.C.copy()
        active_indices = np.flatnonzero(C_prev > env.active_threshold)
        active_candidates = [
            env.contrastive_set[index] for index in active_indices
        ]

        query_candidates = list(active_candidates)
        if env.true_word not in query_candidates:
            query_candidates.append(env.true_word)

        return _Transition(
            batch_idx=action.batch_idx,
            question=question,
            C_prev=C_prev,
            active_indices=active_indices,
            active_candidates=active_candidates,
            query_candidates=query_candidates,
        )

    def _fetch_likelihoods_batched(
        self,
        transitions: list[_Transition],
    ) -> list[dict[str, dict[str, float]]]:
        cache = get_tq_cache(self.likelihood_cache_path)
        rows_by_transition: list[list[Any]] = [
            [None] * len(transition.query_candidates)
            for transition in transitions
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
                    cache[question].get(cached_key)
                    if cached_key is not None
                    else None
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

        if (
            self.usage_tracker is not None
            and hasattr(self.usage_tracker, "record_oracle_lookup_slots")
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
            messages_batch = [
                extract_likelihoods_prompt([question], missing_words)
                for question, missing_words in request_items
            ]
            results = self.tq_batch_client.complete_many(
                messages_batch,
                batch_size=self.oracle_batch_size,
                max_output_tokens=self.oracle_max_output_tokens,
                max_retries=self.oracle_max_retries,
                reasoning_effort=self.oracle_reasoning_effort,
            )

            if (
                self.usage_tracker is not None
                and hasattr(self.usage_tracker, "record_oracle_results")
            ):
                self.usage_tracker.record_oracle_results(results)

            for (question, missing_words), result in zip(
                request_items,
                results,
                strict=True,
            ):
                try:
                    data = parse_likelihoods_response(
                        result.text,
                        [question],
                        missing_words,
                    )
                except Exception as exc:
                    print(
                        "Could not parse batched likelihood response: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    print(f"Raw response:\n{result.text}", flush=True)
                    data = {question: {}}

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
        transition: _Transition,
        likelihoods: dict[str, dict[str, float]],
        env=None,
    ) -> str:
        if env is None:
            env = self.environments[transition.batch_idx]
        true_dist = likelihoods.get(env.true_word)
        if true_dist is None:
            raise ValueError(
                "Missing true-word likelihood: "
                f"question={transition.question!r}, true_word={env.true_word!r}."
            )

        env.last_solved = bool(true_dist.get("solved", False))
        env.solved = env.solved or env.last_solved
        observed_answer = max(
            ANSWER_LABELS,
            key=lambda label: true_dist[label],
        )
        true_likelihood = float(true_dist[observed_answer])
        step_reward = 0.0

        if transition.active_candidates:
            C_new, step_reward, updated = _blau_step_update_and_reward(
                C_prev=transition.C_prev,
                active_indices=transition.active_indices,
                active_candidates=transition.active_candidates,
                likelihoods=likelihoods,
                observed_answer=observed_answer,
                true_likelihood=true_likelihood,
                information_reward_mode=env.information_reward_mode,
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
        return (
            f"Answer: {observed_answer}\n"
            f"Status: {status}"
        )

    def _tq_execute_continuation_questions(self, branch_items):
        """Batch oracle likelihood work for cloned M-sample continuations."""
        transitions = []
        transition_envs = []
        valid_positions = []
        results = [None] * len(branch_items)

        for position, (env, question) in enumerate(branch_items):
            action = _Action(
                batch_idx=position,
                name="ask_question",
                arguments={"question": question},
                question=question,
            )
            try:
                transition = self._make_transition(action, env=env)
            except Exception as exc:
                env.invalid = True
                results[position] = ({"error": str(exc)}, False)
                continue
            transitions.append(transition)
            transition_envs.append(env)
            valid_positions.append(position)

        if transitions:
            likelihoods_by_transition = self._fetch_likelihoods_batched(transitions)
            for position, transition, env, likelihoods in zip(
                valid_positions,
                transitions,
                transition_envs,
                likelihoods_by_transition,
                strict=True,
            ):
                try:
                    result = self._apply_transition(
                        transition, likelihoods, env=env
                    )
                    results[position] = (result, True)
                except Exception as exc:
                    env.invalid = True
                    results[position] = ({"error": str(exc)}, False)

        failures = sum(1 for _, succeeded in results if not succeeded)
        return results, len(branch_items), failures

    def _tq_execute_actions(
        self,
        *,
        active,
        parsed_current,
        prompts,
        completions,
        action_number,
        horizon,
        debug_enabled,
        debug_sample_idx,
    ):
        actions_by_batch_idx = {}
        transitions = []
        transition_actions = []
        result_by_batch_idx = {}
        tool_call_count = 0

        for batch_idx in active:
            _, name, arguments = self._tq_prepare_action_messages(
                batch_idx=batch_idx,
                parsed_current=parsed_current,
                prompts=prompts,
                completions=completions,
            )
            tool_call_count += 1

            question = (
                arguments.get("question")
                if isinstance(arguments, dict)
                else None
            )
            action = _Action(
                batch_idx=batch_idx,
                name=name,
                arguments=arguments if isinstance(arguments, dict) else {},
                question=str(question or ""),
            )
            actions_by_batch_idx[batch_idx] = action

            if name != "ask_question":
                result_by_batch_idx[batch_idx] = (
                    {"error": f"Tool {name} not found."},
                    False,
                )
                continue

            try:
                transition = self._make_transition(action)
            except Exception as exc:
                result_by_batch_idx[batch_idx] = ({"error": str(exc)}, False)
                continue

            transitions.append(transition)
            transition_actions.append(action)

        if transitions:
            likelihoods_by_transition = self._fetch_likelihoods_batched(
                transitions
            )
            for transition, action, likelihoods in zip(
                transitions,
                transition_actions,
                likelihoods_by_transition,
                strict=True,
            ):
                env = self.environments[transition.batch_idx]
                try:
                    result = self._apply_transition(transition, likelihoods)
                    result_by_batch_idx[action.batch_idx] = (result, True)
                except Exception as exc:
                    env.invalid = True
                    result_by_batch_idx[action.batch_idx] = (
                        {"error": str(exc)},
                        False,
                    )

        next_active_candidates = []
        tool_failure_count = 0

        for batch_idx in active:
            action = actions_by_batch_idx[batch_idx]
            result, tool_succeeded = result_by_batch_idx[batch_idx]
            if not tool_succeeded:
                tool_failure_count += 1

            self._tq_append_action_result(
                batch_idx=batch_idx,
                prompts=prompts,
                completions=completions,
                name=action.name,
                result=result,
                tool_succeeded=tool_succeeded,
                arguments=action.arguments,
                action_number=action_number,
                horizon=horizon,
                debug_enabled=debug_enabled,
                debug_sample_idx=debug_sample_idx,
            )

            if tool_succeeded:
                env = self.environments[batch_idx]
                if not bool(getattr(env, "solved", False)):
                    next_active_candidates.append(batch_idx)

        return next_active_candidates, tool_call_count, tool_failure_count

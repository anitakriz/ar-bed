"""K=1 LLM-BED/EIG reward for TQ GRPO."""

from __future__ import annotations

import copy
import json
from typing import Any

from .tq_grpo_trainer import (
    TQGRPOTrainer,
    _GEMMA_COMPLETE_ASK_QUESTION_RE,
    _is_valid_ask_question,
)
from .tq_llm_bed_reward import (
    LLMBEDRewardConfig,
    PolicyLLMBEDReward,
    append_bed_hypotheses_to_prompt,
)


class TQLLMBEDGRPOTrainer(TQGRPOTrainer):
    """GRPO trainer variant with one-step LLM-BED rewards.

    This is intentionally K=1 only. The training model samples fresh
    hypotheses; yes/no/maybe likelihoods are scored either from current-policy
    logits or from the explicit external/cache-backed likelihood source.
    """

    tq_rewarded_token_description = "DIRECT TOOL CALL / QUESTION ONLY"
    tq_first_turn_kept_description = "DIRECT TOOL CALL / QUESTION ONLY"

    def _truncate_to_first_ask_question(
        self,
        ids: list[int],
    ) -> tuple[list[int], bool]:
        """For no-reasoning LLM-BED, keep only the first tool call.

        The base TQ trainer intentionally keeps any reasoning prefix because the
        older Blau runs trained with reasoning. This K=1 LLM-BED variant is a
        direct/no-reasoning run, so any text or thought-channel prefix before
        the tool call is removed from the rewarded span.
        """
        ids = list(ids)
        if not ids:
            return ids, False

        raw_text = self._decode_ids(ids)
        match = _GEMMA_COMPLETE_ASK_QUESTION_RE.search(raw_text)
        if match is None:
            return ids, False

        start = match.start()
        marker = "<|tool_call>"
        marker_pos = raw_text.rfind(marker, 0, match.start())
        if marker_pos >= 0:
            start = marker_pos
        tool_call_text = raw_text[start : match.end()]
        tool_call_ids = self._encode_text(tool_call_text)

        if _is_valid_ask_question(self._parse_ids(tool_call_ids)):
            return tool_call_ids, len(tool_call_ids) < len(ids)

        # Token-boundary fallback for tokenizers whose special tokens do not
        # round-trip perfectly through encode(decode(ids)).
        for cutoff_start in range(0, len(ids)):
            candidate_ids = ids[cutoff_start:]
            candidate_text = self._decode_ids(candidate_ids)
            candidate_match = _GEMMA_COMPLETE_ASK_QUESTION_RE.search(candidate_text)
            if candidate_match is None:
                continue
            candidate_ids = candidate_ids[: len(self._encode_text(candidate_text[: candidate_match.end()]))]
            if _is_valid_ask_question(self._parse_ids(candidate_ids)):
                return candidate_ids, cutoff_start > 0 or len(candidate_ids) < len(ids)

        # If the direct slice cannot be parsed, fall back to the parent behavior
        # so invalid generations still receive the appropriate trajectory signal.
        return super()._truncate_to_first_ask_question(ids)

    def __init__(
        self,
        *args,
        llm_bed_config: LLMBEDRewardConfig,
        category: str,
        max_turns: int,
        chat_template_kwargs: dict[str, Any] | None = None,
        usage_tracker=None,
        **kwargs,
    ):
        self.llm_bed_config = llm_bed_config
        self.llm_bed_category = str(category)
        self.llm_bed_max_turns = int(max_turns)
        self.llm_bed_chat_template_kwargs = dict(chat_template_kwargs or {})
        self.llm_bed_usage_tracker = usage_tracker
        super().__init__(*args, **kwargs)

        if int(self.max_tool_calling_iterations) != 1:
            raise ValueError("TQLLMBEDGRPOTrainer is K=1 only.")

        self.llm_bed_rewarder = PolicyLLMBEDReward(
            model=self.model,
            tokenizer=self._tq_tokenizer(),
            config=self.llm_bed_config,
            category=self.llm_bed_category,
            max_turns=self.llm_bed_max_turns,
            chat_template_kwargs=self.llm_bed_chat_template_kwargs,
            usage_tracker=self.llm_bed_usage_tracker,
        )

        self._llm_bed_rollout_hypotheses_by_batch_idx: list[list[str]] = []
        self._llm_bed_rollout_hypothesis_stats_by_batch_idx: list[dict[str, Any]] = []

    @staticmethod
    def _history_from_training_input(row: dict[str, Any]) -> list[dict[str, str]]:
        raw = row.get("history_json")
        if raw is None:
            raise KeyError(
                "LLM-BED prompt conditioning requires history_json in the training row."
            )
        if isinstance(raw, str):
            history = json.loads(raw)
        else:
            history = copy.deepcopy(raw)
        return [dict(step) for step in history]

    def _condition_inputs_on_bed_hypotheses(
        self,
        inputs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        history_cache: dict[str, tuple[list[str], dict[str, Any]]] = {}
        conditioned_inputs: list[dict[str, Any]] = []
        hypotheses_by_batch_idx: list[list[str]] = []
        stats_by_batch_idx: list[dict[str, Any]] = []

        for row in inputs:
            row_copy = dict(row)
            history = self._history_from_training_input(row)
            key = json.dumps(history, sort_keys=True, ensure_ascii=False)
            if key not in history_cache:
                history_cache[key] = self.llm_bed_rewarder.sample_fresh_hypotheses(
                    history
                )

            candidates, stats = history_cache[key]
            candidates = list(candidates)
            stats = dict(stats)
            stats["sampled_once_before_question_generation"] = True
            stats["history_key"] = key
            row_copy["prompt"] = append_bed_hypotheses_to_prompt(
                row["prompt"],
                candidates,
            )
            conditioned_inputs.append(row_copy)
            hypotheses_by_batch_idx.append(candidates)
            stats_by_batch_idx.append(stats)

        self._llm_bed_rollout_hypotheses_by_batch_idx = hypotheses_by_batch_idx
        self._llm_bed_rollout_hypothesis_stats_by_batch_idx = stats_by_batch_idx
        return conditioned_inputs

    def _generate_and_score_completions(self, inputs):
        conditioned_inputs = self._condition_inputs_on_bed_hypotheses(inputs)
        return super()._generate_and_score_completions(conditioned_inputs)

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
        if int(horizon) != 1:
            raise RuntimeError("LLM-BED reward is K=1 only.")

        actions_by_batch_idx: dict[int, dict[str, Any]] = {}
        valid_batch_indices = []
        valid_histories = []
        valid_questions = []
        result_by_batch_idx: dict[int, tuple[Any, bool]] = {}
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
            action = {
                "name": name,
                "arguments": arguments if isinstance(arguments, dict) else {},
                "question": str(question or "").strip(),
            }
            actions_by_batch_idx[batch_idx] = action

            env = self.environments[batch_idx]
            if name != "ask_question":
                env.invalid = True
                result_by_batch_idx[batch_idx] = (
                    {"error": f"Tool {name} not found."},
                    False,
                )
                continue
            if not action["question"]:
                env.invalid = True
                result_by_batch_idx[batch_idx] = (
                    {"error": "Question must be non-empty."},
                    False,
                )
                continue
            if bool(getattr(env, "solved", False)):
                env.invalid = True
                result_by_batch_idx[batch_idx] = (
                    {"error": "Game already solved; no further question is allowed."},
                    False,
                )
                continue
            if len(getattr(env, "step_rewards", [])) >= int(getattr(env, "horizon", 1)):
                env.invalid = True
                result_by_batch_idx[batch_idx] = (
                    {"error": "Rollout horizon already reached."},
                    False,
                )
                continue

            valid_batch_indices.append(batch_idx)
            valid_histories.append([dict(step) for step in getattr(env, "history", [])])
            valid_questions.append(action["question"])

        if valid_batch_indices:
            precomputed_candidates = getattr(
                self,
                "_llm_bed_rollout_hypotheses_by_batch_idx",
                [],
            )
            precomputed_stats = getattr(
                self,
                "_llm_bed_rollout_hypothesis_stats_by_batch_idx",
                [],
            )
            if not all(batch_idx < len(precomputed_candidates) for batch_idx in valid_batch_indices):
                raise RuntimeError(
                    "Missing precomputed LLM-BED hypotheses for at least one "
                    "generated question. Refusing to resample during reward "
                    "scoring because hypotheses must be sampled once before "
                    "the GRPO completions and reused for EIG."
                )

            candidate_sets = [
                list(precomputed_candidates[batch_idx])
                for batch_idx in valid_batch_indices
            ]
            hypothesis_stats = [
                dict(precomputed_stats[batch_idx])
                if batch_idx < len(precomputed_stats)
                else {}
                for batch_idx in valid_batch_indices
            ]

            reward_results = self.llm_bed_rewarder.evaluate_questions(
                histories=valid_histories,
                questions=valid_questions,
                candidate_sets=candidate_sets,
                hypothesis_stats=hypothesis_stats,
            )
            for batch_idx, reward_result in zip(
                valid_batch_indices,
                reward_results,
                strict=True,
            ):
                env = self.environments[batch_idx]
                action = actions_by_batch_idx[batch_idx]
                try:
                    debug_payload = reward_result if self.llm_bed_config.debug else {
                        "reward_mode": "llm_bed",
                        "num_hypotheses": int(reward_result["num_hypotheses"]),
                        "predictive_distribution": list(
                            reward_result["predictive_distribution"]
                        ),
                        "likelihood_source": reward_result.get("likelihood_source"),
                        "hypothesis_source": reward_result.get("hypothesis_source"),
                        "external_likelihood_cache_path": reward_result.get(
                            "external_likelihood_cache_path"
                        ),
                        "used_stored_sampled_candidates": False,
                        "used_contrastive_set": False,
                        "used_likelihood_cache": bool(
                            reward_result.get("used_likelihood_cache", False)
                        ),
                    }
                    result = env._record_llm_bed_step(
                        question=action["question"],
                        reward=float(reward_result["eig"]),
                        debug=debug_payload,
                    )
                    result_by_batch_idx[batch_idx] = (result, True)
                except Exception as exc:
                    env.invalid = True
                    result_by_batch_idx[batch_idx] = ({"error": str(exc)}, False)

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
                name=action["name"],
                result=result,
                tool_succeeded=tool_succeeded,
                arguments=action["arguments"],
                action_number=action_number,
                horizon=horizon,
                debug_enabled=debug_enabled,
                debug_sample_idx=debug_sample_idx,
            )

            if (
                self.llm_bed_config.debug
                and debug_enabled
                and batch_idx == debug_sample_idx
                and tool_succeeded
            ):
                last_debug = self.environments[batch_idx].step_debug[-1]
                print(
                    "\n" + "=" * 110
                    + "\nLLM-BED EIG DEBUG"
                    + f"\nlikelihood_source={last_debug.get('likelihood_source')!r}"
                    + f"\nexternal_likelihood_cache_path={last_debug.get('external_likelihood_cache_path')!r}"
                    + f"\nhistory={last_debug.get('history')!r}"
                    + f"\nquestion={last_debug.get('question')!r}"
                    + f"\nhypotheses={last_debug.get('hypotheses')!r}"
                    + f"\nanswer_labels={last_debug.get('answer_labels')!r}"
                    + f"\nanswer_probabilities={last_debug.get('answer_probabilities')!r}"
                    + f"\npredictive_distribution={last_debug.get('predictive_distribution')!r}"
                    + f"\neig={float(last_debug.get('eig', 0.0)):.12f}"
                    + "\n" + "=" * 110,
                    flush=True,
                )

        # K=1 myopic reward: no continuation should be generated.
        return [], tool_call_count, tool_failure_count

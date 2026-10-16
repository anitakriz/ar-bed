"""Twenty Questions GRPO trainer with strict K-turn causal contexts.

This trainer implements the rollout semantics needed by the TQ experiments:

    context_t  --(model)-->  model_segment_t + question_t  --(env)--> answer_t

Only the first model-generated segment through ``question_1`` receives
the trajectory-level GRPO advantage. Later model turns are rollout-only: they
affect the K-step environment return but receive no policy loss. The real
environment answers receive no policy loss. For each later rollout turn, the
context is rebuilt from structured question/answer history, so earlier
reasoning is *not* visible to the model.

Solved trajectories terminate immediately and receive no missing-turn penalty.
With M=1, a nonterminal rollout executes at most K actions and never generates an
unexecuted K+1 turn.  For the current K=2 experiment, M>1 samples M independent
turn-2 continuations from the same Q1/A1 state and averages their returns.

The implementation intentionally targets the on-policy Transformers GRPO path
used by this project.  Unsupported configurations fail loudly in ``__init__``
rather than silently using an incorrect flattened-context objective.
"""

from __future__ import annotations

import asyncio
import copy
import re
from typing import Any

import torch
from trl import GRPOTrainer
from trl.chat_template_utils import parse_response as trl_parse_response


_GEMMA_COMPLETE_ASK_QUESTION_RE = re.compile(
    r"call:ask_question\s*\{\s*question\s*:\s*<\|\"\|>"
    r"(?P<question>.*?)"
    r"<\|\"\|>\s*\}<tool_call\|>",
    re.DOTALL,
)


def _tq_question_generation_eos_token_ids(tokenizer) -> int | list[int]:
    """Stop as soon as a complete Gemma tool call hands off control."""
    token_ids = []
    for token in (
        "<tool_call|>",
        "<|tool_response>",
        "<turn|>",
    ):
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            token_id = None
        if (
            token_id is not None
            and token_id != getattr(tokenizer, "unk_token_id", None)
            and token_id not in token_ids
        ):
            token_ids.append(int(token_id))

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and eos_token_id not in token_ids:
        token_ids.append(int(eos_token_id))

    if not token_ids:
        raise RuntimeError("Tokenizer has no usable TQ question stop token.")
    return token_ids[0] if len(token_ids) == 1 else token_ids


def _is_valid_ask_question(parsed: Any) -> bool:
    if not isinstance(parsed, dict):
        return False

    calls = parsed.get("tool_calls") or []
    if not calls:
        return False

    function = calls[0].get("function", {})
    if function.get("name") != "ask_question":
        return False

    arguments = function.get("arguments", {})
    if not isinstance(arguments, dict):
        return False

    question = arguments.get("question")
    return isinstance(question, str) and bool(question.strip())


class TQGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with TQ-specific strict K-turn rollout and loss contexts."""

    tq_debug_rollout: bool = True
    _tq_rollout_debug_done: bool = False
    _tq_loss_debug_done: bool = False
    tq_rewarded_token_description: str = "REASONING + QUESTION"
    tq_first_turn_kept_description: str = "REASONING + FIRST QUESTION"

    def __init__(self, *args, **kwargs):
        self.tq_continuation_samples = int(kwargs.pop("tq_continuation_samples", 1))
        if self.tq_continuation_samples < 1:
            raise ValueError("tq_continuation_samples must be at least 1.")

        self.tq_rewarded_token_mode = str(
            kwargs.pop("tq_rewarded_token_mode", "all")
        ).strip()
        if self.tq_rewarded_token_mode not in {"all", "final_tool_call"}:
            raise ValueError(
                "tq_rewarded_token_mode must be 'all' or 'final_tool_call'. "
                f"Got: {self.tq_rewarded_token_mode!r}."
            )
        if self.tq_rewarded_token_mode == "final_tool_call":
            self.tq_rewarded_token_description = "FINAL TOOL CALL / QUESTION ONLY"
            self.tq_first_turn_kept_description = (
                "REASONING KEPT AS CONTEXT; FINAL TOOL CALL REWARDED"
            )

        super().__init__(*args, **kwargs)
        self._validate_tq_training_configuration()
        if self.tq_continuation_samples > 1 and int(self.max_tool_calling_iterations) != 2:
            raise ValueError(
                "M>1 continuation sampling is currently implemented for K=2 only. "
                "Use tq_continuation_samples=1 for other horizons."
            )
        self._tq_last_turn_segments: list[list[dict[str, list[int]]]] = []

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------
    def _tq_tokenizer(self):
        """Return the real text tokenizer without assuming ``self._tokenizer``.

        Unsloth's compiled GRPO trainer does not always expose the private
        ``_tokenizer`` attribute even though upstream TRL does.  The processing
        class passed by tq_trainer.py is already the inner text tokenizer, so
        use that as the reliable fallback.
        """
        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is not None:
            return tokenizer

        processing_class = self.processing_class
        return getattr(processing_class, "tokenizer", processing_class)

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
        tokenizer = self._tq_tokenizer()
        stop_ids = _tq_question_generation_eos_token_ids(tokenizer)
        old_eos_token_id = getattr(self.generation_config, "eos_token_id", None)
        self.generation_config.eos_token_id = stop_ids
        try:
            return super()._generate_single_turn(
                prompt_ids,
                images,
                multimodal_fields,
            )
        finally:
            self.generation_config.eos_token_id = old_eos_token_id

    def _tq_generate_single_turn(self, prompt_ids, images=None, multimodal_fields=None):
        """Call Unsloth/TRL generation without assuming a fixed return arity.

        Upstream TRL 0.29 returns ``(completion_ids, logprobs)`` from
        ``_generate_single_turn``.  Some Unsloth-compiled variants wrap that
        method and return additional backend metadata.  For this TQ trainer we
        explicitly require ``use_vllm=False``, so only the generated token IDs
        are needed for post-tool turns and sampling log-probs are intentionally
        unused.

        The first returned element must therefore be the completion-id batch.
        Any additional elements are preserved only for diagnostics and ignored.
        """
        result = self._generate_single_turn(
            prompt_ids,
            images,
            multimodal_fields or {},
        )

        if isinstance(result, tuple):
            if not result:
                raise RuntimeError(
                    "_generate_single_turn returned an empty tuple."
                )
            completion_ids = result[0]
            return_arity = len(result)
        elif isinstance(result, list):
            # A plain list is interpreted directly as a batch of completion IDs.
            completion_ids = result
            return_arity = 1
        else:
            raise RuntimeError(
                "Unexpected _generate_single_turn return type: "
                f"{type(result).__name__}."
            )

        if not hasattr(self, "_tq_generate_signature_debug_done"):
            self._tq_generate_signature_debug_done = False

        if (
            bool(getattr(self, "tq_debug_rollout", True))
            and not self._tq_generate_signature_debug_done
        ):
            print(
                "\nTQ GENERATION BACKEND COMPATIBILITY:"
                f" _generate_single_turn returned arity={return_arity}; "
                "using element 0 as completion_ids. "
                "Post-tool sampling logprobs are ignored because "
                "TQGRPOTrainer requires use_vllm=False.",
                flush=True,
            )
            self._tq_generate_signature_debug_done = True

        # Transformers generation path does not use sampling log-probs.
        return completion_ids, None

    def _decode_ids(self, ids) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self._tq_tokenizer().decode(
            list(ids),
            skip_special_tokens=False,
        )

    def _encode_text(self, text: str) -> list[int]:
        return self._tq_tokenizer().encode(
            text,
            add_special_tokens=False,
        )

    def _parse_ids(self, ids: list[int]) -> dict:
        """Parse generated IDs using the patched tokenizer parser when present."""
        tokenizer = self._tq_tokenizer()
        raw_text = tokenizer.decode(ids, skip_special_tokens=False)

        # llm_backend.patch_gemma_parse_response patches this method and is the
        # path we want because it keeps one ask_question and clears content for
        # future context while leaving the raw IDs untouched for the loss.
        if hasattr(tokenizer, "parse_response"):
            try:
                parsed = tokenizer.parse_response(raw_text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        # Fallback to TRL's parser for compatibility with unpatched tokenizers.
        try:
            parsed = trl_parse_response(tokenizer, ids)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _validate_tq_training_configuration(self):
        """Fail loudly for settings this causal per-turn loss does not support."""
        problems = []

        if bool(getattr(self, "use_vllm", False)):
            problems.append("use_vllm must be false")
        if bool(getattr(self, "use_liger_kernel", False)):
            problems.append("use_liger_kernel must be false")
        if float(getattr(self, "beta", 0.0)) != 0.0:
            problems.append("beta must be 0.0")
        if int(getattr(self, "num_iterations", 1)) != 1:
            problems.append("num_iterations must be 1")
        if str(getattr(self, "importance_sampling_level", "token")) != "token":
            problems.append("importance_sampling_level must be 'token'")
        if getattr(self, "off_policy_mask_threshold", None) is not None:
            problems.append("off_policy_mask_threshold must be None")
        if float(getattr(self, "top_entropy_quantile", 1.0)) != 1.0:
            problems.append("top_entropy_quantile must be 1.0")
        if bool(getattr(self, "_is_vlm", False)):
            problems.append("multimodal/VLM inputs are not supported by this TQ loss")

        supported_losses = {"grpo", "dapo", "bnpo", "dr_grpo"}
        if str(getattr(self, "loss_type", "grpo")) not in supported_losses:
            problems.append(
                "loss_type must be one of " + ", ".join(sorted(supported_losses))
            )

        # In this aligned on-policy regime TRL itself skips storing old-policy
        # log-probs.  That lets us use current_logp.detach() turn-by-turn, which
        # is exactly the standard mu=1 GRPO objective under the *correct* context.
        steps_per_generation = int(
            getattr(self.args, "steps_per_generation", 1) or 1
        )
        gradient_accumulation_steps = int(
            getattr(self.args, "gradient_accumulation_steps", 1) or 1
        )
        generate_every = steps_per_generation * int(
            getattr(self, "num_iterations", 1)
        )
        if gradient_accumulation_steps % generate_every != 0:
            problems.append(
                "gradient_accumulation_steps must be divisible by "
                "steps_per_generation * num_iterations so the rollout is on-policy"
            )

        if problems:
            raise ValueError(
                "TQGRPOTrainer per-turn loss currently requires:\n- "
                + "\n- ".join(problems)
            )

    # ------------------------------------------------------------------
    # Generation truncation: keep the prefix + first valid ask_question only
    # ------------------------------------------------------------------
    def _truncate_to_first_ask_question(
        self,
        ids: list[int],
    ) -> tuple[list[int], bool]:
        """Keep everything through the first complete valid ask_question.

        Any pre-tool prefix is intentionally retained. Anything after the first
        complete valid ask_question (hallucinated tool response, another
        question, prose, etc.) is discarded and therefore receives no GRPO loss.

        If no valid complete call exists, keep the full generation so malformed
        behavior can still receive the trajectory's negative format/missing-turn
        signal.
        """
        ids = list(ids)
        if not ids:
            return ids, False

        raw_text = self._decode_ids(ids)
        match = _GEMMA_COMPLETE_ASK_QUESTION_RE.search(raw_text)
        if match is None:
            return ids, False

        prefix_text = raw_text[: match.end()]
        prefix_ids = self._encode_text(prefix_text)

        if (
            len(prefix_ids) <= len(ids)
            and ids[: len(prefix_ids)] == prefix_ids
        ):
            parsed_prefix = self._parse_ids(prefix_ids)
            if _is_valid_ask_question(parsed_prefix):
                return prefix_ids, len(prefix_ids) < len(ids)

        # Some tokenizers do not round-trip generated special-token text exactly
        # through encode(decode(ids)). Map the regex character boundary back to a
        # generated-token boundary by decoding generated prefixes instead.
        for cutoff in range(1, len(ids) + 1):
            candidate_ids = ids[:cutoff]
            if self._decode_ids(candidate_ids) != prefix_text:
                continue
            if _is_valid_ask_question(self._parse_ids(candidate_ids)):
                return candidate_ids, cutoff < len(ids)

        # Conservative token-boundary fallback.
        end_marker_ids = self._encode_text("<tool_call|>")
        if not end_marker_ids:
            raise RuntimeError(
                "Could not tokenize Gemma tool-call end marker '<tool_call|>'."
            )

        marker_len = len(end_marker_ids)
        for start in range(len(ids) - marker_len + 1):
            if ids[start : start + marker_len] != end_marker_ids:
                continue
            cutoff = start + marker_len
            candidate_ids = ids[:cutoff]
            if _is_valid_ask_question(self._parse_ids(candidate_ids)):
                return candidate_ids, cutoff < len(ids)

        raise RuntimeError(
            "Found a complete serialized ask_question in generated text but "
            "could not map its end to a safe token boundary."
        )

    def _truncate_generated_batch(self, generated_ids, generated_logprobs):
        raw_ids = [list(ids) for ids in generated_ids]
        truncated_flags = []

        for idx, ids in enumerate(raw_ids):
            kept_ids, was_truncated = self._truncate_to_first_ask_question(ids)
            generated_ids[idx] = kept_ids
            truncated_flags.append(was_truncated)

            if generated_logprobs is not None:
                generated_logprobs[idx] = generated_logprobs[idx][: len(kept_ids)]

        return raw_ids, generated_ids, generated_logprobs, truncated_flags

    def _decoded_prefix_token_index(self, ids: list[int], prefix_text: str):
        if prefix_text == "":
            return 0

        encoded_prefix = self._encode_text(prefix_text)
        if (
            len(encoded_prefix) <= len(ids)
            and ids[: len(encoded_prefix)] == encoded_prefix
        ):
            return len(encoded_prefix)

        for cutoff in range(1, len(ids) + 1):
            if self._decode_ids(ids[:cutoff]) == prefix_text:
                return cutoff

        return None

    def _final_tool_call_loss_mask(self, ids: list[int]) -> list[int]:
        ids = list(ids)
        if not ids or self.tq_rewarded_token_mode == "all":
            return [1] * len(ids)

        raw_text = self._decode_ids(ids)
        match = _GEMMA_COMPLETE_ASK_QUESTION_RE.search(raw_text)
        if match is None:
            # No valid final output exists; keep the legacy all-token mask so
            # malformed generations can still receive the format penalty.
            return [1] * len(ids)

        start = match.start()
        marker = "<|tool_call>"
        marker_pos = raw_text.rfind(marker, 0, match.start())
        if marker_pos >= 0:
            start = marker_pos
        end = match.end()

        start_idx = self._decoded_prefix_token_index(ids, raw_text[:start])
        end_idx = self._decoded_prefix_token_index(ids, raw_text[:end])
        if start_idx is None or end_idx is None or not start_idx < end_idx:
            # Conservative fallback: do not silently drop all learning signal if
            # a tokenizer cannot map decoded text back to generated tokens.
            return [1] * len(ids)

        mask = [0] * len(ids)
        for index in range(start_idx, min(end_idx, len(mask))):
            mask[index] = 1
        return mask

    # ------------------------------------------------------------------
    # Environment action execution hook
    # ------------------------------------------------------------------
    def _tq_prepare_action_messages(
        self,
        *,
        batch_idx,
        parsed_current,
        prompts,
        completions,
    ):
        parsed = parsed_current[batch_idx]
        calls = (parsed.get("tool_calls") or [])[:1]
        tool_call = calls[0]
        function = tool_call.get("function", {})
        name = function.get("name", "unknown")
        arguments = function.get("arguments", {})

        # Structured future history deliberately contains no reasoning.
        sanitized_assistant = {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call],
        }
        prompts[batch_idx].append(sanitized_assistant)
        if (
            not completions[batch_idx]
            or completions[batch_idx][-1].get("role") != "assistant"
        ):
            completions[batch_idx].append(sanitized_assistant)
        else:
            completions[batch_idx][-1] = sanitized_assistant

        return tool_call, name, arguments

    def _tq_append_action_result(
        self,
        *,
        batch_idx,
        prompts,
        completions,
        name,
        result,
        tool_succeeded,
        arguments,
        action_number,
        horizon,
        debug_enabled,
        debug_sample_idx,
    ):
        content = result if isinstance(result, list) else str(result)
        tool_message = {
            "role": "tool",
            "name": name,
            "content": content,
        }
        prompts[batch_idx].append(tool_message)
        completions[batch_idx].append(tool_message)

        if debug_enabled and batch_idx == debug_sample_idx:
            question = (
                arguments.get("question")
                if isinstance(arguments, dict)
                else None
            )
            env = (
                self.environments[batch_idx]
                if self.environments is not None
                else None
            )
            step_reward = None
            active_count = None
            if (
                tool_succeeded
                and env is not None
                and getattr(env, "step_debug", None)
            ):
                last_step = env.step_debug[-1]
                step_reward = last_step.get("reward")
                active_count = last_step.get("active_count")

            print(
                "\n" + "-" * 110
                + f"\nEXECUTED ENVIRONMENT ACTION {action_number}/{horizon}"
                + f"\nquestion={question!r}"
                + f"\nREAL ANSWER (NOT REWARDED)={content!r}"
                + (
                    f"\ninformation step reward contributing to trajectory="
                    f"{float(step_reward):.6f}"
                    if step_reward is not None
                    else ""
                )
                + (
                    f"\nactive hypotheses after action={active_count}"
                    if active_count is not None
                    else ""
                )
                + (
                    "\nTOOL FAILED: rollout stops before another model turn."
                    if not tool_succeeded
                    else ""
                )
                + "\n" + "-" * 110,
                flush=True,
            )

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
        next_active_candidates = []
        tool_call_count = 0
        tool_failure_count = 0

        for batch_idx in active:
            _, name, arguments = self._tq_prepare_action_messages(
                batch_idx=batch_idx,
                parsed_current=parsed_current,
                prompts=prompts,
                completions=completions,
            )

            tool_call_count += 1
            tool_succeeded = True
            try:
                if name in self._sync_tool_dicts[batch_idx]:
                    result = self._sync_tool_dicts[batch_idx][name](**arguments)
                elif name in self._async_tool_dicts[batch_idx]:
                    future = asyncio.run_coroutine_threadsafe(
                        self._async_tool_dicts[batch_idx][name](**arguments),
                        self.async_loop,
                    )
                    result = future.result()
                else:
                    raise ValueError(f"Tool {name} not found.")
            except Exception as exc:
                tool_failure_count += 1
                tool_succeeded = False
                result = {"error": str(exc)}

            self._tq_append_action_result(
                batch_idx=batch_idx,
                prompts=prompts,
                completions=completions,
                name=name,
                result=result,
                tool_succeeded=tool_succeeded,
                arguments=arguments,
                action_number=action_number,
                horizon=horizon,
                debug_enabled=debug_enabled,
                debug_sample_idx=debug_sample_idx,
            )

            # A failed environment action terminates this rollout. The
            # already-generated reasoning/question segment remains in the
            # loss and receives the resulting invalid/missing-turn penalty,
            # but we do not generate another turn from an error state.
            if tool_succeeded:
                env = self.environments[batch_idx]
                if not bool(getattr(env, "solved", False)):
                    next_active_candidates.append(batch_idx)

        return next_active_candidates, tool_call_count, tool_failure_count

    def _tq_execute_continuation_questions(self, branch_items):
        """Execute independent continuation questions on cloned environments."""
        results = []
        failures = 0
        for env, question in branch_items:
            try:
                result = env.ask_question(question)
                results.append((result, True))
            except Exception as exc:
                env.invalid = True
                results.append(({"error": str(exc)}, False))
                failures += 1
        return results, len(branch_items), failures

    def _tq_rollout_k2_m_continuations(
        self,
        *,
        next_active_candidates,
        prompts,
        completions,
        completion_ids,
        tool_mask,
        logprobs,
        debug_enabled,
        debug_sample_idx,
    ):
        """Estimate Q1 return with M independent Q2 samples (K=2 only)."""
        M = int(self.tq_continuation_samples)
        if not next_active_candidates:
            return 0, 0

        clean_prompts = [prompts[idx] for idx in next_active_candidates]
        next_context_ids, _, next_multimodal_fields = self._tokenize_prompts(
            clean_prompts
        )
        if next_multimodal_fields:
            raise RuntimeError("Unexpected multimodal fields in TQ text-only rollout.")

        model_config = getattr(self.model, "config", None)
        max_model_len = int(
            getattr(model_config, "max_position_embeddings", 10**9)
        )
        kept_roots = [
            (batch_idx, ids)
            for batch_idx, ids in zip(
                next_active_candidates, next_context_ids, strict=True
            )
            if len(ids) < max_model_len
        ]
        if not kept_roots:
            return 0, 0

        expanded_context_ids = []
        branch_meta = []
        for batch_idx, ids in kept_roots:
            for sample_idx in range(M):
                expanded_context_ids.append(list(ids))
                branch_meta.append((batch_idx, sample_idx))

        post_ids, post_logprobs = self._tq_generate_single_turn(
            expanded_context_ids, None, {}
        )
        raw_post, post_ids, post_logprobs, truncated_flags = (
            self._truncate_generated_batch(post_ids, post_logprobs)
        )

        branches = []
        executable = []
        executable_positions = []
        for branch_pos, (batch_idx, sample_idx) in enumerate(branch_meta):
            remaining = max(
                0,
                int(self.max_completion_length) - len(completion_ids[batch_idx]),
            )
            kept = list(post_ids[branch_pos][:remaining])
            branch_logprobs = (
                post_logprobs[branch_pos][:remaining]
                if post_logprobs is not None
                else None
            )
            parsed = self._parse_ids(kept) if kept else {}
            branch_env = copy.deepcopy(self.environments[batch_idx])
            record = {
                "batch_idx": batch_idx,
                "sample_idx": sample_idx,
                "kept": kept,
                "logprobs": branch_logprobs,
                "parsed": parsed,
                "env": branch_env,
                "result": None,
                "tool_succeeded": False,
                "raw": raw_post[branch_pos],
                "truncated": truncated_flags[branch_pos],
            }
            branches.append(record)

            if _is_valid_ask_question(parsed):
                function = (parsed.get("tool_calls") or [])[0].get("function", {})
                arguments = function.get("arguments", {})
                question = arguments.get("question") if isinstance(arguments, dict) else None
                executable.append((branch_env, str(question or "")))
                executable_positions.append(branch_pos)

        executed_results, tool_calls, tool_failures = (
            self._tq_execute_continuation_questions(executable)
        )
        for branch_pos, (result, succeeded) in zip(
            executable_positions, executed_results, strict=True
        ):
            branches[branch_pos]["result"] = result
            branches[branch_pos]["tool_succeeded"] = succeeded

        for batch_idx, _ in kept_roots:
            root_branches = [b for b in branches if b["batch_idx"] == batch_idx]
            blau_values = [b["env"]._get_blau_reward() for b in root_branches]
            format_values = [b["env"]._get_format_reward() for b in root_branches]
            self.environments[batch_idx]._set_reward_overrides(
                blau=sum(blau_values) / len(blau_values),
                fmt=sum(format_values) / len(format_values),
            )

            # Keep one representative Q2 in TRL's flat completion plumbing.
            # The custom loss ignores it; all M branches contribute only via
            # the averaged environment reward above.
            representative = root_branches[0]
            kept = representative["kept"]
            completion_ids[batch_idx].extend(kept)
            tool_mask[batch_idx].extend([1] * len(kept))
            if logprobs is not None and representative["logprobs"] is not None:
                logprobs[batch_idx].extend(representative["logprobs"])

            parsed = representative["parsed"]
            if parsed:
                completions[batch_idx].append(parsed)

            if debug_enabled and batch_idx == debug_sample_idx:
                branch_summary = [
                    {
                        "m": b["sample_idx"],
                        "question": (
                            ((b["parsed"].get("tool_calls") or [{}])[0].get("function", {})
                             .get("arguments", {})
                             .get("question"))
                            if isinstance(b["parsed"], dict)
                            else None
                        ),
                        "step_rewards": list(b["env"].step_rewards),
                        "blau_return": float(b["env"]._get_blau_reward()),
                        "solved": bool(getattr(b["env"], "solved", False)),
                    }
                    for b in root_branches
                ]
                print(
                    "\n" + "=" * 110
                    + f"\nK=2 MONTE-CARLO CONTINUATIONS: M={M}"
                    + f"\nbranches={branch_summary!r}"
                    + f"\naveraged_blau_return="
                    + f"{self.environments[batch_idx]._get_blau_reward():.6f}"
                    + f"\naveraged_format_reward="
                    + f"{self.environments[batch_idx]._get_format_reward():.6f}"
                    + "\n" + "=" * 110,
                    flush=True,
                )

        return tool_calls, tool_failures

    # ------------------------------------------------------------------
    # Strict K-turn rollout
    # ------------------------------------------------------------------
    def _tool_call_loop(
        self,
        prompts,
        prompt_ids,
        completion_ids,
        completions,
        logprobs,
        images,
        multimodal_fields,
    ):
        """Execute the TQ rollout with terminal stopping and root-only credit."""
        del images, multimodal_fields  # guarded text-only implementation

        batch_size = len(completion_ids)
        K = int(self.max_tool_calling_iterations)
        debug_enabled = bool(getattr(self, "tq_debug_rollout", True)) and not bool(
            getattr(self, "_tq_rollout_debug_done", False)
        )
        # Print exactly ONE rollout sample, once: trajectory 0 from the first
        # generated training batch. This affects logging only.
        debug_sample_idx = 0

        # One list of loss segments per rollout. Root-only credit assignment:
        # only turn 1 is registered here and therefore receives GRPO gradient.
        # Turns 2..K are still generated/executed to estimate the K-step return,
        # but are deliberately omitted from the differentiable loss segments.
        turn_segments: list[list[dict[str, list[int]]]] = [
            [] for _ in range(batch_size)
        ]

        # Initial generation was produced by parent _generate().  Sanitize it
        # before it can enter either the loss record or structured history.
        initial_raw = [list(ids) for ids in completion_ids]
        for batch_idx in range(batch_size):
            kept_ids, was_truncated = self._truncate_to_first_ask_question(
                initial_raw[batch_idx]
            )
            completion_ids[batch_idx] = kept_ids
            if logprobs is not None:
                logprobs[batch_idx] = logprobs[batch_idx][: len(kept_ids)]

            turn_segments[batch_idx].append(
                {
                    "context_ids": list(prompt_ids[batch_idx]),
                    "generated_ids": list(kept_ids),
                    "loss_mask": self._final_tool_call_loss_mask(kept_ids),
                }
            )

            parsed_kept = self._parse_ids(kept_ids) if kept_ids else {}
            if parsed_kept:
                completions[batch_idx] = [parsed_kept]

            if debug_enabled and batch_idx == debug_sample_idx:
                print(
                    "\n" + "#" * 110
                    + f"\nTQ CAUSAL MULTI-TURN ROLLOUT DEBUG — SINGLE SAMPLE trajectory={debug_sample_idx}"
                    + f"\nK={K} M={self.tq_continuation_samples}"
                    + "\n" + "#" * 110
                    + "\n\nTURN 1 CONTEXT SEEN BY MODEL (NOT REWARDED):\n"
                    + self._decode_ids(prompt_ids[batch_idx])
                    + "\n\nTURN 1 RAW MODEL OUTPUT:\n"
                    + repr(self._decode_ids(initial_raw[batch_idx]))
                    + "\n\nTURN 1 MODEL TOKENS KEPT FOR CAUSAL SCORING "
                    + f"({self.tq_first_turn_kept_description}):\n"
                    + repr(self._decode_ids(kept_ids))
                    + f"\nraw_tokens={len(initial_raw[batch_idx])} "
                    + "rewarded_tokens="
                    + f"{sum(self._final_tool_call_loss_mask(kept_ids))} "
                    + f"truncated_after_first_question={was_truncated}",
                    flush=True,
                )

        # Flat completion_ids are retained only for TRL reward/logging plumbing.
        # They contain MODEL tokens only.  The custom _compute_loss below never
        # uses this flattened representation.
        tool_mask = [[1] * len(ids) for ids in completion_ids]
        tool_images = [[] for _ in range(batch_size)]
        tool_call_count = 0
        tool_failure_count = 0

        parsed_current = [
            self._parse_ids(list(ids)) if ids else {}
            for ids in completion_ids
        ]
        current_tool_calls = [
            (parsed.get("tool_calls") or [])[:1]
            if isinstance(parsed, dict)
            else []
            for parsed in parsed_current
        ]
        active = [
            idx
            for idx, calls in enumerate(current_tool_calls)
            if calls and _is_valid_ask_question(parsed_current[idx])
        ]

        executed_turn = 0
        while active and executed_turn < K:
            action_number = executed_turn + 1

            # Execute exactly one accepted tool call for each active rollout.
            (
                next_active_candidates,
                action_tool_call_count,
                action_tool_failure_count,
            ) = self._tq_execute_actions(
                active=active,
                parsed_current=parsed_current,
                prompts=prompts,
                completions=completions,
                action_number=action_number,
                horizon=K,
                debug_enabled=debug_enabled,
                debug_sample_idx=debug_sample_idx,
            )
            tool_call_count += action_tool_call_count
            tool_failure_count += action_tool_failure_count

            executed_turn += 1

            # For the current K=2 experiment, M>1 means: keep this exact Q1/A1
            # fixed, sample M independent Q2 continuations, average their returns,
            # and attach that Monte-Carlo estimate to the single root action.
            if K == 2 and self.tq_continuation_samples > 1 and executed_turn == 1:
                continuation_calls, continuation_failures = (
                    self._tq_rollout_k2_m_continuations(
                        next_active_candidates=next_active_candidates,
                        prompts=prompts,
                        completions=completions,
                        completion_ids=completion_ids,
                        tool_mask=tool_mask,
                        logprobs=logprobs,
                        debug_enabled=debug_enabled,
                        debug_sample_idx=debug_sample_idx,
                    )
                )
                tool_call_count += continuation_calls
                tool_failure_count += continuation_failures
                break

            # Critical horizon rule: after A_K, stop immediately.  There is no
            # generation K+1, so no unexecuted model tokens can receive reward.
            if executed_turn >= K:
                if debug_enabled and debug_sample_idx in active:
                    print(
                        "\n" + "=" * 110
                        + f"\nSTOP AFTER K={K} EXECUTED QUESTIONS"
                        + "\nNO K+1 MODEL GENERATION IS CALLED."
                        + "\n" + "=" * 110,
                        flush=True,
                    )
                break

            # Re-render the actual next-turn model context FROM STRUCTURED Q/A
            # HISTORY.  Because assistant content was forced to "", reasoning
            # from earlier turns is absent by construction.
            clean_prompts = [prompts[idx] for idx in next_active_candidates]
            next_context_ids, _, next_multimodal_fields = self._tokenize_prompts(
                clean_prompts
            )

            # Text-only guard means no multimodal fields should be present.
            if next_multimodal_fields:
                raise RuntimeError(
                    "Unexpected multimodal fields in TQ text-only rollout."
                )

            model_config = getattr(self.model, "config", None)
            max_model_len = int(
                getattr(model_config, "max_position_embeddings", 10**9)
            )
            keep_local = [
                local_idx
                for local_idx, ids in enumerate(next_context_ids)
                if len(ids) < max_model_len
            ]
            if len(keep_local) != len(next_active_candidates):
                next_active_candidates = [
                    next_active_candidates[i] for i in keep_local
                ]
                next_context_ids = [next_context_ids[i] for i in keep_local]

            if not next_active_candidates:
                break

            if debug_enabled and debug_sample_idx in next_active_candidates:
                local_idx = next_active_candidates.index(debug_sample_idx)
                print(
                    "\n" + "=" * 110
                    + f"\nTURN {executed_turn + 1} EXACT CONTEXT SEEN BY MODEL "
                    + "(NOT REWARDED)"
                    + "\nThis context is freshly rendered from Q/A history; prior reasoning is absent."
                    + "\n" + "-" * 110
                    + "\n"
                    + self._decode_ids(next_context_ids[local_idx])
                    + "\n" + "=" * 110,
                    flush=True,
                )

            post_ids, post_logprobs = self._tq_generate_single_turn(
                next_context_ids,
                None,
                {},
            )
            raw_post, post_ids, post_logprobs, truncated_flags = (
                self._truncate_generated_batch(post_ids, post_logprobs)
            )

            parsed_post = []
            for local_idx, batch_idx in enumerate(next_active_candidates):
                # Respect the same total MODEL-token budget as max_completion_length.
                remaining = max(
                    0,
                    int(self.max_completion_length)
                    - len(completion_ids[batch_idx]),
                )
                if len(post_ids[local_idx]) > remaining:
                    post_ids[local_idx] = post_ids[local_idx][:remaining]
                    if post_logprobs is not None:
                        post_logprobs[local_idx] = post_logprobs[local_idx][:remaining]

                kept = list(post_ids[local_idx])
                completion_ids[batch_idx].extend(kept)
                tool_mask[batch_idx].extend([1] * len(kept))
                if logprobs is not None and post_logprobs is not None:
                    logprobs[batch_idx].extend(post_logprobs[local_idx])

                # Root-only credit assignment: this later turn contributes to
                # the environment return but is rollout-only. Do not register a
                # loss segment, so no differentiable scoring forward/backward is
                # performed for turns 2..K.
                parsed = self._parse_ids(kept) if kept else {}
                parsed_post.append(parsed)
                if parsed:
                    completions[batch_idx].append(parsed)

                if debug_enabled and batch_idx == debug_sample_idx:
                    print(
                        "\n" + "=" * 110
                        + f"\nTURN {executed_turn + 1} RAW MODEL OUTPUT:\n"
                        + repr(self._decode_ids(raw_post[local_idx]))
                        + f"\n\nTURN {executed_turn + 1} MODEL TOKENS KEPT FOR ROLLOUT ONLY "
                        + "(NO GRPO LOSS / NO GRADIENT):\n"
                        + repr(self._decode_ids(kept))
                        + f"\nraw_tokens={len(raw_post[local_idx])} "
                        + f"rollout_only_tokens={len(kept)} "
                        + f"truncated_after_first_question={truncated_flags[local_idx]}"
                        + "\n" + "=" * 110,
                        flush=True,
                    )

            # Only valid next questions are executed on the following loop.
            active = []
            parsed_current = [{} for _ in range(batch_size)]
            for local_idx, batch_idx in enumerate(next_active_candidates):
                parsed_current[batch_idx] = parsed_post[local_idx]
                if _is_valid_ask_question(parsed_post[local_idx]):
                    active.append(batch_idx)

        self._tq_last_turn_segments = turn_segments
        if debug_enabled:
            self._tq_rollout_debug_done = True

        # Unsloth's compiled TRL 0.29 tool-loop caller in this project
        # expects exactly six return values.  The newer upstream TRL variant
        # additionally returns ``tool_images``, but Twenty Questions is
        # text-only and does not use multimodal tool responses.
        return (
            tool_mask,
            completions,
            completion_ids,
            logprobs,
            tool_call_count,
            tool_failure_count,
        )

    # ------------------------------------------------------------------
    # Pack per-turn causal contexts into tensors that survive TRL shuffle/split
    # ------------------------------------------------------------------
    def _pack_turn_segments(self, segments, device):
        batch_size = len(segments)
        K = int(self.max_tool_calling_iterations)
        tokenizer = self._tq_tokenizer()
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id

        max_context_len = max(
            [
                len(segment["context_ids"])
                for rollout in segments
                for segment in rollout[:K]
            ]
            or [1]
        )
        max_generated_len = max(
            [
                len(segment["generated_ids"])
                for rollout in segments
                for segment in rollout[:K]
            ]
            or [1]
        )
        max_generated_len = max(max_generated_len, 1)

        context_ids = torch.full(
            (batch_size, K, max_context_len),
            int(pad_id),
            dtype=torch.long,
            device=device,
        )
        context_mask = torch.zeros_like(context_ids)
        generated_ids = torch.full(
            (batch_size, K, max_generated_len),
            int(pad_id),
            dtype=torch.long,
            device=device,
        )
        generated_mask = torch.zeros_like(generated_ids)
        generated_loss_mask = torch.zeros_like(generated_ids)
        turn_valid = torch.zeros(
            (batch_size, K),
            dtype=torch.long,
            device=device,
        )

        for batch_idx, rollout in enumerate(segments):
            for turn_idx, segment in enumerate(rollout[:K]):
                ctx = segment["context_ids"]
                gen = segment["generated_ids"]
                loss = segment.get("loss_mask")
                if loss is None:
                    loss = [1] * len(gen)
                if len(loss) != len(gen):
                    raise RuntimeError(
                        "TQ loss mask and generated token segment must have "
                        f"equal length. Got loss={len(loss)}, generated={len(gen)}."
                    )

                if ctx:
                    context_ids[batch_idx, turn_idx, -len(ctx) :] = torch.tensor(
                        ctx, dtype=torch.long, device=device
                    )
                    context_mask[batch_idx, turn_idx, -len(ctx) :] = 1
                if gen:
                    generated_ids[batch_idx, turn_idx, : len(gen)] = torch.tensor(
                        gen, dtype=torch.long, device=device
                    )
                    generated_mask[batch_idx, turn_idx, : len(gen)] = 1
                    generated_loss_mask[
                        batch_idx, turn_idx, : len(loss)
                    ] = torch.tensor(
                        loss, dtype=torch.long, device=device
                    )
                    turn_valid[batch_idx, turn_idx] = 1

        return {
            "tq_turn_context_ids": context_ids,
            "tq_turn_context_mask": context_mask,
            "tq_turn_generated_ids": generated_ids,
            "tq_turn_generated_mask": generated_mask,
            "tq_turn_generated_loss_mask": generated_loss_mask,
            "tq_turn_valid": turn_valid,
        }

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)

        # This implementation relies on the on-policy mu=1 path.  If TRL ever
        # decides old/reference log-probs are required, the flat representation
        # would be causally wrong for later turns, so stop rather than proceed.
        if output.get("old_per_token_logps") is not None:
            raise RuntimeError(
                "TQGRPOTrainer received old_per_token_logps from TRL. "
                "This indicates an off-policy/misaligned setup that the "
                "per-turn loss intentionally does not support."
            )
        if output.get("ref_per_token_logps") is not None:
            raise RuntimeError(
                "TQGRPOTrainer does not support beta != 0/reference log-probs."
            )
        if output.get("sampling_per_token_logps") is not None:
            raise RuntimeError(
                "TQGRPOTrainer does not support external/vLLM sampling log-probs."
            )

        device = output["advantages"].device
        segments = self._tq_last_turn_segments
        if len(segments) != int(output["advantages"].shape[0]):
            raise RuntimeError(
                "Per-turn TQ rollout segments are not aligned with the GRPO batch: "
                f"segments={len(segments)}, batch={output['advantages'].shape[0]}."
            )

        output.update(self._pack_turn_segments(segments, device))

        # Attach the actual trajectory rewards so they remain aligned through
        # TRL's later shuffle/split and can be printed beside the advantage.
        if self.environments is not None:
            blau = []
            fmt = []
            total = []
            turns = []
            for env in self.environments[: len(segments)]:
                b = float(env._get_blau_reward()) if hasattr(env, "_get_blau_reward") else 0.0
                f = float(env._get_format_reward()) if hasattr(env, "_get_format_reward") else 0.0
                blau.append(b)
                fmt.append(f)
                total.append(b + f)
                turns.append(len(getattr(env, "step_rewards", [])))

            output["tq_blau_reward"] = torch.tensor(blau, dtype=torch.float32, device=device)
            output["tq_format_reward"] = torch.tensor(fmt, dtype=torch.float32, device=device)
            output["tq_total_reward"] = torch.tensor(total, dtype=torch.float32, device=device)
            output["tq_executed_turns"] = torch.tensor(turns, dtype=torch.long, device=device)

        return output

    # ------------------------------------------------------------------
    # GRPO loss evaluated only on the root turn's actual causal context
    # ------------------------------------------------------------------
    def _print_loss_debug(self, inputs):
        if not bool(getattr(self, "tq_debug_rollout", True)):
            return
        if bool(getattr(self, "_tq_loss_debug_done", False)):
            return

        print(
            "\n" + "#" * 110
            + "\nTQ GRPO LOSS DEBUG — WHAT RECEIVES THE TRAJECTORY ADVANTAGE"
            + "\n" + "#" * 110,
            flush=True,
        )

        # Match the single rollout sample printed above.
        sample = 0
        reward_text = ""
        if "tq_blau_reward" in inputs:
            reward_text = (
                f"\nInformation trajectory reward={float(inputs['tq_blau_reward'][sample]):.6f}"
                f"\nFormat reward={float(inputs['tq_format_reward'][sample]):.6f}"
                f"\nTotal trajectory reward={float(inputs['tq_total_reward'][sample]):.6f}"
                f"\nExecuted turns={int(inputs['tq_executed_turns'][sample])}"
            )
        advantage = float(inputs["advantages"][sample])
        print(
            reward_text
            + f"\nGRPO trajectory advantage applied to TURN 1 model tokens only={advantage:.6f}",
            flush=True,
        )

        K = inputs["tq_turn_valid"].shape[1]
        for turn_idx in range(K):
            if not bool(inputs["tq_turn_valid"][sample, turn_idx].item()):
                continue

            ctx_mask = inputs["tq_turn_context_mask"][sample, turn_idx].bool()
            gen_mask = inputs["tq_turn_generated_mask"][sample, turn_idx].bool()
            loss_mask = inputs[
                "tq_turn_generated_loss_mask"
            ][sample, turn_idx].bool()
            ctx_ids = inputs["tq_turn_context_ids"][sample, turn_idx][ctx_mask].tolist()
            gen_ids = inputs["tq_turn_generated_ids"][sample, turn_idx][gen_mask].tolist()
            rewarded_ids = inputs["tq_turn_generated_ids"][sample, turn_idx][
                loss_mask
            ].tolist()

            print(
                "\n" + "=" * 110
                + f"\nTURN {turn_idx + 1} LOSS VIEW"
                + "\n\nCONTEXT TOKENS — loss mask 0 (NOT REWARDED):\n"
                + self._decode_ids(ctx_ids)
                + "\n\nMODEL-GENERATED TOKENS — loss mask 1 "
                + f"({self.tq_rewarded_token_description}; RECEIVE GRPO ADVANTAGE):\n"
                + repr(self._decode_ids(rewarded_ids))
                + "\n\nFULL MODEL-GENERATED TOKENS IN CAUSAL SEQUENCE:\n"
                + repr(self._decode_ids(gen_ids))
                + f"\nrewarded_token_count={len(rewarded_ids)}"
                + f"\ngenerated_token_count={len(gen_ids)}"
                + "\n" + "=" * 110,
                flush=True,
            )

        print(
            "\nRoot-only credit assignment is active: turns 2..K are generated "
            "only to determine the trajectory return and receive no gradient. "
            "Environment answers never appear in the rewarded turn-1 segment."
            "\n" + "#" * 110,
            flush=True,
        )
        self._tq_loss_debug_done = True

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """Force the TQ per-turn causal loss even under Unsloth compilation.

        Unsloth's compiled GRPO trainer overrides ``compute_loss`` directly and
        can bypass upstream TRL's ``compute_loss -> self._compute_loss`` dispatch.
        Therefore the TQ subclass must override ``compute_loss`` itself.

        For TQ multi-turn batches, this routes explicitly to the root-only loss:
          context_1                     -> loss mask 0
          model segment through question_1 -> loss mask 1
          turns 2..K                    -> rollout only, no loss segment

        Outside the TQ path, normal parent behavior is preserved.
        """
        if return_outputs:
            raise ValueError(
                "TQGRPOTrainer does not support return_outputs=True."
            )

        if "tq_turn_generated_ids" in inputs:
            # Some Transformers/Unsloth versions pass this separately rather
            # than storing it in the prepared input dict.
            if (
                num_items_in_batch is not None
                and "num_items_in_batch" not in inputs
            ):
                inputs = dict(inputs)
                inputs["num_items_in_batch"] = num_items_in_batch

            if not bool(
                getattr(self, "_tq_compute_loss_dispatch_debug_done", False)
            ):
                print(
                    "\n"
                    + "#" * 110
                    + "\nTQ COMPUTE_LOSS OVERRIDE ACTIVE"
                    + "\nUsing per-turn causal GRPO loss; "
                    + "Unsloth flattened completion loss is BYPASSED."
                    + "\n"
                    + "#" * 110,
                    flush=True,
                )
                self._tq_compute_loss_dispatch_debug_done = True

            return self._compute_loss(model, inputs)

        # Preserve ordinary GRPO behavior for any non-TQ batch.
        try:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        except TypeError:
            # Compatibility with parent versions that do not accept
            # num_items_in_batch.
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
            )

    def _tq_differentiable_generated_logps(
        self,
        model,
        input_ids,
        attention_mask,
        generated_len,
    ):
        """Score generated tokens with a fresh autograd-enabled forward pass.

        We intentionally do NOT use Unsloth's compiled
        ``_get_per_token_logps_and_entropies`` here. In this installation that
        helper is an inference/memory-optimized path and returns tensors that
        are detached from autograd, which cannot drive the policy update.

        For a causal sequence

            [ context ][ generated_1 ... generated_G ]

        the logits that score the G generated targets are the logits at
        positions

            context_end-1 ... sequence_end-2.

        Gemma 4 supports ``logits_to_keep``. Requesting G+1 final logits and
        dropping the last one gives exactly those G causal prediction
        positions while avoiding logits for the full context.
        """
        generated_len = int(generated_len)
        if generated_len < 1:
            raise ValueError("generated_len must be >= 1.")

        # The current policy forward used for GRPO must participate in autograd.
        # Explicitly re-enable gradients in case an optimized generation helper
        # left us inside a no-grad/inference-style context.
        with torch.enable_grad():
            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": False,
                "logits_to_keep": generated_len + 1,
            }

            try:
                outputs = model(**forward_kwargs)
            except TypeError as exc:
                # Compatibility fallback for wrappers that do not expose
                # logits_to_keep even though the underlying Gemma model does.
                if "logits_to_keep" not in str(exc):
                    raise
                forward_kwargs.pop("logits_to_keep")
                outputs = model(**forward_kwargs)

            logits = outputs.logits

            if logits is None:
                raise RuntimeError(
                    "Differentiable TQ forward returned logits=None."
                )

            # If the wrapper ignored logits_to_keep, retain only the final
            # G+1 positions ourselves.
            needed = generated_len + 1
            if logits.size(1) < needed:
                raise RuntimeError(
                    "Model returned too few logits to score the generated "
                    f"segment: logits={tuple(logits.shape)}, "
                    f"needed_positions={needed}."
                )
            if logits.size(1) != needed:
                logits = logits[:, -needed:, :]

            # Logit at position t predicts token t+1. The final logit predicts
            # a token beyond the supplied sequence, so it is not used.
            causal_logits = logits[:, :-1, :]
            target_ids = input_ids[:, -generated_len:]

            if causal_logits.size(1) != generated_len:
                raise RuntimeError(
                    "Causal logit alignment failed: "
                    f"logits={tuple(causal_logits.shape)}, "
                    f"generated_len={generated_len}."
                )

            # Compute log p(target) without materializing a full log_softmax
            # tensor. The logits themselves remain connected to the model
            # parameters, so these selected log-probs carry gradients.
            logits_fp32 = causal_logits.float()
            target_logits = torch.gather(
                logits_fp32,
                dim=-1,
                index=target_ids.unsqueeze(-1),
            ).squeeze(-1)
            log_normalizer = torch.logsumexp(
                logits_fp32,
                dim=-1,
            )
            per_token_logps = target_logits - log_normalizer

        if not bool(
            getattr(self, "_tq_autograd_logprob_debug_done", False)
        ):
            print(
                "\n"
                + "#" * 110
                + "\nTQ DIFFERENTIABLE LOGPROB DEBUG"
                + f"\ntorch.is_grad_enabled()={torch.is_grad_enabled()}"
                + f"\nmodel.training={bool(model.training)}"
                + f"\nraw_logits_shape={tuple(logits.shape)}"
                + f"\ncausal_logits_shape={tuple(causal_logits.shape)}"
                + f"\nper_token_logps_shape={tuple(per_token_logps.shape)}"
                + f"\nlogits.requires_grad={bool(logits.requires_grad)}"
                + f"\nper_token_logps.requires_grad="
                  f"{bool(per_token_logps.requires_grad)}"
                + "\nExpected: logits.requires_grad=True and "
                  "per_token_logps.requires_grad=True."
                + "\n"
                + "#" * 110,
                flush=True,
            )
            self._tq_autograd_logprob_debug_done = True

        if model.training and not per_token_logps.requires_grad:
            raise RuntimeError(
                "TQ differentiable log-prob forward is still detached from "
                "autograd. Refusing to return a zero-gradient GRPO loss."
            )

        # Entropy is not required for the TQ objective. Returning None avoids
        # materializing another vocabulary-sized probability tensor.
        return per_token_logps, None

    def _compute_loss(self, model, inputs):
        # Preserve normal TRL behavior outside the TQ multi-turn path.
        if "tq_turn_generated_ids" not in inputs:
            return super()._compute_loss(model, inputs)

        self._print_loss_debug(inputs)

        advantages = inputs["advantages"]
        if advantages.dim() != 1:
            raise RuntimeError(
                f"Expected trajectory-level advantages with shape (B,), got {tuple(advantages.shape)}."
            )

        batch_size = advantages.shape[0]
        K = inputs["tq_turn_valid"].shape[1]
        device = advantages.device

        seq_loss_sum = torch.zeros(
            batch_size,
            dtype=torch.float32,
            device=device,
        )
        seq_token_count = torch.zeros(
            batch_size,
            dtype=torch.float32,
            device=device,
        )

        entropy_sum = torch.zeros((), dtype=torch.float32, device=device)
        entropy_token_count = torch.zeros((), dtype=torch.float32, device=device)
        total_token_count = torch.zeros((), dtype=torch.float32, device=device)
        low_clip_count = torch.zeros((), dtype=torch.float32, device=device)
        high_clip_count = torch.zeros((), dtype=torch.float32, device=device)
        region_clip_count = torch.zeros((), dtype=torch.float32, device=device)

        for turn_idx in range(K):
            valid = inputs["tq_turn_valid"][:, turn_idx].bool()
            if not bool(valid.any()):
                continue

            indices = valid.nonzero(as_tuple=True)[0]
            context_ids = inputs["tq_turn_context_ids"][indices, turn_idx]
            context_mask = inputs["tq_turn_context_mask"][indices, turn_idx]
            generated_ids = inputs["tq_turn_generated_ids"][indices, turn_idx]
            generated_mask = inputs["tq_turn_generated_mask"][indices, turn_idx]
            generated_loss_mask = inputs["tq_turn_generated_loss_mask"][
                indices, turn_idx
            ]

            input_ids = torch.cat([context_ids, generated_ids], dim=1)
            attention_mask = torch.cat([context_mask, generated_mask], dim=1)
            logits_to_keep = generated_ids.size(1)

            per_token_logps, entropies = self._tq_differentiable_generated_logps(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                generated_len=logits_to_keep,
            )

            # The differentiable TQ scorer is designed to return exactly
            # one log-prob per generated target token. Keep the checks below
            # as hard guards against any future model-wrapper shape change.
            generated_len = generated_ids.size(1)
            returned_logp_len = per_token_logps.size(1)

            if returned_logp_len < generated_len:
                raise RuntimeError(
                    "Per-token log-prob output is shorter than the generated "
                    "segment: "
                    f"returned={returned_logp_len}, generated={generated_len}."
                )

            if returned_logp_len != generated_len:
                if not bool(
                    getattr(self, "_tq_logprob_alignment_debug_done", False)
                ):
                    print(
                        "\nTQ LOGPROB ALIGNMENT COMPATIBILITY:"
                        f" helper_return_shape={tuple(per_token_logps.shape)}, "
                        f"generated_shape={tuple(generated_ids.shape)}; "
                        f"keeping final {generated_len} positions so the "
                        "loss is aligned only to generated reasoning/question "
                        "tokens.",
                        flush=True,
                    )
                    self._tq_logprob_alignment_debug_done = True

                per_token_logps = per_token_logps[:, -generated_len:]

            if entropies is not None:
                returned_entropy_len = entropies.size(1)
                if returned_entropy_len < generated_len:
                    raise RuntimeError(
                        "Entropy output is shorter than the generated segment: "
                        f"returned={returned_entropy_len}, generated={generated_len}."
                    )
                if returned_entropy_len != generated_len:
                    entropies = entropies[:, -generated_len:]

            if per_token_logps.shape != generated_mask.shape:
                raise RuntimeError(
                    "Generated-token/log-prob alignment failed after slicing: "
                    f"logps={tuple(per_token_logps.shape)}, "
                    f"mask={tuple(generated_mask.shape)}."
                )

            if entropies is not None and entropies.shape != generated_mask.shape:
                raise RuntimeError(
                    "Generated-token/entropy alignment failed after slicing: "
                    f"entropies={tuple(entropies.shape)}, "
                    f"mask={tuple(generated_mask.shape)}."
                )

            # On-policy mu=1 GRPO: old log-prob equals the rollout policy value
            # numerically, but is detached so the ratio still carries gradient.
            old_per_token_logps = per_token_logps.detach()
            log_ratio = per_token_logps - old_per_token_logps
            coef_1 = torch.exp(log_ratio)
            coef_2 = torch.clamp(
                coef_1,
                1 - self.epsilon_low,
                1 + self.epsilon_high,
            )
            if getattr(self.args, "delta", None) is not None:
                coef_1 = torch.clamp(coef_1, max=float(self.args.delta))

            turn_advantages = advantages[indices].unsqueeze(1)
            per_token_loss1 = coef_1 * turn_advantages
            per_token_loss2 = coef_2 * turn_advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

            mask = (generated_mask * generated_loss_mask).to(per_token_loss.dtype)
            turn_loss_sum = (per_token_loss * mask).sum(dim=1)
            turn_token_count = mask.sum(dim=1)

            # Functional index_add preserves autograd from turn_loss_sum.
            seq_loss_sum = seq_loss_sum.index_add(0, indices, turn_loss_sum.float())
            seq_token_count = seq_token_count.index_add(
                0, indices, turn_token_count.float()
            )

            # Some Unsloth-compiled generation/model helpers return
            # entropies=None even when compute_entropy=True. Entropy is only a
            # diagnostic metric here; it is NOT part of the TQ GRPO objective.
            # Therefore accumulate it only when it is actually available.
            if entropies is not None:
                entropy_sum = entropy_sum + (entropies.float() * mask).sum()
                entropy_token_count = entropy_token_count + mask.sum()

            total_token_count = total_token_count + mask.sum()

            low_clipped = (coef_1 < 1 - self.epsilon_low) & (turn_advantages < 0)
            high_clipped = (coef_1 > 1 + self.epsilon_high) & (turn_advantages > 0)
            region_clipped = low_clipped | high_clipped
            low_clip_count = low_clip_count + (low_clipped.float() * mask).sum()
            high_clip_count = high_clip_count + (high_clipped.float() * mask).sum()
            region_clip_count = region_clip_count + (region_clipped.float() * mask).sum()

        loss_type = str(self.loss_type)
        mode = "train" if model.training else "eval"
        grad_normalizer = (
            float(getattr(self, "current_gradient_accumulation_steps", 1))
            if mode == "train"
            else 1.0
        )

        if loss_type == "grpo":
            loss = (
                seq_loss_sum / seq_token_count.clamp(min=1.0)
            ).mean() / grad_normalizer
        elif loss_type == "bnpo":
            loss = seq_loss_sum.sum() / seq_token_count.sum().clamp(min=1.0)
            loss = loss / grad_normalizer
        elif loss_type == "dr_grpo":
            loss = seq_loss_sum.sum() / (
                batch_size * float(self.max_completion_length)
            )
            loss = loss / grad_normalizer
        elif loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = seq_loss_sum.sum() / normalizer.clamp(min=1.0)
        else:
            raise RuntimeError(f"Unsupported TQ loss_type={loss_type!r}.")

        # Match TRL's core diagnostics using only rewarded model tokens.
        denom = total_token_count.clamp(min=1.0)
        low_clip = low_clip_count / denom
        high_clip = high_clip_count / denom
        region_clip = region_clip_count / denom

        # Entropy is optional under the Unsloth compiled path. Do not fabricate
        # a zero entropy value when the backend did not return entropy.
        if float(entropy_token_count.detach().item()) > 0:
            mean_entropy = entropy_sum / entropy_token_count.clamp(min=1.0)
            self._metrics[mode]["entropy"].append(
                self.accelerator.gather(mean_entropy.detach()).nanmean().item()
            )
        elif not bool(
            getattr(self, "_tq_entropy_unavailable_debug_done", False)
        ):
            print(
                "\nTQ ENTROPY DIAGNOSTIC:"
                " Unsloth returned entropies=None; skipping the entropy "
                "logging metric. This does not affect the GRPO loss or "
                "gradients.",
                flush=True,
            )
            self._tq_entropy_unavailable_debug_done = True
        gathered_low = self.accelerator.gather(low_clip.detach())
        gathered_high = self.accelerator.gather(high_clip.detach())
        gathered_region = self.accelerator.gather(region_clip.detach())
        self._metrics[mode]["clip_ratio/low_mean"].append(
            gathered_low.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/low_min"].append(
            gathered_low.nan_to_num().min().item()
        )
        self._metrics[mode]["clip_ratio/high_mean"].append(
            gathered_high.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/high_max"].append(
            gathered_high.nan_to_num().max().item()
        )
        self._metrics[mode]["clip_ratio/region_mean"].append(
            gathered_region.nanmean().item()
        )

        return loss

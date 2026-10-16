from __future__ import annotations

import copy
import csv
import json
import random
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from transformers import TrainerCallback

from .build_tq_dataset import extract_json_object, format_history
from .tq_environment import (
    make_multiturn_prompt,
    make_tq_environment_factory,
)
from .tq_prompting import tq_chat_template_kwargs, tq_reasoning_enabled


def _extract_single_tool_call(parsed):
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


def _question_has_literal_line_escape(question: str) -> bool:
    """Detect the specific escaped-line corruption observed in direct policies."""
    text = str(question)

    # Ad-hoc guard for outputs such as ``Is the snake a',\\nAnaconda?``.  The
    # tool parser accepts that value as a non-empty string, but the oracle may
    # normalize the escape when emitting its JSON key.  We then cannot match
    # the returned likelihoods to the requested question and the eval aborts.
    #
    # Keep this deliberately narrow for now: genuine newline characters are
    # allowed, while literal backslash-n/backslash-r sequences trigger the
    # existing question-generation retry path.  If more malformed forms show
    # up, consider a general question validator (template-token detection,
    # balanced punctuation/quoting, and control-character checks) and/or an
    # oracle parser fallback when exactly one question was requested.
    return "\\n" in text or "\\r" in text


def _ensure_post_tool_generation_marker(rendered: str, messages) -> str:
    """
    Match the continuation observed in the real GRPO training trace.

    Gemma's normal chat template renders the tool response correctly, but
    after a final role='tool' message it may stop without opening the next
    model turn. Real GRPO training continues with:

        ... <|tool_response|><|turn>model

    If the template has already emitted a generation marker (for example a
    thinking channel), do not add another one.
    """
    if not messages or messages[-1].get("role") != "tool":
        return rendered

    stripped = rendered.rstrip()

    if (
        stripped.endswith("<|turn>model")
        or stripped.endswith("<|channel>thought")
    ):
        return rendered

    return rendered + "<|turn>model\n"


def _question_generation_eos_token_ids(tokenizer):
    """Stop once Gemma hands control to the tool/environment."""
    token_ids = []
    for token_id in (
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|tool_response>"),
        tokenizer.convert_tokens_to_ids("<turn|>"),
    ):
        if (
            token_id is not None
            and token_id != tokenizer.unk_token_id
            and token_id not in token_ids
        ):
            token_ids.append(int(token_id))

    if not token_ids:
        raise RuntimeError("Tokenizer has no usable question-generation stop token.")

    return token_ids[0] if len(token_ids) == 1 else token_ids


def _normalize_entity(text: Optional[str]) -> str:
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text)).strip().casefold()
    value = value.strip('"\'`')
    return " ".join(value.split())


def _parse_belief_guess(text: str) -> Optional[str]:
    """Parse a belief-probe guess robustly from imperfect model output.

    The probe is instructed to emit ``{"guess": "..."}``, but Gemma can append
    chat-template sentinels, code fences, or extra text after an otherwise valid
    JSON object.  In some cases that makes the generic JSON extractor fail and
    the old line-based fallback returned ``"{"`` instead of the actual guess.

    Prefer a normally parsed JSON object, then recover the first explicit
    ``guess`` string field anywhere in the completion.  Only if neither route
    succeeds do we fall back to a short plain-text first line.
    """
    raw = str(text)
    if not raw.strip():
        return None

    # Fast path: preserve the existing structured JSON parser when the
    # completion is clean enough for it.
    try:
        data = extract_json_object(raw)
        if isinstance(data, dict):
            guess = data.get("guess")
            if guess is not None and str(guess).strip():
                return str(guess).strip()
    except Exception:
        pass

    # Robust recovery path.  Match the first explicit JSON-style guess field
    # even if the object is followed by malformed template tokens or prose.
    # Capturing the entire quoted JSON string lets json.loads correctly unescape
    # embedded escapes such as \".
    match = re.search(
        r'["\']guess["\']\s*:\s*("(?:\\.|[^"\\])*")',
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is not None:
        try:
            guess = json.loads(match.group(1))
        except json.JSONDecodeError:
            guess = match.group(1)[1:-1]
        guess = str(guess).strip()
        if guess:
            return guess

    # Be permissive about single-quoted values as a last structured recovery
    # route, even though the requested format is strict JSON.
    match = re.search(
        r'["\']guess["\']\s*:\s*\'((?:\\.|[^\'\\])*)\'',
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is not None:
        guess = match.group(1).strip()
        if guess:
            return guess

    # Final fallback for genuinely plain-text answers.  Remove leading code
    # fences, but never treat JSON punctuation or chat-template sentinels as a
    # semantic guess.
    stripped = raw.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    if not stripped:
        return None

    first_line = stripped.splitlines()[0].strip().strip('"\'`')
    if (
        not first_line
        or first_line in {"{", "}", "[", "]"}
        or first_line.startswith(("<|", "<turn", "<eos", "<pad"))
        or first_line.endswith("|>")
        or first_line.startswith("{")
        or '"guess"' in first_line.casefold()
    ):
        return None
    if len(first_line) <= 120:
        return first_line
    return None


class TQFullGameEvalCallback(TrainerCallback):
    """
    Run 20-turn Twenty Questions games during normal Trainer evaluation.

    The rollout loop is intentionally the same as the verified notebook:

        messages
          -> apply_chat_template(...)
          -> model.generate(...)
          -> tokenizer.parse_response(...)
          -> env.ask_question(...)
          -> append parsed assistant tool call
          -> append tool response
          -> repeat

    No private TRL generation/tool-loop helpers are used.

    Raw model completions are saved for every turn. If the model emits
    rationale/reasoning text before the tool call, that text is preserved
    verbatim in `raw_completion`, even though the patched parser removes it
    from the history passed to subsequent turns.
    """

    def __init__(
        self,
        config,
        trainer,
        eval_words: Optional[Sequence[str]] = None,
        theta_set: Optional[Sequence[str]] = None,
        usage_tracker=None,
    ):
        self.config = config
        self.trainer = trainer
        self.usage_tracker = usage_tracker

        game_config = config.get("game", {})
        eval_config = config.get("eval", {})
        training_config = config.get("training", {})

        self.category = str(
            game_config.get("category")
        )
        self.max_turns = int(
            eval_config.get(
                "full_game_max_turns",
                game_config.get("max_turns", 20),
            )
        )
        self.active_threshold = float(
            game_config.get("active_threshold", 0.01)
        )
        self.information_reward_mode = str(
            game_config.get("information_reward_mode", "realized")
        ).strip().lower()
        self.zero_inactive_hypotheses = bool(
            game_config.get("zero_inactive_hypotheses", False)
        )
        if self.information_reward_mode not in {"realized", "expected"}:
            raise ValueError(
                "game.information_reward_mode must be 'realized' or 'expected'. "
                f"Got: {self.information_reward_mode!r}."
            )

        if theta_set is None:
            raise ValueError(
                "theta_set is required for full-game evaluation."
            )
        self.theta_set = list(theta_set)

        if eval_words is None:
            eval_words = self.theta_set
        self.eval_words = list(eval_words)

        num_eval_words = eval_config.get(
            "full_game_num_eval_words"
        )
        if num_eval_words is not None:
            self.eval_words = self.eval_words[-int(num_eval_words):]

        # Inference batching only; each game has an independent environment.
        self.batch_size = int(
            eval_config.get(
                "full_game_batch_size",
                training_config.get(
                    "per_device_eval_batch_size",
                    4,
                ),
            )
        )

        # Per-turn generation settings. These default to the settings from the
        # verified notebook / training config.
        self.max_new_tokens = int(
            eval_config.get(
                "full_game_max_new_tokens",
                384,
            )
        )
        self.do_sample = bool(
            eval_config.get(
                "full_game_do_sample",
                False,
            )
        )
        self.temperature = float(
            eval_config.get(
                "full_game_temperature",
                training_config.get("temperature", 1.0),
            )
        )
        self.top_p = float(
            eval_config.get("full_game_top_p", 1.0)
        )
        self.top_k = int(
            eval_config.get("full_game_top_k", 0)
        )
        bed_eval_config = config.get("evaluation", {}).get("bed_llm", {})
        self.belief_max_new_tokens = int(
            eval_config.get(
                "full_game_belief_max_new_tokens",
                bed_eval_config.get("guess_max_new_tokens", 64),
            )
        )

        self.reasoning_enabled = tq_reasoning_enabled(config)
        self.chat_template_kwargs = tq_chat_template_kwargs(config)

        self.store_rendered_prompts = bool(
            eval_config.get(
                "full_game_store_rendered_prompts",
                False,
            )
        )
        self.store_structured_context = bool(
            eval_config.get(
                "full_game_store_structured_context",
                False,
            )
        )

        self.every_n_steps = max(
            1,
            int(
                eval_config.get(
                    "full_game_every_n_steps",
                )
            ),
        )
        self._eval_call_count = 0

        # Use a deterministic eval seed but restore all training RNG state
        # afterward so sampled full-game evaluation does not perturb GRPO.
        self.eval_seed = int(
            eval_config.get(
                "full_game_seed",
                game_config.get("seed", 1),
            )
        )

        self.text_tokenizer = trainer.processing_class
        self.text_tokenizer.padding_side = "left"
        self.text_tokenizer.truncation_side = "left"

        if self.text_tokenizer.pad_token_id is None:
            self.text_tokenizer.pad_token = (
                self.text_tokenizer.eos_token
            )

        self.environment_factory = (
            make_tq_environment_factory(
                config,
                usage_tracker=usage_tracker,
            )
        )

        output_override = eval_config.get(
            "full_game_output_dir"
        )
        if output_override:
            self.output_dir = Path(output_override)
        else:
            self.output_dir = (
                Path(trainer.args.output_dir)
                / "full_game_eval"
            )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _synchronize_generation_device() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _check_prompt_lengths(
        self,
        model,
        inputs,
        *,
        max_new_tokens: int,
        prompt_kind: str,
    ) -> None:
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            prompt_lengths = [int(inputs["input_ids"].shape[-1])]
        else:
            prompt_lengths = [int(value) for value in attention_mask.sum(dim=1)]

        max_prompt_tokens = max(prompt_lengths, default=0)
        model_config = getattr(model, "config", None)
        context_limit = getattr(model_config, "max_position_embeddings", None)
        if context_limit is None:
            text_config = getattr(model_config, "text_config", None)
            context_limit = getattr(text_config, "max_position_embeddings", None)

        if context_limit is None:
            return

        required_tokens = max_prompt_tokens + int(max_new_tokens)
        if required_tokens > int(context_limit):
            raise RuntimeError(
                f"{prompt_kind} generation requires up to {required_tokens} tokens "
                f"({max_prompt_tokens} prompt + {max_new_tokens} generated), "
                f"exceeding the model context limit of {context_limit}. "
                "No prompt was truncated; reduce the history or generation budget."
            )

    def _belief_probe_messages(self, history):
        return [
            {
                "role": "system",
                "content": (
                    "Based on all questions and answers so far, what is your "
                    f"single best guess for the hidden {self.category}? Return JSON only."
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

    def _run_belief_probes(self, model, games, turn: int) -> None:
        if not games:
            return

        device = next(model.parameters()).device
        template_kwargs = dict(self.chat_template_kwargs)
        template_kwargs["enable_thinking"] = False

        for start in range(0, len(games), self.batch_size):
            chunk_games = games[start : start + self.batch_size]
            rendered_prompts = [
                self.text_tokenizer.apply_chat_template(
                    self._belief_probe_messages(game["env"].history),
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
                for game in chunk_games
            ]
            inputs = self.text_tokenizer(
                rendered_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            self._check_prompt_lengths(
                model,
                inputs,
                max_new_tokens=self.belief_max_new_tokens,
                prompt_kind="Belief-probe",
            )
            inputs = inputs.to(device)

            self._synchronize_generation_device()
            inference_start = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.belief_max_new_tokens,
                do_sample=False,
                pad_token_id=self.text_tokenizer.pad_token_id,
                eos_token_id=self.text_tokenizer.eos_token_id,
            )
            self._synchronize_generation_device()
            elapsed = time.perf_counter() - inference_start

            prompt_width = inputs["input_ids"].shape[-1]
            raw_completions = self.text_tokenizer.batch_decode(
                outputs[:, prompt_width:],
                skip_special_tokens=False,
            )
            per_game_seconds = elapsed / len(chunk_games)

            for game, raw_completion in zip(
                chunk_games,
                raw_completions,
                strict=True,
            ):
                guess = _parse_belief_guess(raw_completion)
                probe = {
                    "turn": int(turn),
                    "guess": guess,
                    "correct": (
                        _normalize_entity(guess)
                        == _normalize_entity(game["true_word"])
                    ),
                    "raw_completion": raw_completion,
                    "model_inference_seconds": float(per_game_seconds),
                }
                game["belief_probes"].append(probe)
                if turn > 0 and game["steps"]:
                    game["steps"][-1]["belief_probe"] = probe

    def on_step_end(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        if not getattr(
            state,
            "is_world_process_zero",
            True,
        ):
            return control

        if state.global_step == 0:
            return control

        if state.global_step % self.every_n_steps != 0:
            return control

        self._eval_call_count += 1

        metrics = self.evaluate(
            global_step=int(state.global_step)
        )

        log_metrics = {
            f"eval_full_game/{key}": value
            for key, value in metrics.items()
        }

        print(
            "\n=== FULL-GAME TWENTY QUESTIONS EVAL ===\n"
            + json.dumps(
                log_metrics,
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )

        self.trainer.log(log_metrics)

        return control

    @torch.no_grad()
    def evaluate(self, global_step: int) -> Dict[str, float]:
        model = self.trainer.model

        # ------------------------------------------------------------
        # Preserve training state.
        # ------------------------------------------------------------
        was_training = model.training

        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.random.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        )

        # Use the SAME evaluation random stream at every checkpoint so
        # changes in metrics reflect model changes rather than a different
        # sampled evaluation trajectory.
        eval_seed = self.eval_seed
        random.seed(eval_seed)
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)

        model.eval()

        results = []

        try:
            for start in range(
                0,
                len(self.eval_words),
                self.batch_size,
            ):
                words = self.eval_words[
                    start:
                    start + self.batch_size
                ]

                print(
                    f"\nFull-game eval @ step {global_step}: "
                    f"{start + 1}-"
                    f"{start + len(words)}"
                    f"/{len(self.eval_words)}",
                    flush=True,
                )

                games = [
                    self._new_game(true_word)
                    for true_word in words
                ]
                self._run_belief_probes(model, games, turn=0)

                for turn in range(
                    1,
                    self.max_turns + 1,
                ):
                    active_games = [
                        game
                        for game in games
                        if not game["finished"]
                    ]

                    if not active_games:
                        break

                    rendered_prompts = []
                    for game in active_games:
                        rendered = self.text_tokenizer.apply_chat_template(
                            game["messages"],
                            tools=[game["env"].ask_question],
                            tokenize=False,
                            add_generation_prompt=True,
                            **self.chat_template_kwargs,
                        )
                        rendered_prompts.append(
                            _ensure_post_tool_generation_marker(
                                rendered,
                                game["messages"],
                            )
                        )

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
                    inputs = inputs.to(next(model.parameters()).device)

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

                    self._synchronize_generation_device()
                    inference_start = time.perf_counter()
                    outputs = model.generate(**inputs, **generation_kwargs)
                    self._synchronize_generation_device()
                    per_game_generation_seconds = (
                        time.perf_counter() - inference_start
                    ) / len(active_games)

                    prompt_width = inputs["input_ids"].shape[-1]
                    generated_batch = outputs[:, prompt_width:]

                    for game, rendered, generated_ids in zip(
                        active_games,
                        rendered_prompts,
                        generated_batch,
                        strict=True,
                    ):
                        raw_completion = self.text_tokenizer.decode(
                            generated_ids,
                            skip_special_tokens=False,
                        )
                        parsed = self.text_tokenizer.parse_response(raw_completion)
                        tool_name, tool_args = _extract_single_tool_call(parsed)
                        step_record: Dict[str, Any] = {
                            "turn": turn,
                            "raw_completion": raw_completion,
                            "parsed": copy.deepcopy(parsed),
                            "model_inference_seconds": float(
                                per_game_generation_seconds
                            ),
                        }
                        if self.store_rendered_prompts:
                            step_record["rendered_prompt"] = rendered
                        if self.store_structured_context:
                            step_record["structured_context_before"] = copy.deepcopy(
                                game["messages"]
                            )

                        question = (
                            str(tool_args.get("question", "")).strip()
                            if isinstance(tool_args, dict)
                            else ""
                        )
                        if (
                            tool_name is None
                            or not hasattr(game["env"], tool_name)
                            or _question_has_literal_line_escape(question)
                        ):
                            step_record.update(
                                {
                                    "question": None,
                                    "answer": None,
                                    "tool_response": None,
                                    "valid_tool_call": False,
                                }
                            )

                            game["steps"].append(
                                step_record
                            )
                            game["finished"] = True
                            game["stop_reason"] = (
                                "invalid_tool_call"
                            )
                            continue

                        tool_result = getattr(
                            game["env"],
                            tool_name,
                        )(**tool_args)

                        debug = dict(
                            game[
                                "env"
                            ].step_debug[-1]
                        )

                        step_record.update(
                            {
                                "question":
                                    debug["question"],
                                "answer":
                                    debug["answer"],
                                "tool_response":
                                    str(tool_result),
                                "reward":
                                    float(debug["reward"]),
                                "reward_mode":
                                    self.information_reward_mode,
                                "active_count":
                                    int(
                                        debug[
                                            "active_count"
                                        ]
                                    ),
                                "sum_C":
                                    float(debug["sum_C"]),
                                "solved":
                                    bool(debug.get("solved", False)),
                                "valid_tool_call":
                                    True,
                            }
                        )

                        game["steps"].append(
                            step_record
                        )

                        # EXACT history augmentation from the verified notebook.
                        game["messages"].append(
                            copy.deepcopy(parsed)
                        )
                        game["messages"].append(
                            {
                                "role": "tool",
                                "name": tool_name,
                                "content": str(
                                    tool_result
                                ),
                            }
                        )

                        # Stop the full game as soon as the oracle says the
                        # true target has been identified. The final successful
                        # question and answer remain stored in the trajectory.
                        if debug.get("solved", False):
                            game["finished"] = True
                            game["stop_reason"] = "solved"

                    probed_games = [
                        game
                        for game in active_games
                        if game["steps"]
                        and game["steps"][-1].get("turn") == turn
                        and game["steps"][-1].get("valid_tool_call", False)
                    ]
                    self._run_belief_probes(model, probed_games, turn=turn)

                results.extend(
                    self._finish_game(game)
                    for game in games
                )

        finally:
            if was_training:
                model.train()

            # Restore every RNG stream so evaluation does not alter subsequent
            # sampled GRPO generations.
            random.setstate(
                python_rng_state
            )
            np.random.set_state(
                numpy_rng_state
            )
            torch.random.set_rng_state(
                torch_rng_state
            )
            if (
                cuda_rng_state is not None
                and torch.cuda.is_available()
            ):
                torch.cuda.set_rng_state_all(
                    cuda_rng_state
                )

        self._save(
            results=results,
            global_step=global_step,
            eval_seed=eval_seed,
        )

        return self._summarize(results)

    def _new_game(self, true_word):
        env = self.environment_factory()

        # The training factory captures rollout_horizon (e.g. K=2).
        # Full-game evaluation explicitly overrides the environment instance.
        env.horizon = int(
            self.max_turns
        )

        env.reset(
            true_word=true_word,
            history_json="[]",
            contrastive_set_json=json.dumps(
                self.theta_set
            ),
            C_json=json.dumps(
                [1.0] * len(self.theta_set)
            ),
        )

        if env.horizon != self.max_turns:
            raise RuntimeError(
                "Failed to set full-game environment horizon: "
                f"env.horizon={env.horizon}, "
                f"expected={self.max_turns}."
            )

        messages = make_multiturn_prompt(
            history=[],
            category=self.category,
            turn=1,
            max_turns=self.max_turns,
            config=self.config,
        )

        # messages[0]["content"] += (
        #     "\nDo NOT ask whether the target is a specific entity. "
        #     "Ask only attribute-based questions that can reduce the hypothesis space."
        # )

        return {
            "true_word": str(true_word),
            "env": env,
            "messages": copy.deepcopy(
                messages
            ),
            "steps": [],
            "belief_probes": [],
            "finished": False,
            "stop_reason": "max_turns",
        }

    def _finish_game(self, game):
        env = game["env"]

        valid_steps = [
            step
            for step in game["steps"]
            if step.get(
                "valid_tool_call",
                False,
            )
        ]

        solved = game["stop_reason"] == "solved"

        return {
            "true_word": game["true_word"],
            "theta_set_size": len(self.theta_set),
            "max_turns": self.max_turns,
            "stop_reason": game["stop_reason"],
            "solved": solved,
            "num_turns": len(valid_steps),
            # Solving early is a valid trajectory; only malformed tool use is
            # considered invalid here.
            "valid_trajectory": (
                game["stop_reason"] != "invalid_tool_call"
            ),
            # Keep the full-game reward metric undiscounted so evaluation
            # remains comparable across training runs with different gamma.
            "trajectory_reward": float(
                env._get_blau_reward(gamma_override=1.0)
            ),
            "trajectory_reward_mode": self.information_reward_mode,
            "zero_inactive_hypotheses": self.zero_inactive_hypotheses,
            "final_active_hypotheses": int(
                np.sum(
                    env.C
                    > self.active_threshold
                )
            ),
            "belief_probes": game["belief_probes"],
            "steps": game["steps"],
        }

    def _save(
        self,
        results,
        global_step: int,
        eval_seed: int,
    ):
        eval_dir = (
            self.output_dir
            / (
                f"step_{global_step:06d}"
                f"_eval_{self._eval_call_count:03d}"
            )
        )
        eval_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        trajectories_path = (
            eval_dir
            / "trajectories.json"
        )

        with trajectories_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "global_step":
                        global_step,
                    "eval_seed":
                        eval_seed,
                    "category":
                        self.category,
                    "max_turns":
                        self.max_turns,
                    "active_threshold":
                        self.active_threshold,
                    "information_reward_mode":
                        self.information_reward_mode,
                    "enable_reasoning":
                        self.reasoning_enabled,
                    "zero_inactive_hypotheses":
                        self.zero_inactive_hypotheses,
                    "num_games":
                        len(results),
                    "theta_set":
                        self.theta_set,
                    "eval_words":
                        self.eval_words,
                    "batched_full_game": hasattr(self, "batch_client"),
                    "generation_settings": {
                        "batch_size": self.batch_size,
                        "max_turns": self.max_turns,
                        "max_new_tokens": self.max_new_tokens,
                        "question_retries": getattr(self, "question_retries", 1),
                        "belief_max_new_tokens": self.belief_max_new_tokens,
                        "do_sample": self.do_sample,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": self.top_k,
                        "enable_reasoning": self.reasoning_enabled,
                        "information_reward_mode": self.information_reward_mode,
                        "active_threshold": self.active_threshold,
                        "zero_inactive_hypotheses": self.zero_inactive_hypotheses,
                    },
                    "api_usage":
                        (
                            self.usage_tracker.snapshot()
                            if self.usage_tracker is not None
                            else None
                        ),
                    "games":
                        results,
                },
                handle,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        summary = self._summarize(results)
        summary_path = eval_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)

        belief_csv_path = eval_dir / "belief_accuracy_by_turn.csv"
        with belief_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["turn", "accuracy", "num_games"])
            writer.writeheader()
            for turn, row in summary.get("belief_accuracy_by_turn", {}).items():
                writer.writerow(
                    {
                        "turn": int(turn),
                        "accuracy": row["accuracy"],
                        "num_games": row["num_games"],
                    }
                )

        # Flat JSONL is convenient for inspecting/searching raw completions.
        completions_path = (
            eval_dir
            / "completions.jsonl"
        )

        with completions_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for game in results:
                for step in game["steps"]:
                    row = {
                        "global_step":
                            global_step,
                        "true_word":
                            game["true_word"],
                        "turn":
                            step["turn"],
                        "question":
                            step.get("question"),
                        "answer":
                            step.get("answer"),
                        "solved":
                            step.get("solved", False),
                        "reward":
                            step.get("reward"),
                        "active_count":
                            step.get(
                                "active_count"
                            ),
                        "raw_completion":
                            step.get(
                                "raw_completion"
                            ),
                        "parsed":
                            step.get("parsed"),
                        "model_inference_seconds":
                            step.get("model_inference_seconds"),
                        "belief_probe":
                            step.get("belief_probe"),
                    }

                    handle.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\n"
                    )

        print(
            "\nSaved full-game evaluation:"
            f"\n  {trajectories_path}"
            f"\n  {completions_path}"
            f"\n  {summary_path}"
            f"\n  {belief_csv_path}",
            flush=True,
        )

    @staticmethod
    def _trajectory_metrics(results):
        if not results:
            return {
                "num_games": 0.0,
                "solve_rate": 0.0,
                "avg_turns_solved": 0.0,
                "median_turns_solved": 0.0,
                "avg_num_turns": 0.0,
                "avg_final_active_hypotheses": 0.0,
                "avg_trajectory_reward": 0.0,
                "normalized_active_auc": 0.0,
                "repeated_question_rate": 0.0,
                "final_belief_accuracy": 0.0,
                "belief_accuracy_by_turn": {},
                "stop_reason_counts": {},
            }

        solved_results = [
            result
            for result in results
            if result.get("solved", False)
        ]
        solved_turns = [result["num_turns"] for result in solved_results]

        probes_by_turn = {}
        for result in results:
            for probe in result.get("belief_probes", []):
                probes_by_turn.setdefault(int(probe["turn"]), []).append(
                    bool(probe["correct"])
                )
        belief_accuracy_by_turn = {
            str(turn): {
                "accuracy": float(np.mean(values)),
                "num_games": len(values),
            }
            for turn, values in sorted(probes_by_turn.items())
        }
        final_belief_correct = [
            bool(result["belief_probes"][-1]["correct"])
            for result in results
            if result.get("belief_probes")
        ]

        repeated_questions = 0
        total_questions = 0
        auc_values = []
        for result in results:
            steps = result.get("steps", [])
            theta_size = max(
                1,
                int(
                    result.get(
                        "theta_set_size",
                        max(
                            [
                                result.get("final_active_hypotheses", 1),
                                *[
                                    step.get("active_count", 0) or 0
                                    for step in steps
                                ],
                            ]
                        ),
                    )
                ),
            )
            max_turns = int(
                result.get(
                    "max_turns",
                    max([step.get("turn", 0) for step in steps], default=0),
                )
            )
            seen = set()
            last_active = float(theta_size)
            active_trajectory = [last_active]
            steps_by_turn = {
                int(step["turn"]): step
                for step in steps
                if step.get("turn") is not None
            }
            solved_turn = result["num_turns"] if result.get("solved", False) else None

            for turn in range(1, max_turns + 1):
                step = steps_by_turn.get(turn)
                if step is not None and step.get("question"):
                    question_key = " ".join(step["question"].casefold().split())
                    total_questions += 1
                    if question_key in seen:
                        repeated_questions += 1
                    seen.add(question_key)
                    if step.get("active_count") is not None:
                        last_active = float(step["active_count"])

                value = (
                    0.0
                    if solved_turn is not None and turn > solved_turn
                    else last_active
                )
                active_trajectory.append(value)

            auc_values.append(float(np.mean(active_trajectory) / theta_size))

        return {
            "num_games": float(len(results)),
            "solve_rate": float(len(solved_results) / len(results)),
            "avg_turns_solved": (
                float(np.mean(solved_turns)) if solved_turns else 0.0
            ),
            "median_turns_solved": (
                float(np.median(solved_turns)) if solved_turns else 0.0
            ),
            "avg_num_turns": float(
                np.mean([result["num_turns"] for result in results])
            ),
            "avg_final_active_hypotheses": float(
                np.mean(
                    [result["final_active_hypotheses"] for result in results]
                )
            ),
            "avg_trajectory_reward": float(
                np.mean([result["trajectory_reward"] for result in results])
            ),
            "normalized_active_auc": float(np.mean(auc_values)),
            "repeated_question_rate": (
                float(repeated_questions / total_questions)
                if total_questions
                else 0.0
            ),
            "final_belief_accuracy": (
                float(np.mean(final_belief_correct))
                if final_belief_correct
                else 0.0
            ),
            "belief_accuracy_by_turn": belief_accuracy_by_turn,
            "stop_reason_counts": dict(
                Counter(result["stop_reason"] for result in results)
            ),
        }

    @staticmethod
    def _inference_timing(results):
        model_inference_seconds = sum(
            float(step.get("model_inference_seconds", 0.0) or 0.0)
            for result in results
            for step in result.get("steps", [])
        )
        belief_probe_seconds = sum(
            float(probe.get("model_inference_seconds", 0.0) or 0.0)
            for result in results
            for probe in result.get("belief_probes", [])
        )
        denominator = max(len(results), 1)
        return {
            "model_inference_seconds": float(model_inference_seconds),
            "model_inference_seconds_per_game": float(
                model_inference_seconds / denominator
            ),
            "belief_probe_seconds": float(belief_probe_seconds),
            "belief_probe_seconds_per_game": float(
                belief_probe_seconds / denominator
            ),
        }

    @staticmethod
    def _summarize(results):
        if not results:
            return {
                **TQFullGameEvalCallback._trajectory_metrics([]),
                "num_valid_games": 0,
                "num_invalid_games": 0,
                "valid_trajectory_rate": 0.0,
                "invalid_trajectory_rate": 0.0,
                "invalid_tool_call_rate": 0.0,
                "valid_solve_rate": 0.0,
                "valid_avg_turns_solved": 0.0,
                "valid_median_turns_solved": 0.0,
                "valid_avg_num_turns": 0.0,
                "valid_avg_final_active_hypotheses": 0.0,
                "valid_avg_trajectory_reward": 0.0,
                "valid_normalized_active_auc": 0.0,
                "valid_repeated_question_rate": 0.0,
                "valid_final_belief_accuracy": 0.0,
                "valid_belief_accuracy_by_turn": {},
                "valid_stop_reason_counts": {},
                "invalid_stop_reason_counts": {},
                "model_inference_seconds": 0.0,
                "model_inference_seconds_per_game": 0.0,
                "belief_probe_seconds": 0.0,
                "belief_probe_seconds_per_game": 0.0,
                "valid_model_inference_seconds": 0.0,
                "valid_model_inference_seconds_per_game": 0.0,
                "valid_belief_probe_seconds": 0.0,
                "valid_belief_probe_seconds_per_game": 0.0,
                "inference_timing": TQFullGameEvalCallback._inference_timing([]),
                "valid_inference_timing": TQFullGameEvalCallback._inference_timing([]),
            }

        valid_results = [
            result
            for result in results
            if bool(result.get("valid_trajectory", False))
        ]
        invalid_results = [
            result
            for result in results
            if not bool(result.get("valid_trajectory", False))
        ]
        overall = TQFullGameEvalCallback._trajectory_metrics(results)
        valid = TQFullGameEvalCallback._trajectory_metrics(valid_results)
        timing = TQFullGameEvalCallback._inference_timing(results)
        valid_timing = TQFullGameEvalCallback._inference_timing(valid_results)

        return {
            **overall,
            "num_valid_games": len(valid_results),
            "num_invalid_games": len(invalid_results),
            "valid_trajectory_rate": float(len(valid_results) / len(results)),
            "invalid_trajectory_rate": float(len(invalid_results) / len(results)),
            "invalid_tool_call_rate": float(
                np.mean(
                    [
                        result["stop_reason"] == "invalid_tool_call"
                        for result in results
                    ]
                )
            ),
            "valid_solve_rate": valid["solve_rate"],
            "valid_avg_turns_solved": valid["avg_turns_solved"],
            "valid_median_turns_solved": valid["median_turns_solved"],
            "valid_avg_num_turns": valid["avg_num_turns"],
            "valid_avg_final_active_hypotheses": valid[
                "avg_final_active_hypotheses"
            ],
            "valid_avg_trajectory_reward": valid["avg_trajectory_reward"],
            "valid_normalized_active_auc": valid["normalized_active_auc"],
            "valid_repeated_question_rate": valid["repeated_question_rate"],
            "valid_final_belief_accuracy": valid["final_belief_accuracy"],
            "valid_belief_accuracy_by_turn": valid["belief_accuracy_by_turn"],
            "valid_stop_reason_counts": valid["stop_reason_counts"],
            "invalid_stop_reason_counts": dict(
                Counter(result["stop_reason"] for result in invalid_results)
            ),
            **timing,
            "valid_model_inference_seconds": valid_timing[
                "model_inference_seconds"
            ],
            "valid_model_inference_seconds_per_game": valid_timing[
                "model_inference_seconds_per_game"
            ],
            "valid_belief_probe_seconds": valid_timing[
                "belief_probe_seconds"
            ],
            "valid_belief_probe_seconds_per_game": valid_timing[
                "belief_probe_seconds_per_game"
            ],
            "inference_timing": timing,
            "valid_inference_timing": valid_timing,
        }

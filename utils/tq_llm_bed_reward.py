from __future__ import annotations

import contextlib
import copy
import json
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .build_tq_dataset import (
    ANSWER_LABELS,
    extract_beliefs_prompt,
    extract_json_object,
    normalize_lookup_key,
    tq_feedback,
)


LLM_BED_REWARD_TYPES = {"llm_bed", "asig_eig"}
LLM_BED_LIKELIHOOD_SOURCE_ALIASES = {
    "policy": "policy",
    "current_policy": "policy",
    "model": "policy",
    "training_model": "policy",
    "external": "external",
    "external_llm": "external",
    "gpt": "external",
    "gpt5": "external",
    "gpt_5": "external",
    "cache": "external",
    "cached_external": "external",
}


def normalize_likelihood_source(value: Any) -> str:
    source = str(value).strip().lower().replace("-", "_")
    try:
        return LLM_BED_LIKELIHOOD_SOURCE_ALIASES[source]
    except KeyError as exc:
        raise ValueError(
            "game.llm_bed.likelihood_source must be 'policy' or 'external'. "
            f"Got: {value!r}."
        ) from exc


def _blank_path(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip() in {"", "???", "null", "None"}


@dataclass
class LLMBEDRewardConfig:
    num_hypotheses: int = 15
    hypothesis_sampling_temperature: float = 1.3
    consistency_threshold: float = 0.2
    refill_rounds: int = 3
    likelihood_epsilon: float = 1e-12
    belief_max_new_tokens: int = 256
    likelihood_batch_size: int = 64
    max_seq_length: int = 32768
    likelihood_source: str = "policy"
    external_likelihood_cache_path: str | None = None
    debug: bool = False
    answer_prompt_enable_thinking: bool = False
    hypothesis_prompt_enable_thinking: bool = False


def normalize_reward_type(value: Any) -> str:
    reward_type = str(value).strip().lower().replace("-", "_")
    return "llm_bed" if reward_type in LLM_BED_REWARD_TYPES else reward_type


def is_llm_bed_reward_type(value: Any) -> bool:
    return normalize_reward_type(value) == "llm_bed"


def llm_bed_config_from_game(game_config: dict[str, Any]) -> LLMBEDRewardConfig:
    raw = dict(game_config.get("llm_bed") or {})

    def _get(name: str, default: Any) -> Any:
        return raw.get(name, game_config.get(name, default))

    config = LLMBEDRewardConfig(
        num_hypotheses=int(_get("num_hypotheses", 15)),
        hypothesis_sampling_temperature=float(
            _get("hypothesis_sampling_temperature", raw.get("temperature", 1.3))
        ),
        consistency_threshold=float(_get("consistency_threshold", 0.2)),
        refill_rounds=int(_get("refill_rounds", 3)),
        likelihood_epsilon=float(_get("likelihood_epsilon", 1e-12)),
        belief_max_new_tokens=int(_get("belief_max_new_tokens", 256)),
        likelihood_batch_size=int(_get("likelihood_batch_size", 64)),
        max_seq_length=int(_get("max_seq_length", 32768)),
        likelihood_source=normalize_likelihood_source(
            _get("likelihood_source", "policy")
        ),
        external_likelihood_cache_path=(
            None
            if _blank_path(_get("external_likelihood_cache_path", None))
            else str(_get("external_likelihood_cache_path", None))
        ),
        debug=bool(_get("debug", False)),
        answer_prompt_enable_thinking=bool(
            _get("answer_prompt_enable_thinking", False)
        ),
        hypothesis_prompt_enable_thinking=bool(
            _get("hypothesis_prompt_enable_thinking", False)
        ),
    )

    if config.num_hypotheses < 1:
        raise ValueError("game.llm_bed.num_hypotheses must be >= 1.")
    if config.hypothesis_sampling_temperature <= 0:
        raise ValueError(
            "game.llm_bed.hypothesis_sampling_temperature must be > 0."
        )
    if config.consistency_threshold < 0:
        raise ValueError("game.llm_bed.consistency_threshold must be >= 0.")
    if config.refill_rounds < 1:
        raise ValueError("game.llm_bed.refill_rounds must be >= 1.")
    if config.likelihood_epsilon <= 0:
        raise ValueError("game.llm_bed.likelihood_epsilon must be > 0.")
    if config.belief_max_new_tokens < 1:
        raise ValueError("game.llm_bed.belief_max_new_tokens must be >= 1.")
    if config.likelihood_batch_size < 1:
        raise ValueError("game.llm_bed.likelihood_batch_size must be >= 1.")
    if config.max_seq_length < 1:
        raise ValueError("game.llm_bed.max_seq_length must be >= 1.")
    if config.likelihood_source == "external":
        if not config.external_likelihood_cache_path:
            raise ValueError(
                "game.llm_bed.external_likelihood_cache_path is required when "
                "game.llm_bed.likelihood_source='external'."
            )
    elif config.external_likelihood_cache_path:
        raise ValueError(
            "game.llm_bed.external_likelihood_cache_path is only valid when "
            "game.llm_bed.likelihood_source='external'."
        )

    return config


def reject_cache_config_for_llm_bed(game_config: dict[str, Any]) -> None:
    """Keep the LLM-BED cache path explicit and separate from Blau config."""
    if not is_llm_bed_reward_type(game_config.get("reward_type")):
        return

    value = game_config.get("likelihood_cache_path")
    if value is None:
        return
    text = str(value).strip()
    if text in {"", "???", "null", "None"}:
        return
    raise ValueError(
        "reward_type='llm_bed' must not configure the top-level "
        "game.likelihood_cache_path. For external LLM likelihoods, set "
        "game.llm_bed.likelihood_source='external' and "
        "game.llm_bed.external_likelihood_cache_path instead."
    )


def append_bed_hypotheses_to_prompt(
    messages: Sequence[dict[str, Any]],
    candidates: Sequence[str],
) -> list[dict[str, Any]]:
    """Return a prompt where the question generator can see BED hypotheses."""
    output = copy.deepcopy(list(messages))
    candidates = [str(candidate) for candidate in candidates if str(candidate).strip()]
    if not candidates:
        return output
    if len(output) < 2 or output[1].get("role") != "user":
        raise ValueError("Expected TQ prompt with system then user message.")

    candidate_text = json.dumps(candidates, ensure_ascii=False)
    output[1]["content"] = str(output[1].get("content", "")) + (
        "\n\nCurrent BED candidate hypotheses sampled from your belief:\n"
        f"{candidate_text}\n\n"
        "Use these candidates to propose a question that is useful for "
        "distinguishing among plausible targets. The true target is not "
        "guaranteed to be in this finite candidate set."
    )
    return output


def parse_candidate_list(text: str, existing: Sequence[str]) -> list[str]:
    try:
        data = extract_json_object(text)
    except Exception:
        return []

    if not isinstance(data, dict):
        return []

    raw = data.get("candidates", [])
    if not isinstance(raw, list):
        return []

    seen = {
        normalize_lookup_key(candidate)
        for candidate in existing
        if str(candidate).strip()
    }
    output = []
    for candidate in raw:
        candidate = str(candidate).strip()
        key = normalize_lookup_key(candidate)
        if key and key not in seen:
            output.append(candidate)
            seen.add(key)
    return output


def compute_uniform_eig(
    answer_probabilities: Sequence[Sequence[float]],
    *,
    eps: float = 1e-12,
) -> tuple[float, list[float]]:
    matrix = np.asarray(answer_probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(ANSWER_LABELS):
        raise ValueError(
            "answer_probabilities must have shape "
            f"(N, {len(ANSWER_LABELS)})."
        )
    if matrix.shape[0] < 2:
        predictive = (
            matrix[0].tolist()
            if matrix.shape[0] == 1
            else [1.0 / len(ANSWER_LABELS)] * len(ANSWER_LABELS)
        )
        return 0.0, [float(value) for value in predictive]
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("answer_probabilities must be finite and non-negative.")

    row_sums = matrix.sum(axis=1, keepdims=True)
    valid = row_sums[:, 0] > 0
    matrix = matrix[valid]
    if matrix.shape[0] < 2:
        predictive = (
            matrix[0].tolist()
            if matrix.shape[0] == 1
            else [1.0 / len(ANSWER_LABELS)] * len(ANSWER_LABELS)
        )
        return 0.0, [float(value) for value in predictive]

    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    matrix = np.clip(matrix, float(eps), 1.0)
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    predictive = matrix.mean(axis=0)
    predictive = np.clip(predictive, float(eps), 1.0)
    predictive = predictive / predictive.sum()

    eig = float(
        np.mean(
            np.sum(
                matrix * (np.log(matrix) - np.log(predictive)[None, :]),
                axis=1,
            )
        )
    )
    if -1e-10 < eig < 0.0:
        eig = 0.0
    return float(eig), [float(value) for value in predictive]


def _normalise_external_answer_distribution(
    distribution: Any,
    *,
    eps: float,
) -> np.ndarray | None:
    if not isinstance(distribution, dict):
        return None

    try:
        values = np.asarray(
            [float(distribution[label]) for label in ANSWER_LABELS],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None

    if not np.all(np.isfinite(values)) or np.any(values < 0):
        return None

    total = float(values.sum())
    if total <= 0:
        return None

    values = np.clip(values / total, float(eps), 1.0)
    return values / values.sum()


class PolicyLLMBEDReward:
    """LLM-BED EIG for K=1 Twenty Questions GRPO."""

    def __init__(
        self,
        *,
        model,
        tokenizer,
        config: LLMBEDRewardConfig,
        category: str,
        max_turns: int,
        chat_template_kwargs: dict[str, Any] | None = None,
        usage_tracker=None,
    ) -> None:
        self.model = model
        self.tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        self.config = config
        self.category = str(category)
        self.max_turns = int(max_turns)
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.usage_tracker = usage_tracker
        self.timing: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self._last_answer_scoring_debug: list[list[dict[str, Any]]] = []

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def _add_time(self, key: str, elapsed: float) -> None:
        self.timing[key] = float(self.timing.get(key, 0.0) + elapsed)

    def _add_count(self, key: str, value: int = 1) -> None:
        self.counts[key] = int(self.counts.get(key, 0) + int(value))

    @contextlib.contextmanager
    def _evaluation_mode(self):
        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()
        try:
            yield
        finally:
            if was_training:
                self.model.train()

    def _device(self):
        return next(self.model.parameters()).device

    def _model_forward(self, *, input_ids, attention_mask, use_cache=False, logits_to_keep=None):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }
        if logits_to_keep is None:
            return self.model(**kwargs)
        try:
            return self.model(**kwargs, logits_to_keep=logits_to_keep)
        except TypeError:
            try:
                return self.model(**kwargs, num_logits_to_keep=logits_to_keep)
            except TypeError:
                return self.model(**kwargs)

    def _model_context_limit(self) -> int | None:
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
        if prompt_tokens > int(self.config.max_seq_length):
            raise RuntimeError(
                f"{operation} prompt has {prompt_tokens} tokens, exceeding "
                f"game.llm_bed.max_seq_length={self.config.max_seq_length}."
            )
        context_limit = self._model_context_limit()
        required = int(prompt_tokens) + int(extra_tokens)
        if context_limit is not None and required > context_limit:
            raise RuntimeError(
                f"{operation} requires {required} tokens, exceeding model "
                f"context limit {context_limit}."
            )

    def _render_messages(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        enable_thinking: bool | None,
    ) -> str:
        template_kwargs = dict(self.chat_template_kwargs)
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = bool(enable_thinking)
        return self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )

    @torch.no_grad()
    def _generate_messages(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        do_sample: bool,
        temperature: float,
        max_new_tokens: int,
        enable_thinking: bool | None,
    ) -> list[dict[str, Any]]:
        rendered = self._render_messages(
            messages,
            enable_thinking=enable_thinking,
        )
        prompt_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        self._check_sequence_length(
            prompt_tokens=len(prompt_ids),
            extra_tokens=int(max_new_tokens),
            operation="LLM-BED generation",
        )

        old_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer(
                [rendered],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(self._device())
            kwargs = {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": bool(do_sample),
                "num_return_sequences": 1,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if do_sample:
                kwargs["temperature"] = float(temperature)

            with self._evaluation_mode():
                start = time.perf_counter()
                outputs = self.model.generate(**inputs, **kwargs)
                self._add_time("hypothesis_generation_seconds", time.perf_counter() - start)

            prompt_width = inputs["input_ids"].shape[-1]
            generated = outputs[:, prompt_width:]
            decoded = self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=False,
            )
            rows = []
            for text in decoded:
                row = {
                    "text": text,
                    "generated_tokens": int(generated.shape[-1]),
                }
                if self.config.debug:
                    row["prompt"] = rendered
                rows.append(row)
            return rows
        finally:
            self.tokenizer.padding_side = old_padding

    def _answer_prompt(
        self,
        *,
        history: Sequence[dict[str, str]],
        question: str,
        candidate: str,
    ) -> list[dict[str, str]]:
        del history
        return [
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
                "content": f"Ground Truth: {candidate}\nQuestion: {question}",
            },
        ]

    def _continuation_ids(self, prompt_text: str, prompt_ids: list[int], label: str):
        full_ids = self.tokenizer.encode(
            prompt_text + label,
            add_special_tokens=False,
        )
        if full_ids[: len(prompt_ids)] == prompt_ids:
            return full_ids[len(prompt_ids) :], True
        return self.tokenizer.encode(label, add_special_tokens=False), False

    def _prepare_scoring_pairs(
        self,
        items: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        pairs = []
        labels = [str(label).strip().lower() for label in ANSWER_LABELS]
        for item_index, item in enumerate(items):
            history = list(item["history"])
            question = str(item["question"]).strip()
            for candidate_index, candidate in enumerate(item["candidates"]):
                messages = self._answer_prompt(
                    history=history,
                    question=question,
                    candidate=str(candidate),
                )
                prompt_text = self._render_messages(
                    messages,
                    enable_thinking=self.config.answer_prompt_enable_thinking,
                )
                prompt_ids = self.tokenizer.encode(
                    prompt_text,
                    add_special_tokens=False,
                )
                continuations = [
                    self._continuation_ids(prompt_text, prompt_ids, label)
                    for label in labels
                ]
                label_ids = [entry[0] for entry in continuations]
                prefix_ok = all(entry[1] for entry in continuations)
                if any(not ids for ids in label_ids):
                    raise ValueError(
                        "Could not tokenize all answer labels: "
                        f"{ANSWER_LABELS!r}."
                    )
                self._check_sequence_length(
                    prompt_tokens=len(prompt_ids),
                    extra_tokens=max(len(ids) for ids in label_ids),
                    operation="LLM-BED likelihood scoring",
                )
                pairs.append(
                    {
                        "item_index": item_index,
                        "candidate_index": candidate_index,
                        "prompt_text": prompt_text,
                        "prompt_ids": prompt_ids,
                        "label_ids": label_ids,
                        "prefix_ok": prefix_ok,
                    }
                )
        return pairs

    @torch.no_grad()
    def _score_single_token_pairs(
        self,
        pairs: Sequence[dict[str, Any]],
    ) -> list[np.ndarray]:
        rows = [None] * len(pairs)
        batch_size = int(self.config.likelihood_batch_size)
        old_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            for start in range(0, len(pairs), batch_size):
                batch_pairs = list(pairs[start : start + batch_size])
                max_len = max(len(row["prompt_ids"]) for row in batch_pairs)
                input_ids = torch.full(
                    (len(batch_pairs), max_len),
                    fill_value=int(self.tokenizer.pad_token_id),
                    dtype=torch.long,
                    device=self._device(),
                )
                attention_mask = torch.zeros(
                    (len(batch_pairs), max_len),
                    dtype=torch.long,
                    device=self._device(),
                )
                last_positions = []
                for row_idx, row in enumerate(batch_pairs):
                    ids = torch.tensor(
                        row["prompt_ids"],
                        dtype=torch.long,
                        device=self._device(),
                    )
                    pad_len = max_len - len(row["prompt_ids"])
                    input_ids[row_idx, pad_len:] = ids
                    attention_mask[row_idx, pad_len:] = 1
                    last_positions.append(max_len - 1)

                with self._evaluation_mode():
                    start_time = time.perf_counter()
                    outputs = self._model_forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        logits_to_keep=1,
                    )
                    self._add_time(
                        "likelihood_scoring_seconds",
                        time.perf_counter() - start_time,
                    )

                for row_idx, row in enumerate(batch_pairs):
                    token_ids = [
                        int(label_ids[0])
                        for label_ids in row["label_ids"]
                    ]
                    logits_position = (
                        -1
                        if int(outputs.logits.shape[1]) == 1
                        else last_positions[row_idx]
                    )
                    logits = outputs.logits[
                        row_idx,
                        logits_position,
                        token_ids,
                    ].float()
                    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                    rows[start + row_idx] = probs.astype(np.float64)

                del outputs, input_ids, attention_mask
        finally:
            self.tokenizer.padding_side = old_padding

        return [np.asarray(row, dtype=np.float64) for row in rows]

    @torch.no_grad()
    def _score_sequence_pairs(
        self,
        pairs: Sequence[dict[str, Any]],
    ) -> list[np.ndarray]:
        labels_per_pair = len(ANSWER_LABELS)
        log_scores = np.full(
            (len(pairs), labels_per_pair),
            -np.inf,
            dtype=np.float64,
        )
        rows = []
        for pair_index, pair in enumerate(pairs):
            for label_index, label_ids in enumerate(pair["label_ids"]):
                rows.append(
                    {
                        "pair_index": pair_index,
                        "label_index": label_index,
                        "input_ids": pair["prompt_ids"] + list(label_ids),
                        "label_start": len(pair["prompt_ids"]),
                        "label_length": len(label_ids),
                    }
                )

        batch_size = int(self.config.likelihood_batch_size)
        old_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            for start in range(0, len(rows), batch_size):
                batch_rows = rows[start : start + batch_size]
                max_len = max(len(row["input_ids"]) for row in batch_rows)
                input_ids = torch.full(
                    (len(batch_rows), max_len),
                    fill_value=int(self.tokenizer.pad_token_id),
                    dtype=torch.long,
                    device=self._device(),
                )
                attention_mask = torch.zeros(
                    (len(batch_rows), max_len),
                    dtype=torch.long,
                    device=self._device(),
                )
                label_starts = []
                for row_idx, row in enumerate(batch_rows):
                    ids = torch.tensor(
                        row["input_ids"],
                        dtype=torch.long,
                        device=self._device(),
                    )
                    pad_len = max_len - len(row["input_ids"])
                    input_ids[row_idx, pad_len:] = ids
                    attention_mask[row_idx, pad_len:] = 1
                    label_starts.append(pad_len + int(row["label_start"]))

                with self._evaluation_mode():
                    start_time = time.perf_counter()
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                    self._add_time(
                        "likelihood_scoring_seconds",
                        time.perf_counter() - start_time,
                    )

                for row_idx, row in enumerate(batch_rows):
                    label_start = label_starts[row_idx]
                    label_end = label_start + int(row["label_length"])
                    if label_start <= 0:
                        raise ValueError("Answer label has no preceding token.")
                    positions = torch.arange(
                        label_start - 1,
                        label_end - 1,
                        device=outputs.logits.device,
                    )
                    target_ids = input_ids[row_idx, label_start:label_end]
                    answer_logits = outputs.logits[row_idx, positions, :].float()
                    log_probs = torch.log_softmax(answer_logits, dim=-1)
                    token_log_probs = log_probs.gather(
                        dim=-1,
                        index=target_ids.unsqueeze(-1),
                    ).squeeze(-1)
                    log_scores[
                        int(row["pair_index"]),
                        int(row["label_index"]),
                    ] = float(token_log_probs.sum().item())

                del outputs, input_ids, attention_mask
        finally:
            self.tokenizer.padding_side = old_padding

        probs_by_pair = []
        for scores in log_scores:
            if not np.all(np.isfinite(scores)):
                probs = np.full(labels_per_pair, 1.0 / labels_per_pair)
            else:
                shifted = scores - np.max(scores)
                probs = np.exp(shifted)
                probs = probs / probs.sum()
            probs_by_pair.append(probs.astype(np.float64))

        return probs_by_pair

    def score_answer_probability_matrices(
        self,
        items: Sequence[dict[str, Any]],
    ) -> list[np.ndarray]:
        items = [
            {
                "history": list(item["history"]),
                "question": str(item["question"]),
                "candidates": list(item["candidates"]),
            }
            for item in items
        ]
        if self.config.likelihood_source == "external":
            return self._score_external_answer_probability_matrices(items)

        matrices = [
            np.empty((len(item["candidates"]), len(ANSWER_LABELS)), dtype=np.float64)
            for item in items
        ]
        pairs = self._prepare_scoring_pairs(items)
        self._last_answer_scoring_debug = [[] for _ in items]
        if self.config.debug:
            for pair in pairs:
                item_index = int(pair["item_index"])
                candidate_index = int(pair["candidate_index"])
                self._last_answer_scoring_debug[item_index].append(
                    {
                        "candidate": items[item_index]["candidates"][candidate_index],
                        "likelihood_source": "policy",
                        "prompt": pair["prompt_text"],
                        "prompt_tokens": int(len(pair["prompt_ids"])),
                        "answer_label_token_ids": {
                            label: list(map(int, label_ids))
                            for label, label_ids in zip(
                                ANSWER_LABELS,
                                pair["label_ids"],
                                strict=True,
                            )
                        },
                        "single_token_restricted_softmax": bool(
                            pair["prefix_ok"]
                            and all(len(label_ids) == 1 for label_ids in pair["label_ids"])
                        ),
                    }
                )
        if not pairs:
            return matrices

        all_single_token = all(
            pair["prefix_ok"]
            and all(len(label_ids) == 1 for label_ids in pair["label_ids"])
            for pair in pairs
        )
        pair_probs = (
            self._score_single_token_pairs(pairs)
            if all_single_token
            else self._score_sequence_pairs(pairs)
        )

        eps = float(self.config.likelihood_epsilon)
        for pair, probs in zip(pairs, pair_probs, strict=True):
            probs = np.asarray(probs, dtype=np.float64)
            probs = np.clip(probs, eps, 1.0)
            probs = probs / probs.sum()
            item_index = int(pair["item_index"])
            candidate_index = int(pair["candidate_index"])
            matrices[item_index][candidate_index] = probs
            if self.config.debug:
                self._last_answer_scoring_debug[item_index][candidate_index][
                    "answer_probabilities"
                ] = {
                    label: float(probability)
                    for label, probability in zip(ANSWER_LABELS, probs, strict=True)
                }

        self._add_count("likelihood_pairs", len(pairs))
        return matrices

    def _score_external_answer_probability_matrices(
        self,
        items: Sequence[dict[str, Any]],
    ) -> list[np.ndarray]:
        cache_path = self.config.external_likelihood_cache_path
        if not cache_path:
            raise RuntimeError(
                "External LLM-BED likelihood scoring requires "
                "game.llm_bed.external_likelihood_cache_path."
            )

        matrices = [
            np.empty((len(item["candidates"]), len(ANSWER_LABELS)), dtype=np.float64)
            for item in items
        ]
        self._last_answer_scoring_debug = [[] for _ in items]

        grouped_item_indices: dict[tuple[str, ...], list[int]] = {}
        for item_index, item in enumerate(items):
            candidates = tuple(str(candidate) for candidate in item["candidates"])
            if not candidates:
                continue
            grouped_item_indices.setdefault(candidates, []).append(item_index)

        eps = float(self.config.likelihood_epsilon)
        fallback = np.full(
            len(ANSWER_LABELS),
            1.0 / len(ANSWER_LABELS),
            dtype=np.float64,
        )
        missing_count = 0
        pair_count = 0

        batch_size = int(self.config.likelihood_batch_size)
        batch_count = 0

        for candidates, item_indices in grouped_item_indices.items():
            for start_index in range(0, len(item_indices), batch_size):
                batch_item_indices = item_indices[start_index : start_index + batch_size]
                questions = [
                    str(items[item_index]["question"])
                    for item_index in batch_item_indices
                ]
                start_time = time.perf_counter()
                rows_by_question = tq_feedback(
                    questions=questions,
                    ground_truths=list(candidates),
                    cache_path=str(cache_path),
                    usage_tracker=self.usage_tracker,
                    track_cache_status=self.usage_tracker is not None,
                )
                self._add_time(
                    "external_likelihood_scoring_seconds",
                    time.perf_counter() - start_time,
                )
                batch_count += 1

                if len(rows_by_question) != len(batch_item_indices):
                    raise RuntimeError(
                        "External likelihood scorer returned "
                        f"{len(rows_by_question)} rows for "
                        f"{len(batch_item_indices)} questions."
                    )

                for item_index, raw_rows in zip(
                    batch_item_indices,
                    rows_by_question,
                    strict=True,
                ):
                    if len(raw_rows) != len(candidates):
                        raise RuntimeError(
                            "External likelihood scorer returned "
                            f"{len(raw_rows)} entries for {len(candidates)} candidates."
                        )
                    for candidate_index, raw_distribution in enumerate(raw_rows):
                        probs = _normalise_external_answer_distribution(
                            raw_distribution,
                            eps=eps,
                        )
                        missing_or_malformed = probs is None
                        if probs is None:
                            probs = fallback.copy()
                            missing_count += 1

                        matrices[item_index][candidate_index] = probs
                        pair_count += 1

                        if self.config.debug:
                            self._last_answer_scoring_debug[item_index].append(
                                {
                                    "candidate": candidates[candidate_index],
                                    "likelihood_source": "external",
                                    "external_likelihood_cache_path": str(cache_path),
                                    "external_likelihood_batch_size": batch_size,
                                    "raw_distribution": raw_distribution,
                                    "missing_or_malformed_external_distribution": bool(
                                        missing_or_malformed
                                    ),
                                    "answer_probabilities": {
                                        label: float(probability)
                                        for label, probability in zip(
                                            ANSWER_LABELS,
                                            probs,
                                            strict=True,
                                        )
                                    },
                                }
                            )

        self._add_count("likelihood_pairs", pair_count)
        self._add_count("external_likelihood_batches", batch_count)
        self._add_count("external_likelihood_questions", len(items))
        self._add_count("external_likelihood_pairs", pair_count)
        if missing_count:
            self._add_count("external_likelihood_missing_or_malformed", missing_count)
        return matrices

    def _filter_candidates(
        self,
        candidates: Sequence[str],
        history: Sequence[dict[str, str]],
    ) -> list[str]:
        candidates = list(candidates)
        if not candidates:
            return []
        if not history:
            return candidates

        items = []
        observed_answers = []
        history = list(history)
        for turn_idx, step in enumerate(history):
            items.append(
                {
                    "history": history[:turn_idx],
                    "question": str(step["question"]),
                    "candidates": candidates,
                }
            )
            observed_answers.append(str(step["answer"]).strip().lower())

        matrices = self.score_answer_probability_matrices(items)
        compatible = []
        for candidate_index, candidate in enumerate(candidates):
            keep = True
            for matrix, observed_answer in zip(
                matrices,
                observed_answers,
                strict=True,
            ):
                try:
                    answer_index = ANSWER_LABELS.index(observed_answer)
                except ValueError:
                    keep = False
                    break
                if (
                    float(matrix[candidate_index, answer_index])
                    < float(self.config.consistency_threshold)
                ):
                    keep = False
                    break
            if keep:
                compatible.append(candidate)
        return compatible

    def sample_fresh_hypotheses(
        self,
        history: Sequence[dict[str, str]],
    ) -> tuple[list[str], dict[str, Any]]:
        candidates: list[str] = []
        stats: dict[str, Any] = {
            "belief_generations": 0,
            "proposed_candidates": 0,
            "accepted_candidates": 0,
            "fresh_hypotheses": True,
        }
        if self.config.debug:
            stats["hypothesis_generation_prompts"] = []
            stats["raw_hypothesis_outputs"] = []

        for _ in range(int(self.config.refill_rounds)):
            needed = int(self.config.num_hypotheses) - len(candidates)
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
                do_sample=True,
                temperature=float(self.config.hypothesis_sampling_temperature),
                max_new_tokens=int(self.config.belief_max_new_tokens),
                enable_thinking=self.config.hypothesis_prompt_enable_thinking,
            )
            stats["belief_generations"] += 1
            if self.config.debug:
                for row in rows:
                    if "prompt" in row:
                        stats["hypothesis_generation_prompts"].append(row["prompt"])
                    stats["raw_hypothesis_outputs"].append(row.get("text", ""))
            if not rows:
                continue

            proposed = parse_candidate_list(rows[0]["text"], existing=candidates)
            stats["proposed_candidates"] += len(proposed)
            compatible = self._filter_candidates(proposed, history)
            seen = {normalize_lookup_key(candidate) for candidate in candidates}
            for candidate in compatible:
                key = normalize_lookup_key(candidate)
                if key and key not in seen:
                    candidates.append(candidate)
                    seen.add(key)
                    stats["accepted_candidates"] += 1
                if len(candidates) >= int(self.config.num_hypotheses):
                    break

        self._add_count("belief_generations", int(stats["belief_generations"]))
        self._add_count("proposed_hypotheses", int(stats["proposed_candidates"]))
        self._add_count("accepted_hypotheses", int(stats["accepted_candidates"]))
        return candidates[: int(self.config.num_hypotheses)], stats

    def evaluate_questions(
        self,
        *,
        histories: Sequence[Sequence[dict[str, str]]],
        questions: Sequence[str],
        candidate_sets: Sequence[Sequence[str]] | None = None,
        hypothesis_stats: Sequence[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if len(histories) != len(questions):
            raise ValueError("histories and questions must have equal length.")
        if candidate_sets is not None and len(candidate_sets) != len(questions):
            raise ValueError("candidate_sets and questions must have equal length.")
        if hypothesis_stats is not None and len(hypothesis_stats) != len(questions):
            raise ValueError("hypothesis_stats and questions must have equal length.")

        candidates_by_item = []
        stats_by_item = []
        if candidate_sets is None:
            history_cache: dict[str, tuple[list[str], dict[str, Any]]] = {}
            for history in histories:
                history_list = [dict(step) for step in history]
                key = json.dumps(history_list, sort_keys=True, ensure_ascii=False)
                if key not in history_cache:
                    history_cache[key] = self.sample_fresh_hypotheses(history_list)
                candidates, stats = history_cache[key]
                candidates_by_item.append(list(candidates))
                stats_by_item.append(dict(stats))
        else:
            for item_index, candidates in enumerate(candidate_sets):
                candidates = [str(candidate) for candidate in candidates]
                candidates_by_item.append(candidates)
                if hypothesis_stats is not None:
                    stats = dict(hypothesis_stats[item_index])
                else:
                    stats = {
                        "belief_generations": 0,
                        "proposed_candidates": len(candidates),
                        "accepted_candidates": len(candidates),
                    }
                stats.setdefault("fresh_hypotheses", False)
                stats["precomputed_hypotheses"] = True
                stats_by_item.append(stats)

        scoring_items = [
            {
                "history": [dict(step) for step in history],
                "question": question,
                "candidates": candidates,
            }
            for history, question, candidates in zip(
                histories,
                questions,
                candidates_by_item,
                strict=True,
            )
        ]
        matrices = self.score_answer_probability_matrices(scoring_items)

        results = []
        for history, question, candidates, stats, matrix in zip(
            histories,
            questions,
            candidates_by_item,
            stats_by_item,
            matrices,
            strict=True,
        ):
            if len(candidates) < 2:
                eig = 0.0
                predictive = [1.0 / len(ANSWER_LABELS)] * len(ANSWER_LABELS)
            else:
                eig, predictive = compute_uniform_eig(
                    matrix,
                    eps=float(self.config.likelihood_epsilon),
                )
            result = {
                "reward_mode": "llm_bed",
                "history": [dict(step) for step in history],
                "question": str(question),
                "hypotheses": list(candidates),
                "num_hypotheses": int(len(candidates)),
                "hypothesis_stats": stats,
                "answer_labels": list(ANSWER_LABELS),
                "answer_probabilities": matrix.tolist(),
                "predictive_distribution": predictive,
                "eig": float(eig),
                "likelihood_source": self.config.likelihood_source,
                "hypothesis_source": "policy",
                "external_likelihood_cache_path": (
                    self.config.external_likelihood_cache_path
                    if self.config.likelihood_source == "external"
                    else None
                ),
                "used_stored_sampled_candidates": False,
                "used_contrastive_set": False,
                "used_likelihood_cache": self.config.likelihood_source == "external",
            }
            if self.config.debug:
                result["answer_scoring_debug"] = self._last_answer_scoring_debug[
                    len(results)
                ]
            results.append(result)
        return results


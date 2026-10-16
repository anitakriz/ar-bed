from __future__ import annotations

import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence

from openai import (
    APIConnectionError,
    APITimeoutError,
    AzureOpenAI,
    InternalServerError,
    RateLimitError,
)


@dataclass
class ChatResult:
    text: str
    usage: Dict[str, int] = field(default_factory=dict)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def response_usage_counts(response: Any) -> Dict[str, int]:
    # Normalize OpenAI/Azure SDK usage fields into stable local counter names.
    usage = _get(response, "usage")
    if usage is None:
        return {}

    input_tokens = (
        _get(usage, "prompt_tokens")
        or _get(usage, "input_tokens")
        or 0
    )
    output_tokens = (
        _get(usage, "completion_tokens")
        or _get(usage, "output_tokens")
        or 0
    )
    total_tokens = (
        _get(usage, "total_tokens")
        or (input_tokens + output_tokens)
        or 0
    )
    input_details = (
        _get(usage, "prompt_tokens_details")
        or _get(usage, "input_tokens_details")
    )
    output_details = (
        _get(usage, "completion_tokens_details")
        or _get(usage, "output_tokens_details")
    )

    return {
        "input_tokens": int(input_tokens or 0),
        "cached_input_tokens": int(_get(input_details, "cached_tokens", 0) or 0),
        "output_tokens": int(output_tokens or 0),
        "reasoning_output_tokens": int(
            _get(output_details, "reasoning_tokens", 0) or 0
        ),
        "total_tokens": int(total_tokens or 0),
    }


GPT5_DEFAULT_PRICES_PER_M_TOKEN = {
    "input": 1.25,
    "cached_input": 0.125,
    "output": 10.0,
}


def deployment_prices_per_m_token(
    deployment_name: Optional[str] = None,
) -> Dict[str, float]:
    # Use the GPT-5 rates as the default/fallback for the current Azure deployment.
    deployment = (
        deployment_name
        or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
    ).lower()
    if "gpt-5" in deployment:
        return dict(GPT5_DEFAULT_PRICES_PER_M_TOKEN)
    return dict(GPT5_DEFAULT_PRICES_PER_M_TOKEN)


def estimate_prefix_cost_usd(
    counts: Dict[str, int],
    *,
    prefix: str,
    prices: Optional[Dict[str, float]] = None,
) -> float:
    # Pricing is per 1M tokens: uncached input, cached input, and output.
    prices = prices or deployment_prices_per_m_token()
    input_tokens = int(counts.get(f"{prefix}_input_tokens", 0))
    cached_tokens = int(counts.get(f"{prefix}_cached_input_tokens", 0))
    output_tokens = int(counts.get(f"{prefix}_output_tokens", 0))
    uncached_input_tokens = max(input_tokens - cached_tokens, 0)
    cost = (
        uncached_input_tokens * float(prices["input"])
        + cached_tokens * float(prices["cached_input"])
        + output_tokens * float(prices["output"])
    ) / 1_000_000.0
    return round(cost, 6)


def add_estimated_costs(
    counts: Dict[str, Any],
    *,
    deployment_name: Optional[str] = None,
) -> Dict[str, Any]:
    # Add derived rough USD costs to a token usage/count dictionary.
    enriched = dict(counts)
    deployment = deployment_name or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
    prices = deployment_prices_per_m_token(deployment)
    questioner_cost = estimate_prefix_cost_usd(
        enriched,
        prefix="questioner",
        prices=prices,
    )
    oracle_cost = estimate_prefix_cost_usd(
        enriched,
        prefix="oracle",
        prices=prices,
    )
    enriched["estimated_questioner_cost_usd"] = questioner_cost
    enriched["estimated_oracle_cost_usd"] = oracle_cost
    enriched["estimated_total_cost_usd"] = round(questioner_cost + oracle_cost, 6)
    enriched["estimated_cost_pricing_per_m_tokens"] = {
        "deployment": deployment,
        "input": prices["input"],
        "cached_input": prices["cached_input"],
        "output": prices["output"],
    }
    return enriched


class ApiUsageTracker:
    """Thread-safe API usage/cost accumulator for GPT-backed helpers."""

    def __init__(self, initial_counts: Optional[Dict[str, Any]] = None) -> None:
        self.counts = Counter()
        self._lock = Lock()
        if initial_counts:
            self.update(initial_counts)

    def update(self, counts: Dict[str, Any]) -> None:
        """Merge persisted raw counters, ignoring derived cost metadata."""
        ignored = {
            "estimated_questioner_cost_usd",
            "estimated_oracle_cost_usd",
            "estimated_total_cost_usd",
            "estimated_cost_pricing_per_m_tokens",
        }
        with self._lock:
            for key, value in counts.items():
                if key in ignored:
                    continue
                if isinstance(value, bool):
                    continue
                if isinstance(value, int):
                    self.counts[key] += value
                elif isinstance(value, float) and value.is_integer():
                    self.counts[key] += int(value)

    def _record_usage_counts(self, prefix: str, usage: Dict[str, int]) -> None:
        with self._lock:
            self.counts[f"{prefix}_api_calls"] += 1
            self.counts[f"{prefix}_input_tokens"] += int(
                usage.get("input_tokens", 0)
            )
            self.counts[f"{prefix}_cached_input_tokens"] += int(
                usage.get("cached_input_tokens", 0)
            )
            self.counts[f"{prefix}_output_tokens"] += int(
                usage.get("output_tokens", 0)
            )
            self.counts[f"{prefix}_reasoning_output_tokens"] += int(
                usage.get("reasoning_output_tokens", 0)
            )
            self.counts[f"{prefix}_total_tokens"] += int(
                usage.get("total_tokens", 0)
            )

    def record_response_usage(self, prefix: str, response: Any) -> None:
        """Count one SDK response and its token usage if metadata is present."""
        self._record_usage_counts(prefix, response_usage_counts(response))

    def record_questioner_response(self, response: Any) -> None:
        self.record_response_usage("questioner", response)

    def record_questioner_result(self, result: ChatResult) -> None:
        self._record_usage_counts("questioner", result.usage)

    def record_questioner_results(self, results: Sequence[ChatResult]) -> None:
        for result in results:
            self.record_questioner_result(result)

    def record_oracle_response(self, response: Any) -> None:
        self.record_response_usage("oracle", response)

    def record_oracle_result(self, result: ChatResult) -> None:
        self._record_usage_counts("oracle", result.usage)

    def record_oracle_results(self, results: Sequence[ChatResult]) -> None:
        for result in results:
            self.record_oracle_result(result)

    def record_oracle_lookup(
        self,
        *,
        num_questions: int,
        num_ground_truths: int,
        cache_hits: int,
        cache_misses: int,
    ) -> None:
        self.record_oracle_lookup_slots(
            num_questions=num_questions,
            num_ground_truth_slots=int(num_questions) * int(num_ground_truths),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

    def record_oracle_lookup_slots(
        self,
        *,
        num_questions: int,
        num_ground_truth_slots: int,
        cache_hits: int,
        cache_misses: int,
    ) -> None:
        """Count cache lookup slots for ragged question/candidate batches."""
        with self._lock:
            self.counts["oracle_tq_feedback_invocations"] += 1
            self.counts["oracle_question_batches"] += int(num_questions)
            self.counts["oracle_ground_truth_slots"] += int(num_ground_truth_slots)
            self.counts["oracle_cache_hits"] += int(cache_hits)
            self.counts["oracle_cache_misses"] += int(cache_misses)
            if cache_misses:
                self.counts["oracle_batches_with_cache_misses"] += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            counts = dict(sorted(self.counts.items()))
        return add_estimated_costs(counts)


class AzureChatBatchClient:
    def __init__(
        self,
        *,
        api_key_env: str = "AZURE_OPENAI_API_KEY",
        endpoint_env: str = "AZURE_OPENAI_ENDPOINT",
        api_version_env: str = "AZURE_OPENAI_API_VERSION",
        deployment_env: str = "AZURE_OPENAI_DEPLOYMENT",
        timeout: float = 180.0,
    ) -> None:
        self.api_key = os.environ[api_key_env]
        self.azure_endpoint = os.environ[endpoint_env]
        self.api_version = os.environ[api_version_env]
        self.deployment_name = os.environ[deployment_env]
        self.timeout = float(timeout)

    def _client(self) -> AzureOpenAI:
        return AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
            max_retries=0,
            timeout=self.timeout,
        )

    def complete_one(
        self,
        messages: List[Dict[str, str]],
        *,
        max_output_tokens: int,
        max_retries: int,
        reasoning_effort: Optional[str],
    ) -> ChatResult:
        retryable_errors = (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )
        # GPT-5 reasoning models on Chat Completions use
        # `max_completion_tokens`. Do not probe legacy `max_tokens`/no-limit
        # fallbacks after a 400, because that can hide an invalid request.
        request = {
            "model": self.deployment_name,
            "messages": messages,
            "max_completion_tokens": int(max_output_tokens),
        }

        normalized_reasoning_effort = (
            str(reasoning_effort).strip()
            if reasoning_effort is not None
            else None
        )
        if normalized_reasoning_effort:
            request["reasoning_effort"] = normalized_reasoning_effort

        num_attempts = max(1, int(max_retries))
        for attempt in range(num_attempts):
            try:
                response = self._client().chat.completions.create(**request)
                return ChatResult(
                    text=response.choices[0].message.content or "",
                    usage=response_usage_counts(response),
                )
            except retryable_errors as error:
                if attempt == num_attempts - 1:
                    raise
                delay = min(60.0, 5.0 * (2 ** attempt))
                delay += random.uniform(0.0, 2.0)
                print(
                    f"Azure chat request failed: {error}. "
                    f"Retrying in {delay:.1f}s "
                    f"({attempt + 1}/{num_attempts})...",
                    flush=True,
                )
                time.sleep(delay)

        raise RuntimeError("Azure chat request failed before receiving a response.")

    def complete_many(
        self,
        messages_batch: Sequence[List[Dict[str, str]]],
        *,
        batch_size: int,
        max_output_tokens: int,
        max_retries: int,
        reasoning_effort: Optional[str],
    ) -> List[ChatResult]:
        # Execute each chunk concurrently and preserve input order in the result list.
        messages_batch = list(messages_batch)
        if not messages_batch:
            return []

        batch_size = max(1, int(batch_size))
        results: List[Optional[ChatResult]] = [None] * len(messages_batch)

        for start in range(0, len(messages_batch), batch_size):
            end = min(start + batch_size, len(messages_batch))
            with ThreadPoolExecutor(max_workers=end - start) as executor:
                futures = {
                    executor.submit(
                        self.complete_one,
                        messages_batch[index],
                        max_output_tokens=max_output_tokens,
                        max_retries=max_retries,
                        reasoning_effort=reasoning_effort,
                    ): index
                    for index in range(start, end)
                }
                for future in as_completed(futures):
                    results[futures[future]] = future.result()

        missing = [index for index, result in enumerate(results) if result is None]
        if missing:
            raise RuntimeError(f"Missing Azure chat results for indices: {missing}")
        return [result for result in results if result is not None]
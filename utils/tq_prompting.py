from __future__ import annotations

from typing import Any, Mapping


REASONING_SYSTEM_PROMPT = """You are playing Twenty Questions.

Ask one informative question at a time by calling the `ask_question` tool.
The question must be answerable with "yes", "no", or "maybe".
Call `ask_question` exactly once.
Do not output the question as ordinary text.

The tool response contains both an answer and a status:
- `Status: continue` means the exact hidden target has not yet been identified. Continue asking questions, even if the answer was "yes".
- `Status: solved` means the exact hidden target has been identified and the game is complete.
""".strip()


DIRECT_SYSTEM_PROMPT = """You are playing Twenty Questions.

Ask one informative question at a time by calling the `ask_question` tool.
The question must be answerable with "yes", "no", or "maybe".
Call `ask_question` exactly once.
Directly output the tool call with no reasoning, explanation, or ordinary text before or after it.

The tool response contains both an answer and a status:
- `Status: continue` means the exact hidden target has not yet been identified. Continue asking questions, even if the answer was "yes".
- `Status: solved` means the exact hidden target has been identified and the game is complete.
""".strip()


def tq_question_guidance(config: Mapping[str, Any]) -> str | None:
    """Return optional domain guidance appended to the standard TQ prompt."""
    game_config = config.get("game", {}) or {}
    value = game_config.get("question_guidance")
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def tq_reasoning_enabled(config: Mapping[str, Any]) -> bool:
    """Resolve the canonical TQ reasoning switch with legacy compatibility."""
    model_config = config.get("model", {}) or {}
    if "enable_reasoning" in model_config:
        value = model_config["enable_reasoning"]
    else:
        training_config = config.get("training", {}) or {}
        template_kwargs = training_config.get("chat_template_kwargs", {}) or {}
        value = template_kwargs.get("enable_thinking", True)

    if not isinstance(value, bool):
        raise ValueError(
            "model.enable_reasoning must be a boolean. "
            f"Got: {value!r}."
        )
    return value


def tq_system_prompt(config: Mapping[str, Any]) -> str:
    prompt = (
        REASONING_SYSTEM_PROMPT
        if tq_reasoning_enabled(config)
        else DIRECT_SYSTEM_PROMPT
    )
    guidance = tq_question_guidance(config)
    if guidance:
        prompt = f"{prompt}\n\nDomain guidance:\n{guidance}"
    return prompt


def tq_chat_template_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return template kwargs synchronized with the reasoning switch."""
    training_config = config.get("training", {}) or {}
    kwargs = dict(training_config.get("chat_template_kwargs", {}) or {})
    kwargs["enable_thinking"] = tq_reasoning_enabled(config)
    return kwargs


def tq_row_category(
    row: Mapping[str, Any],
    *,
    category: str,
    config: Mapping[str, Any] | None = None,
) -> str:
    categories = ((config or {}).get("game", {}) or {}).get("category_by_domain")
    if categories is None:
        return category
    if not isinstance(categories, Mapping):
        raise ValueError("game.category_by_domain must be a mapping.")
    domain = row.get("domain")
    if not isinstance(domain, str) or domain not in categories:
        raise ValueError(f"No configured category for row domain: {domain!r}")
    resolved = categories[domain]
    if not isinstance(resolved, str) or not resolved.strip():
        raise ValueError(f"Category for domain {domain!r} must be nonempty text.")
    return resolved.strip()

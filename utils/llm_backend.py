import re
from types import MethodType


_GEMMA_ASK_QUESTION_RE = re.compile(
    r'call:ask_question\s*\{\s*question\s*:\s*<\|"\|>'
    r'(?P<question>.*?)'
    r'<\|"\|>\s*\}',
    re.DOTALL,
)


def patch_gemma_parse_response(text_tokenizer):
    """
    Keep the tokenizer's normal response parsing, but recover a well-formed
    native Gemma ``ask_question`` call if it is left serialized inside
    assistant content instead of being returned under ``tool_calls``.

    Malformed or incomplete tool calls are intentionally left untouched so
    that TRL can treat them as invalid rather than guessing the model's intent.
    """
    if not hasattr(text_tokenizer, "parse_response"):
        return

    # Avoid wrapping the same tokenizer more than once.
    if getattr(
        text_tokenizer,
        "_tq_parse_response_patched",
        False,
    ):
        return

    original_parse_response = text_tokenizer.parse_response

    def safe_parse_response(
        self,
        *args,
        **kwargs,
    ):
        parsed = original_parse_response(
            *args,
            **kwargs,
        )

        # Normal case: the tokenizer already produced a structured tool call.
        # Keep only the action in future rollout context; rationale text is
        # deliberately discarded so all history has the same question/answer
        # tool-call representation.
        if (
            isinstance(parsed, dict)
            and parsed.get("tool_calls")
        ):
            parsed["content"] = ""

            # Enforce exactly one environment action per assistant turn.
            # TRL executes every structured tool call returned by the parser
            # before generating again, so keep only the first call. This
            # guarantees:
            #
            #   generate Q_t -> observe A_t -> generate Q_{t+1}
            #
            # rather than allowing multiple questions from one generation.
            tool_calls = parsed.get("tool_calls") or []

            if len(tool_calls) > 1:
                print(
                    f"WARNING: model produced {len(tool_calls)} tool calls "
                    "in one assistant turn; keeping only the first.",
                    flush=True,
                )

            parsed["tool_calls"] = tool_calls[:1]

            return parsed

        # Search every plausible textual representation. In some Gemma 4 /
        # Transformers paths, parse_response leaves the native tool-call
        # sequence inside parsed["content"] instead of exposing tool_calls.
        candidate_texts = []

        if args and isinstance(args[0], str):
            candidate_texts.append(args[0])

        if isinstance(kwargs.get("response"), str):
            candidate_texts.append(kwargs["response"])

        if isinstance(kwargs.get("text"), str):
            candidate_texts.append(kwargs["text"])

        if isinstance(parsed, dict):
            parsed_content = parsed.get("content")
            if isinstance(parsed_content, str):
                candidate_texts.append(parsed_content)

        for candidate_text in candidate_texts:
            match = _GEMMA_ASK_QUESTION_RE.search(
                candidate_text
            )

            if match is None:
                continue

            question = match.group(
                "question"
            ).strip()

            if not question:
                continue

            # Recover only the action. Do not carry rationale text into the
            # next rollout turn, so generated turns match historical turns.
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "ask_question",
                            "arguments": {
                                "question": question,
                            },
                        },
                    }
                ],
            }

        # Preserve compatibility with code that expects a content key.
        if (
            isinstance(parsed, dict)
            and not parsed.get("content")
        ):
            parsed["content"] = ""

        return parsed

    text_tokenizer.parse_response = MethodType(
        safe_parse_response,
        text_tokenizer,
    )
    text_tokenizer._tq_parse_response_patched = True


def load_model_and_tokenizer(
    config,
    for_training=False,
    for_inference=False,
):
    from unsloth import FastLanguageModel

    model_config = config.get("model", {})
    lora_config = model_config.get("lora", {})

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.get(
            "name",
            "unsloth/gemma-4-E4B-it",
        ),
        load_in_4bit=bool(
            model_config.get("load_in_4bit", True)
        ),
        fast_inference=bool(
            model_config.get("fast_inference", False)
        ),
    )

    model_name = model_config.get(
        "name",
        "unsloth/gemma-4-E4B-it",
    )

    if model_name.casefold().startswith(("unsloth/gemma", "google/gemma")):
        model.config.max_position_embeddings = (
            model.config.text_config.max_position_embeddings
        )

    # Gemma 4 returns a processor containing the actual text tokenizer.
    text_tokenizer = getattr(
        tokenizer,
        "tokenizer",
        tokenizer,
    )

    text_tokenizer.padding_side = "left"
    text_tokenizer.truncation_side = "left"

    if text_tokenizer.pad_token_id is None:
        text_tokenizer.pad_token = (
            text_tokenizer.eos_token
        )

    # Patch before GRPOTrainer receives this tokenizer as processing_class.
    patch_gemma_parse_response(
        text_tokenizer
    )

    if (
        for_training
        and bool(lora_config.get("enabled", True))
    ):
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(lora_config.get("r", 16)),
            lora_alpha=int(
                lora_config.get(
                    "alpha",
                    int(lora_config.get("r", 16)) * 2,
                )
            ),
            target_modules=lora_config.get(
                "target_modules",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
            bias=lora_config.get("bias", "none"),
            use_gradient_checkpointing=lora_config.get(
                "use_gradient_checkpointing",
                "unsloth",
            ),
            random_state=int(
                lora_config.get("random_state", 3407)
            ),
        )

    if for_inference:
        FastLanguageModel.for_inference(model)
        model.eval()

    return model, tokenizer


def load_model_and_tokenizer_from_checkpoint(
    config,
    checkpoint_path,
    for_inference=True,
):
    from unsloth import FastLanguageModel

    model_config = config.get("model", {})

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint_path),
        load_in_4bit=bool(
            model_config.get("load_in_4bit", True)
        ),
        fast_inference=bool(
            model_config.get("fast_inference", False)
        ),
    )

    model_name = model_config.get(
        "name",
        "unsloth/gemma-4-E4B-it",
    )

    if model_name.casefold().startswith(("unsloth/gemma", "google/gemma")):
        model.config.max_position_embeddings = (
            model.config.text_config.max_position_embeddings
        )

    # Keep checkpoint inference aligned with training: Gemma 4 returns
    # a processor containing the actual text tokenizer.
    text_tokenizer = getattr(
        tokenizer,
        "tokenizer",
        tokenizer,
    )

    text_tokenizer.padding_side = "left"
    text_tokenizer.truncation_side = "left"

    if text_tokenizer.pad_token_id is None:
        text_tokenizer.pad_token = (
            text_tokenizer.eos_token
        )

    # Use the same recovery behavior for checkpoint/full-game inference.
    patch_gemma_parse_response(
        text_tokenizer
    )

    if for_inference:
        FastLanguageModel.for_inference(model)
        model.eval()

    return model, tokenizer

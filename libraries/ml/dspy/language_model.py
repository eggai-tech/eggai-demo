import re
from enum import Enum
from time import perf_counter
from typing import Any, get_origin

import dspy
from dotenv import load_dotenv

_FIELD_HEADER = re.compile(r"\[\[ ## \w+ ## \]\]")
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
NO_TOOL_CALL_INSTRUCTION = (
    "You are not given any tools or functions. Never respond with a JSON function "
    "call. Respond only with the output fields in the exact `[[ ## field ## ]]` "
    "format shown above."
)


class ChatAdapter(dspy.ChatAdapter):
    """ChatAdapter that tolerates field headers glued onto one line.

    Deliberately named ``ChatAdapter``: dspy's ``StreamListener`` dispatches on
    ``settings.adapter.__class__.__name__`` against the literal strings
    ``ChatAdapter``/``XMLAdapter``/``JSONAdapter`` and rejects any other name
    ("Unsupported adapter for streaming"). Use the ``LenientChatAdapter`` alias
    in application code.

    Small local models often emit ``[[ ## field ## ]]value[[ ## completed ## ]]``
    without newlines. dspy's parser only recognises a header at the start of a
    line, so the trailing ``[[ ## completed ## ]]`` ends up inside the value
    (e.g. ``PolicyAgent[[ ## completed ## ]]`` failing enum parsing). Put every
    header on its own line before parsing.

    The JSONAdapter fallback is disabled: on a parse failure it triggers a full
    non-streamed regeneration (~20s dead gap after the token stream ends), and
    JSONAdapter's ``{``-prefixed output is swallowed as a tool call by vLLM's
    llama3_json tool parser, so it is not a usable alternative anyway.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("use_json_adapter_fallback", False)
        super().__init__(**kwargs)

    def format(self, signature, demos, inputs):
        messages = super().format(signature, demos, inputs)
        # Llama-3 models read dspy's "Your input fields are ... Your output
        # fields are ..." system prompt as a function definition and answer with
        # a native tool call ({"name": ..., "parameters": ...}) instead of the
        # field format (measured: 7/8 classifier calls on Llama-3.1-8B; on vLLM
        # that JSON is swallowed by the tool parser and content comes back
        # empty). One explicit sentence eliminates it (8/8).
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += "\n\n" + NO_TOOL_CALL_INSTRUCTION
        return messages

    def parse(self, signature, completion: str):
        # Thinking models (Qwen3, DeepSeek-R1) may leak <think>...</think> into
        # content; reasoning about the output format must not be parsed as it.
        completion = _THINK_BLOCK.sub("", completion)
        completion = _FIELD_HEADER.sub(lambda m: f"\n{m.group(0)}\n", completion)
        completion = self._normalise_sections(signature, completion)
        return super().parse(signature, completion)

    @staticmethod
    def _normalise_sections(signature, completion: str) -> str:
        """Repair per-field values that dspy's strict parser rejects.

        - Blank ``dict``/``list`` fields become ``{}``/``[]`` (ReAct's ``finish``
          step is commonly emitted with an empty ``[[ ## next_tool_args ## ]]``).
        - Enum fields reduce to the first member name found in the value:
          ``"ChattyAgent"``, ``TargetAgent.ChattyAgent``,
          ``PolicyAgent(policy_number=None)`` -> the bare member.
        Only the sections of known output fields are touched.
        """
        empty_literal = {dict: "{}", list: "[]"}
        parts = _FIELD_HEADER.split(completion)
        headers = _FIELD_HEADER.findall(completion)
        out = [parts[0]]
        for header, body in zip(headers, parts[1:], strict=True):
            name = header[len("[[ ## ") : -len(" ## ]]")]
            field = signature.output_fields.get(name)
            if field is not None:
                annotation = field.annotation
                origin = get_origin(annotation) or annotation
                value = body.strip()
                if not value and origin in empty_literal:
                    body = f"\n{empty_literal[origin]}\n"
                elif value and isinstance(annotation, type) and issubclass(annotation, Enum):
                    body = f"\n{_extract_enum_member(value, annotation)}\n"
            out.append(header + body)
        return "".join(out)


def _extract_enum_member(value: str, enum_cls: type[Enum]) -> str:
    """Return the first enum member name/value appearing as a whole word.

    Covers every decoration seen from small Llama models -- ``"X"``,
    ``TargetAgent('X')``, ``TargetAgent.X``, ``X(policy_number=None)`` -- without
    enumerating them. Falls back to the raw value so dspy reports the original.
    """
    candidates = {member.name for member in enum_cls} | {
        str(member.value) for member in enum_cls if isinstance(member.value, str)
    }
    pattern = "|".join(re.escape(c) for c in sorted(candidates, key=len, reverse=True))
    match = re.search(rf"(?<!\w)(?:{pattern})(?!\w)", value)
    return match.group(0) if match else value


LenientChatAdapter = ChatAdapter


class TrackingLM(dspy.LM):
    def __init__(self, *args, **kwargs):
        model_name = args[0] if args else ""
        self.is_lm_studio = "lm_studio" in model_name or "lm-studio" in model_name

        if self.is_lm_studio:
            kwargs.pop("response_format", None)
            self.max_context_window = 128000
        else:
            self.max_context_window = 16384

        super().__init__(*args, **kwargs)
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        self.latency_ms = 0
        self.run_logs = []
        self.model_name = model_name

    def __call__(self, *args, **kwargs):
        self.start_run()
        start_time = perf_counter()
        res = super().__call__(*args, **kwargs)
        self.latency_ms = (perf_counter() - start_time) * 1000
        return res

    def start_run(self):
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        self.latency_ms = 0

    def _truncate_prompt(self, prompt, max_length=None):
        if prompt is None:
            return None

        # ~4 chars per token
        estimated_tokens = len(prompt) / 4
        max_tokens = max_length or self.max_context_window
        available_tokens = int(max_tokens * 0.8)

        if estimated_tokens > available_tokens:
            truncation_ratio = available_tokens / estimated_tokens
            keep_chars = int(len(prompt) * truncation_ratio)
            return "..." + prompt[-keep_chars:]
        return prompt

    def _truncate_messages(self, messages, max_length=None):
        if not messages:
            return messages

        max_tokens = max_length or self.max_context_window
        available_tokens = int(max_tokens * 0.8)

        total_estimated_tokens = 0
        for msg in messages:
            content = msg.get("content", "")
            total_estimated_tokens += len(content) / 4 + 20

        if total_estimated_tokens <= available_tokens:
            return messages

        result_messages = []

        if messages and messages[0].get("role") == "system":
            result_messages.append(messages[0])
            messages = messages[1:]

        if len(messages) > 4:
            result_messages.extend(messages[-4:])
        else:
            result_messages.extend(messages)

        return result_messages

    def forward(self, prompt=None, messages=None, **kwargs):
        if self.is_lm_studio:
            kwargs.pop("response_format", None)
            if prompt:
                prompt = self._truncate_prompt(prompt, self.max_context_window)
            if messages:
                messages = self._truncate_messages(messages, self.max_context_window)

        # dspy.LM.forward is unannotated and pyright infers a union that
        # includes litellm's async overload. It is synchronous - the result is a
        # litellm ModelResponse - so the awaitable branch is not reachable here.
        forward_result: Any = super().forward(prompt, messages, **kwargs)
        self.completion_tokens += forward_result.usage.get("completion_tokens", 0)
        self.prompt_tokens += forward_result.usage.get("prompt_tokens", 0)
        self.total_tokens += forward_result.usage.get("total_tokens", 0)
        return forward_result


def dspy_set_language_model(settings, overwrite_cache_enabled: bool | None = None):
    load_dotenv()

    cache_enabled = settings.cache_enabled
    if overwrite_cache_enabled is not None:
        cache_enabled = overwrite_cache_enabled

    language_model = TrackingLM(
        settings.language_model,
        cache=cache_enabled,
        api_base=settings.language_model_api_base if settings.language_model_api_base else None,
    )

    if hasattr(settings, "max_context_window") and settings.max_context_window:
        language_model.max_context_window = settings.max_context_window

    from libraries.observability.logger import get_console_logger

    logger = get_console_logger("dspy_language_model")
    logger.info(f"Configured language model: {settings.language_model}")
    logger.info(f"Max context window: {language_model.max_context_window}")
    logger.info(f"LM Studio model: {language_model.is_lm_studio}")

    dspy.configure(lm=language_model, adapter=LenientChatAdapter())
    dspy.settings.configure(track_usage=True)

    return language_model


if __name__ == "__main__":

    class Settings:
        language_model = "lm_studio/gemma-3-12b-it-qat"
        cache_enabled = False
        language_model_api_base = "http://localhost:1234/v1"
        max_context_window = 128000

    lm = dspy_set_language_model(Settings())

    class ExtractInfo(dspy.Signature):
        """Extract structured information from text."""

        text: str = dspy.InputField()
        title: str = dspy.OutputField()
        headings: list[str] = dspy.OutputField()
        entities: list[dict[str, str]] = dspy.OutputField(
            desc="a list of entities and their metadata"
        )

    module = dspy.Predict(ExtractInfo)

    text = (
        "Apple Inc. announced its latest iPhone 14 today."
        "The CEO, Tim Cook, highlighted its new features in a press release."
    )
    response = module(text=text)

    print("Tokens printed: ", lm.total_tokens, lm.prompt_tokens, lm.completion_tokens)
    print("lm_usage: ", response.get_lm_usage())

    text = (
        "Microsoft Corporation is a technology company based in Redmond, Washington."
        "The company was founded by Bill Gates and Paul Allen in 1975."
    )
    r = module(text=text)

    print("Tokens printed: ", lm.total_tokens, lm.prompt_tokens, lm.completion_tokens)
    print("lm_usage: ", r.get_lm_usage())

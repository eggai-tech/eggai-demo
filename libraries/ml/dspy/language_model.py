from time import perf_counter

import dspy
from dotenv import load_dotenv


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

        forward_result = super().forward(prompt, messages, **kwargs)
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
        api_base=settings.language_model_api_base
        if settings.language_model_api_base
        else None,
    )

    if hasattr(settings, "max_context_window") and settings.max_context_window:
        language_model.max_context_window = settings.max_context_window

    from libraries.observability.logger import get_console_logger

    logger = get_console_logger("dspy_language_model")
    logger.info(f"Configured language model: {settings.language_model}")
    logger.info(f"Max context window: {language_model.max_context_window}")
    logger.info(f"LM Studio model: {language_model.is_lm_studio}")

    dspy.configure(lm=language_model)
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

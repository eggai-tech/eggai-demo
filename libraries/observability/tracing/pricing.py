#!/usr/bin/env python3

from typing import Any

# Pricing per 1K tokens (input/output) by model
DEFAULT_PRICING_DATA = {
    "chat": {
        "gpt-4o": {"input": 0.0005, "output": 0.0015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4": {"input": 0.003, "output": 0.006},
        "gpt-4-turbo": {"input": 0.001, "output": 0.003},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "gemini-1.5-pro": {"input": 0.0035, "output": 0.0105},
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
        "gemini-1.0-pro": {"input": 0.0005, "output": 0.0015},
        "llama3-8b-8192": {"input": 0.00005, "output": 0.00008},
        "llama3-70b-8192": {"input": 0.00059, "output": 0.00079},
        "mixtral-8x7b-32768": {"input": 0.00024, "output": 0.00024},
    }
}


class PricingCalculator:
    def __init__(self):
        self.pricing_data = self._load_pricing_data()

    def _load_pricing_data(self) -> dict[str, Any]:
        return DEFAULT_PRICING_DATA

    def _normalize_model_name(self, model_name: str) -> str:
        if not model_name:
            return "gpt-3.5-turbo"

        model_lower = model_name.lower().strip()

        if model_lower in ["gpt-4o", "gpt-4o-2024-05-13"]:
            return "gpt-4o"
        elif model_lower in ["gpt-4o-mini", "gpt-4o-mini-2024-07-18"]:
            return "gpt-4o-mini"
        elif model_lower in ["gpt-4", "gpt-4-0613", "gpt-4-0314"]:
            return "gpt-4"
        elif model_lower in ["gpt-4-turbo", "gpt-4-turbo-2024-04-09"]:
            return "gpt-4-turbo"
        elif model_lower in ["gpt-3.5-turbo", "gpt-3.5-turbo-0125"]:
            return "gpt-3.5-turbo"

        elif "claude-3.5-sonnet" in model_lower or "claude-3-5-sonnet" in model_lower:
            return "claude-3-5-sonnet-20241022"
        elif "claude-3-sonnet" in model_lower:
            return "claude-3-sonnet-20240229"
        elif "claude-3-haiku" in model_lower:
            return "claude-3-haiku-20240307"

        elif "gemini-1.5-pro" in model_lower:
            return "gemini-1.5-pro"
        elif "gemini-1.5-flash" in model_lower:
            return "gemini-1.5-flash"
        elif "gemini-pro" in model_lower or "gemini-1.0-pro" in model_lower:
            return "gemini-1.0-pro"

        elif "llama3-8b" in model_lower or "llama-3-8b" in model_lower:
            return "llama3-8b-8192"
        elif "llama3-70b" in model_lower or "llama-3-70b" in model_lower:
            return "llama3-70b-8192"

        elif "mixtral-8x7b" in model_lower:
            return "mixtral-8x7b-32768"

        elif "lm_studio" in model_lower or "lm-studio" in model_lower:
            if "llama" in model_lower and ("8b" in model_lower or "7b" in model_lower):
                return "llama3-8b-8192"
            elif "mixtral" in model_lower:
                return "mixtral-8x7b-32768"
            else:
                return "llama3-8b-8192"

        return "gpt-3.5-turbo"

    def get_model_pricing(self, model_name: str) -> tuple[float, float]:
        normalized_name = self._normalize_model_name(model_name)

        if normalized_name in self.pricing_data["chat"]:
            pricing = self.pricing_data["chat"][normalized_name]
            return pricing["input"], pricing["output"]
        else:
            default_pricing = self.pricing_data["chat"]["gpt-3.5-turbo"]
            return default_pricing["input"], default_pricing["output"]

    def calculate_cost(
        self, model_name: str, prompt_tokens: int, completion_tokens: int
    ) -> dict[str, Any]:
        prompt_tokens = max(0, prompt_tokens)
        completion_tokens = max(0, completion_tokens)

        input_price_per_1k, output_price_per_1k = self.get_model_pricing(model_name)

        prompt_cost = (prompt_tokens / 1000.0) * input_price_per_1k
        completion_cost = (completion_tokens / 1000.0) * output_price_per_1k
        total_cost = prompt_cost + completion_cost

        return {
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": total_cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "model": model_name,
            "prompt_price_per_1k": input_price_per_1k,
            "completion_price_per_1k": output_price_per_1k,
        }

    def get_supported_models(self) -> dict[str, dict[str, float]]:
        return self.pricing_data["chat"]


_pricing_calculator = None


def get_pricing_calculator() -> PricingCalculator:
    global _pricing_calculator
    if _pricing_calculator is None:
        _pricing_calculator = PricingCalculator()
    return _pricing_calculator


def calculate_request_cost(
    model_name: str, prompt_tokens: int, completion_tokens: int
) -> dict[str, Any]:
    calculator = get_pricing_calculator()
    return calculator.calculate_cost(model_name, prompt_tokens, completion_tokens)



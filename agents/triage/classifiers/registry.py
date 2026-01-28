from __future__ import annotations

import importlib
import time
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from agents.triage.classifiers.base import (
    BaseClassifier,
    ClassificationResult,
    Classifier,
    ClassifierInfo,
)
from agents.triage.models import ClassifierMetrics

_CLASSIFIER_REGISTRY: dict[str, tuple[str, str, ClassifierInfo]] = {
    "v0": (
        "agents.triage.classifiers.v0",
        "classifier_v0",
        ClassifierInfo(
            version="v0",
            name="Minimal Prompt",
            description="Basic DSPy classifier with minimal prompt engineering.",
            requires_llm=True,
            requires_training=False,
            trainable=False,
            estimated_latency_ms=(300.0, 800.0),
            estimated_cost_per_request=0.001,
        ),
    ),
    "v1": (
        "agents.triage.classifiers.v1",
        "classifier_v1",
        ClassifierInfo(
            version="v1",
            name="Enhanced Prompt",
            description="DSPy classifier with improved prompt and agent descriptions.",
            requires_llm=True,
            requires_training=False,
            trainable=False,
            estimated_latency_ms=(300.0, 800.0),
            estimated_cost_per_request=0.002,
        ),
    ),
    "v2": (
        "agents.triage.classifiers.v2.classifier_v2",
        "classifier_v2",
        ClassifierInfo(
            version="v2",
            name="COPRO Optimized",
            description="DSPy classifier with COPRO-optimized few-shot examples.",
            requires_llm=True,
            requires_training=False,
            trainable=True,
            estimated_latency_ms=(300.0, 700.0),
            estimated_cost_per_request=0.001,
        ),
    ),
    "v3": (
        "agents.triage.classifiers.v3.classifier_v3",
        "classifier_v3",
        ClassifierInfo(
            version="v3",
            name="Few-Shot MLflow",
            description="Few-shot classifier with MLflow model registry integration.",
            requires_llm=False,
            requires_training=True,
            trainable=True,
            estimated_latency_ms=(30.0, 100.0),
            estimated_cost_per_request=0.0,
        ),
    ),
    "v4": (
        "agents.triage.classifiers.v4.classifier_v4",
        "classifier_v4",
        ClassifierInfo(
            version="v4",
            name="Zero-Shot COPRO",
            description="Zero-shot DSPy classifier with COPRO-optimized instructions. Recommended default.",
            requires_llm=True,
            requires_training=False,
            trainable=True,
            estimated_latency_ms=(200.0, 600.0),
            estimated_cost_per_request=0.001,
        ),
    ),
    "v5": (
        "agents.triage.classifiers.v5.classifier_v5",
        "classifier_v5",
        ClassifierInfo(
            version="v5",
            name="Attention Network",
            description="PyTorch attention-based neural classifier. Fast inference, requires training.",
            requires_llm=False,
            requires_training=True,
            trainable=True,
            estimated_latency_ms=(10.0, 50.0),
            estimated_cost_per_request=0.0,
        ),
    ),
    "v6": (
        "agents.triage.classifiers.v6.classifier_v6",
        "classifier_v6",
        ClassifierInfo(
            version="v6",
            name="OpenAI Fine-tuned",
            description="Fine-tuned GPT model via OpenAI API. High accuracy, requires API key.",
            requires_llm=True,
            requires_training=True,
            trainable=True,
            estimated_latency_ms=(200.0, 500.0),
            estimated_cost_per_request=0.01,
        ),
    ),
    "v7": (
        "agents.triage.classifiers.v7.classifier_v7",
        "classifier_v7",
        ClassifierInfo(
            version="v7",
            name="Gemma Fine-tuned",
            description="Fine-tuned Gemma model via HuggingFace. Local inference, requires training.",
            requires_llm=False,
            requires_training=True,
            trainable=True,
            estimated_latency_ms=(50.0, 200.0),
            estimated_cost_per_request=0.0,
        ),
    ),
    "v8": (
        "agents.triage.classifiers.v8.classifier_v8",
        "classifier_v8",
        ClassifierInfo(
            version="v8",
            name="RoBERTa LoRA",
            description="RoBERTa with LoRA fine-tuning. Memory efficient, local GPU training.",
            requires_llm=False,
            requires_training=True,
            trainable=True,
            estimated_latency_ms=(20.0, 80.0),
            estimated_cost_per_request=0.0,
        ),
    ),
}


class WrappedClassifier(BaseClassifier):
    def __init__(
        self,
        version: str,
        classifier_fn: Callable[..., Any],
        classifier_info: ClassifierInfo,
    ):
        self._version = version
        self._classifier_fn = classifier_fn
        self._info = classifier_info

    @property
    def info(self) -> ClassifierInfo:
        return self._info

    def classify(self, chat_history: str) -> ClassificationResult:
        start_time = time.perf_counter()
        result = self._classifier_fn(chat_history=chat_history)
        latency_ms = (time.perf_counter() - start_time) * 1000
        metrics: ClassifierMetrics | None = getattr(result, "metrics", None)

        if metrics:
            return ClassificationResult(
                target_agent=result.target_agent,
                latency_ms=metrics.latency_ms if metrics.latency_ms > 0 else latency_ms,
                confidence=metrics.confidence,
                total_tokens=metrics.total_tokens,
                prompt_tokens=metrics.prompt_tokens,
                completion_tokens=metrics.completion_tokens,
            )
        else:
            return ClassificationResult(
                target_agent=result.target_agent,
                latency_ms=latency_ms,
            )

    def is_ready(self) -> bool:
        return True


@lru_cache(maxsize=16)
def _load_classifier_fn(version: str) -> tuple[Callable[..., Any], ClassifierInfo]:
    if version not in _CLASSIFIER_REGISTRY:
        available = ", ".join(sorted(_CLASSIFIER_REGISTRY.keys()))
        raise ValueError(f"Unknown classifier version: {version}. Available: {available}")

    module_path, fn_name, info = _CLASSIFIER_REGISTRY[version]

    try:
        module = importlib.import_module(module_path)
        classifier_fn = getattr(module, fn_name)
        return classifier_fn, info
    except ImportError as e:
        raise ImportError(
            f"Failed to import classifier {version} from {module_path}: {e}"
        ) from e
    except AttributeError as e:
        raise AttributeError(
            f"Classifier function '{fn_name}' not found in {module_path}: {e}"
        ) from e


def get_classifier(version: str) -> Classifier:
    classifier_fn, info = _load_classifier_fn(version)
    return WrappedClassifier(version, classifier_fn, info)


def list_classifiers() -> list[ClassifierInfo]:
    return [info for _, _, info in _CLASSIFIER_REGISTRY.values()]


def compare_classifiers(
    versions: list[str] | None = None,
) -> dict[str, ClassifierInfo]:
    versions = versions or list(_CLASSIFIER_REGISTRY.keys())
    return {v: _CLASSIFIER_REGISTRY[v][2] for v in versions if v in _CLASSIFIER_REGISTRY}


def get_available_versions() -> list[str]:
    return sorted(_CLASSIFIER_REGISTRY.keys())

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agents.triage.models import TargetAgent


@dataclass(frozen=True)
class ClassificationResult:
    target_agent: TargetAgent
    latency_ms: float
    confidence: float | None = None
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class ClassifierInfo:
    version: str
    name: str
    description: str
    requires_llm: bool = True
    requires_training: bool = False
    trainable: bool = False
    estimated_latency_ms: tuple[float, float] = field(default=(100.0, 1000.0))
    estimated_cost_per_request: float = 0.0

    def __str__(self) -> str:
        return f"{self.version}: {self.name}"


@runtime_checkable
class Classifier(Protocol):
    @property
    def info(self) -> ClassifierInfo: ...

    def classify(self, chat_history: str) -> ClassificationResult: ...

    def is_ready(self) -> bool: ...


class BaseClassifier(ABC):
    _instances: dict[type, BaseClassifier] = {}

    @classmethod
    def get_instance(cls) -> BaseClassifier:
        if cls not in cls._instances:
            cls._instances[cls] = cls()
        return cls._instances[cls]

    @property
    @abstractmethod
    def info(self) -> ClassifierInfo:
        pass

    @abstractmethod
    def classify(self, chat_history: str) -> ClassificationResult:
        pass

    def is_ready(self) -> bool:
        return True

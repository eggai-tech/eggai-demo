"""
Base classes and protocols for the unified classifier interface.

All classifiers implement the Classifier protocol, providing a consistent
interface for classification, metadata access, and readiness checking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agents.triage.models import TargetAgent


@dataclass(frozen=True)
class ClassificationResult:
    """
    Immutable result from any classifier.

    Attributes:
        target_agent: The classified target agent to route to.
        latency_ms: Time taken for classification in milliseconds.
        confidence: Optional confidence score (0.0-1.0) if supported.
        total_tokens: Total tokens used (for LLM-based classifiers).
        prompt_tokens: Prompt tokens used (for LLM-based classifiers).
        completion_tokens: Completion tokens used (for LLM-based classifiers).
    """

    target_agent: TargetAgent
    latency_ms: float
    confidence: float | None = None
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class ClassifierInfo:
    """
    Metadata describing a classifier's characteristics.

    Used for documentation, comparison, and selection guidance.
    """

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
    """
    Protocol that all classifiers must implement.

    This provides a unified interface for:
    - Getting classifier metadata
    - Performing classification
    - Checking if the classifier is ready

    Example:
        def use_classifier(c: Classifier) -> None:
            if c.is_ready():
                result = c.classify("User: Hello")
                print(f"Routed to: {result.target_agent}")
    """

    @property
    def info(self) -> ClassifierInfo:
        """Return classifier metadata."""
        ...

    def classify(self, chat_history: str) -> ClassificationResult:
        """
        Classify the chat history and return a result.

        Args:
            chat_history: The conversation history to classify.

        Returns:
            ClassificationResult with target_agent and metrics.
        """
        ...

    def is_ready(self) -> bool:
        """
        Check if the classifier is ready to serve requests.

        Returns:
            True if the classifier can accept classification requests.
        """
        ...


class BaseClassifier(ABC):
    """
    Abstract base class with shared functionality for classifiers.

    Provides:
    - Singleton pattern for lazy loading expensive resources
    - Default is_ready() implementation
    - Abstract methods that must be implemented

    Subclasses should implement:
    - info property
    - classify() method
    - Optionally override is_ready() for lazy-loaded models
    """

    _instances: dict[type, BaseClassifier] = {}

    @classmethod
    def get_instance(cls) -> BaseClassifier:
        """
        Get or create singleton instance with lazy loading.

        Returns:
            The singleton instance of this classifier.
        """
        if cls not in cls._instances:
            cls._instances[cls] = cls()
        return cls._instances[cls]

    @property
    @abstractmethod
    def info(self) -> ClassifierInfo:
        """Return classifier metadata."""
        pass

    @abstractmethod
    def classify(self, chat_history: str) -> ClassificationResult:
        """Classify the chat history and return a result."""
        pass

    def is_ready(self) -> bool:
        """
        Default implementation - override for lazy-loaded models.

        Returns:
            True by default. Override to check model loading state.
        """
        return True

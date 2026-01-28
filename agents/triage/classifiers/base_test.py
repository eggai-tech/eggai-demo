import pytest

from agents.triage.classifiers.base import (
    ClassificationResult,
    ClassifierInfo,
)
from agents.triage.models import TargetAgent


class TestClassificationResult:

    def test_create_minimal(self):
        result = ClassificationResult(
            target_agent=TargetAgent.BillingAgent,
            latency_ms=100.0,
        )
        assert result.target_agent == TargetAgent.BillingAgent
        assert result.latency_ms == 100.0
        assert result.confidence is None
        assert result.total_tokens == 0

    def test_create_full(self):
        result = ClassificationResult(
            target_agent=TargetAgent.PolicyAgent,
            latency_ms=250.5,
            confidence=0.95,
            total_tokens=150,
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert result.target_agent == TargetAgent.PolicyAgent
        assert result.latency_ms == 250.5
        assert result.confidence == 0.95
        assert result.total_tokens == 150
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50

    def test_immutable(self):
        result = ClassificationResult(
            target_agent=TargetAgent.ClaimsAgent,
            latency_ms=100.0,
        )
        with pytest.raises(AttributeError):
            result.target_agent = TargetAgent.BillingAgent


class TestClassifierInfo:

    def test_create_minimal(self):
        info = ClassifierInfo(
            version="v0",
            name="Test Classifier",
            description="A test classifier.",
        )
        assert info.version == "v0"
        assert info.name == "Test Classifier"
        assert info.requires_llm is True  # default
        assert info.requires_training is False  # default

    def test_str_representation(self):
        info = ClassifierInfo(
            version="v4",
            name="Zero-Shot COPRO",
            description="Test",
        )
        assert str(info) == "v4: Zero-Shot COPRO"

    def test_immutable(self):
        info = ClassifierInfo(
            version="v1",
            name="Test",
            description="Test",
        )
        with pytest.raises(AttributeError):
            info.version = "v2"

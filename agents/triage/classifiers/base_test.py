"""Tests for the unified classifier interface."""

import pytest

from agents.triage.classifiers.base import (
    ClassificationResult,
    ClassifierInfo,
)
from agents.triage.models import TargetAgent


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_create_minimal(self):
        """Test creating result with minimal fields."""
        result = ClassificationResult(
            target_agent=TargetAgent.BillingAgent,
            latency_ms=100.0,
        )
        assert result.target_agent == TargetAgent.BillingAgent
        assert result.latency_ms == 100.0
        assert result.confidence is None
        assert result.total_tokens == 0

    def test_create_full(self):
        """Test creating result with all fields."""
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
        """Test that result is immutable."""
        result = ClassificationResult(
            target_agent=TargetAgent.ClaimsAgent,
            latency_ms=100.0,
        )
        with pytest.raises(AttributeError):
            result.target_agent = TargetAgent.BillingAgent


class TestClassifierInfo:
    """Tests for ClassifierInfo dataclass."""

    def test_create_minimal(self):
        """Test creating info with minimal fields."""
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
        """Test string representation."""
        info = ClassifierInfo(
            version="v4",
            name="Zero-Shot COPRO",
            description="Test",
        )
        assert str(info) == "v4: Zero-Shot COPRO"

    def test_immutable(self):
        """Test that info is immutable."""
        info = ClassifierInfo(
            version="v1",
            name="Test",
            description="Test",
        )
        with pytest.raises(AttributeError):
            info.version = "v2"

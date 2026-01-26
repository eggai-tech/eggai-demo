"""
Unified classifier interface for the Triage Agent.

This module provides a common interface for all 8 classifier implementations,
making it easy to compare, evaluate, and switch between them.

Usage:
    from agents.triage.classifiers import get_classifier, list_classifiers

    # Get a specific classifier
    classifier = get_classifier("v4")
    result = classifier.classify("User: What's my bill?")
    print(result.target_agent)  # BillingAgent

    # List all available classifiers
    for info in list_classifiers():
        print(f"{info.version}: {info.name}")

    # Compare classifiers
    from agents.triage.classifiers import compare_classifiers
    comparison = compare_classifiers(["v0", "v4", "v6"])
"""

from agents.triage.classifiers.base import (
    BaseClassifier,
    ClassificationResult,
    Classifier,
    ClassifierInfo,
)
from agents.triage.classifiers.registry import (
    compare_classifiers,
    get_available_versions,
    get_classifier,
    list_classifiers,
)

__all__ = [
    # Protocol and types
    "Classifier",
    "BaseClassifier",
    "ClassificationResult",
    "ClassifierInfo",
    # Registry functions
    "get_classifier",
    "get_available_versions",
    "list_classifiers",
    "compare_classifiers",
]

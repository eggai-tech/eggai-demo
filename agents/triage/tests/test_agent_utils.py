import pytest

from agents.triage.agent import (
    build_conversation_string,
    get_current_classifier,
)
from agents.triage.classifiers import get_available_versions
from agents.triage.config import settings


def test_build_conversation_string_empty():
    # No messages yields empty string
    assert build_conversation_string([]) == ""


def test_build_conversation_string_basic():
    msgs = [
        {"agent": "User", "content": "Hello"},
        {"agent": "Bot", "content": "Hi there!"},
    ]
    expected = "User: Hello\nBot: Hi there!\n"
    assert build_conversation_string(msgs) == expected


def test_build_conversation_string_filters_empty_content():
    msgs = [
        {"agent": "User", "content": ""},
        {"agent": "Bot", "content": "Response"},
    ]
    expected = "Bot: Response\n"
    assert build_conversation_string(msgs) == expected


@pytest.mark.parametrize("version", get_available_versions())
def test_get_current_classifier_valid(monkeypatch, version):
    # Ensure get_current_classifier returns a callable for each supported version
    monkeypatch.setattr(settings, "classifier_version", version)
    classifier = get_current_classifier()
    # The registry returns a Classifier instance with a classify method
    assert hasattr(classifier, "classify")
    assert callable(classifier.classify)


def test_get_current_classifier_invalid(monkeypatch):
    # Unsupported version should raise ValueError
    monkeypatch.setattr(settings, "classifier_version", "nonexistent")
    with pytest.raises(ValueError):
        get_current_classifier()

import pytest

from agents.claims.dspy_modules.claims import (
    ClaimsSignature,
    ModelConfig,
    process_claims,
    truncate_long_history,
)
from libraries.testing.utils.dspy_helpers import (
    test_model_config_validation as shared_model_config,
)
from libraries.testing.utils.dspy_helpers import (
    test_optimized_dspy_basic as shared_dspy_basic,
)
from libraries.testing.utils.dspy_helpers import (
    test_optimized_dspy_empty_conversation as shared_dspy_empty,
)
from libraries.testing.utils.dspy_helpers import (
    test_signature_fields as shared_signature_fields,
)
from libraries.testing.utils.dspy_helpers import (
    test_signature_structure as shared_signature_structure,
)
from libraries.testing.utils.dspy_helpers import (
    test_truncate_long_history_edge_cases as shared_truncate_edge_cases,
)
from libraries.testing.utils.dspy_helpers import (
    test_truncate_long_history_return_structure as shared_truncate_structure,
)
from libraries.testing.utils.dspy_helpers import (
    test_truncate_long_history_with_config as shared_truncate_config,
)


def test_truncate_long_history_edge_cases():
    shared_truncate_edge_cases(truncate_long_history, ModelConfig)


def test_claims_signature():
    shared_signature_structure(ClaimsSignature)


def test_claims_signature_fields():
    shared_signature_fields(ClaimsSignature)


@pytest.mark.asyncio
async def test_claims_optimized_dspy_basic():
    conversation = "User: I need to file a claim for my car accident\nClaimsAgent: I can help with that."
    expected_response = "I'll help you file your claim. Please provide your policy number and details about the incident."
    await shared_dspy_basic(process_claims, conversation, expected_response)


@pytest.mark.asyncio
async def test_claims_optimized_dspy_empty_conversation():
    expected_response = "I need more information about your claim to help you."
    await shared_dspy_empty(process_claims, expected_response)


def test_model_config_validation():
    shared_model_config(ModelConfig)


def test_truncate_long_history_with_config():
    shared_truncate_config(truncate_long_history, ModelConfig, "claim")


def test_truncate_long_history_return_structure():
    shared_truncate_structure(truncate_long_history)

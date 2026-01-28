import pytest

from agents.billing.dspy_modules.billing import (
    BillingSignature,
    ModelConfig,
    process_billing,
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


def test_billing_signature():
    shared_signature_structure(BillingSignature)


def test_billing_signature_fields():
    shared_signature_fields(BillingSignature)


@pytest.mark.asyncio
async def test_process_billing_basic():
    conversation = (
        "User: What's my current bill?\nBillingAgent: Let me check that for you."
    )
    expected_response = (
        "Your current balance is $125.50. Your next payment is due on March 15th."
    )
    await shared_dspy_basic(process_billing, conversation, expected_response)


@pytest.mark.asyncio
async def test_process_billing_empty_conversation():
    expected_response = "I need more information about your account to help you."
    await shared_dspy_empty(process_billing, expected_response)


def test_model_config_validation():
    shared_model_config(ModelConfig)


def test_truncate_long_history_with_config():
    shared_truncate_config(truncate_long_history, ModelConfig, "billing")


def test_truncate_long_history_return_structure():
    shared_truncate_structure(truncate_long_history)

import json

import pytest

from agents.billing.dspy_modules.billing_data import (
    BILLING_DATABASE,
    get_billing_info,
    get_policy_record,
    update_billing_info,
)


def test_get_policy_record_existing():
    record = get_policy_record("A12345")
    assert record is not None
    assert record["policy_number"] == "A12345"
    assert record["billing_cycle"] == "Monthly"
    assert record["amount_due"] == pytest.approx(120.0)


def test_get_policy_record_nonexistent():
    record = get_policy_record("NONEXISTENT")
    assert record is None


def test_get_policy_record_with_whitespace():
    record = get_policy_record("  A12345  ")
    assert record is not None
    assert record["policy_number"] == "A12345"


def test_get_billing_info_existing_policy():
    result = get_billing_info("B67890")
    data = json.loads(result)

    assert data["policy_number"] == "B67890"
    assert data["billing_cycle"] == "Quarterly"
    assert data["amount_due"] == pytest.approx(300.0)
    assert data["due_date"] == "2026-03-15"
    assert data["status"] == "Pending"


def test_get_billing_info_nonexistent_policy():
    result = get_billing_info("NONEXISTENT")
    data = json.loads(result)

    assert "error" in data
    assert data["error"] == "Policy not found."


def test_update_billing_info_valid_field():
    # Save original to restore after test
    original_record = get_policy_record("C24680")
    original_status = original_record["status"]

    # Update the status
    result = update_billing_info("C24680", "status", "Paid")
    data = json.loads(result)

    assert data["policy_number"] == "C24680"
    assert data["status"] == "Paid"

    updated_record = get_policy_record("C24680")
    assert updated_record["status"] == "Paid"

    update_billing_info("C24680", "status", original_status)


def test_update_billing_info_amount_due():
    original_record = get_policy_record("A12345")
    original_amount = original_record["amount_due"]

    result = update_billing_info("A12345", "amount_due", "150.50")
    data = json.loads(result)

    assert data["amount_due"] == pytest.approx(150.50)

    update_billing_info("A12345", "amount_due", str(original_amount))


def test_update_billing_info_invalid_amount():
    result = update_billing_info("A12345", "amount_due", "invalid_number")
    data = json.loads(result)

    assert "error" in data
    assert "Invalid numeric value" in data["error"]


def test_update_billing_info_nonexistent_policy():
    result = update_billing_info("NONEXISTENT", "status", "Paid")
    data = json.loads(result)

    assert "error" in data
    assert data["error"] == "Policy not found."


def test_update_billing_info_invalid_field():
    result = update_billing_info("A12345", "invalid_field", "some_value")
    data = json.loads(result)

    assert "error" in data
    assert "Field 'invalid_field' not found" in data["error"]


def test_billing_database_structure():
    assert len(BILLING_DATABASE) >= 3

    for record in BILLING_DATABASE:
        assert "policy_number" in record
        assert "billing_cycle" in record
        assert "amount_due" in record
        assert "due_date" in record
        assert "status" in record

        assert isinstance(record["policy_number"], str)
        assert isinstance(record["billing_cycle"], str)
        assert isinstance(record["amount_due"], (int, float))
        assert isinstance(record["due_date"], str)
        assert isinstance(record["status"], str)


def test_update_billing_info_string_field():
    original_record = get_policy_record("B67890")
    original_cycle = original_record["billing_cycle"]

    result = update_billing_info("B67890", "billing_cycle", "Monthly")
    data = json.loads(result)

    assert data["billing_cycle"] == "Monthly"

    update_billing_info("B67890", "billing_cycle", original_cycle)


def test_update_billing_info_due_date():
    original_record = get_policy_record("C24680")
    original_date = original_record["due_date"]

    new_date = "2026-06-01"
    result = update_billing_info("C24680", "due_date", new_date)
    data = json.loads(result)

    assert data["due_date"] == new_date

    update_billing_info("C24680", "due_date", original_date)

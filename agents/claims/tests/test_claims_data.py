import json

import pytest

from agents.claims.dspy_modules.claims_data import (
    CLAIMS_DATABASE,
    ClaimDataException,
    ErrorCategory,
    FieldValidators,
    file_claim,
    get_claim_status,
    update_claim_info,
)


def test_get_claim_status_existing():
    result = get_claim_status("1001")
    data = json.loads(result)

    assert data["claim_number"] == "1001"
    assert data["status"] == "In Review"
    assert data["estimate"] == pytest.approx(2300.0)
    assert data["estimate_date"] == "2026-05-15"
    assert "Repair estimates" in data["outstanding_items"]


def test_get_claim_status_nonexistent():
    result = get_claim_status("NONEXISTENT")
    data = json.loads(result)

    assert "error" in data
    assert "I couldn't find a claim with that number" in data["error"]


def test_get_claim_status_empty():
    result = get_claim_status("")
    data = json.loads(result)

    assert "error" in data


def test_file_claim_success():
    result = file_claim("D99999", "Test incident for filing")
    data = json.loads(result)

    assert "claim_number" in data
    assert data["policy_number"] == "D99999"
    assert data["status"] == "Filed"
    assert "Test incident for filing" in data["details"]


def test_file_claim_missing_policy():
    result = file_claim("", "Test incident")
    data = json.loads(result)

    assert "error" in data
    assert "policy number" in data["error"].lower()


def test_file_claim_missing_description():
    result = file_claim("D99999", "")
    data = json.loads(result)

    assert "error" in data
    assert "claim details" in data["error"].lower()


def test_file_claim_missing_contact():
    # Since file_claim only takes 2 params, this test checks empty details
    result = file_claim("D99999", "")
    data = json.loads(result)

    assert "error" in data
    assert "claim details" in data["error"].lower()


def test_update_claim_info_valid_field():
    # Update next_steps for claim 1003
    result = update_claim_info("1003", "next_steps", "Contact adjuster for inspection")
    data = json.loads(result)

    assert data["claim_number"] == "1003"
    assert data["next_steps"] == "Contact adjuster for inspection"


def test_update_claim_info_nonexistent_claim():
    result = update_claim_info("NONEXISTENT", "next_steps", "Some value")
    data = json.loads(result)

    assert "error" in data
    assert "I couldn't find a claim with that number" in data["error"]


def test_update_claim_info_invalid_field():
    result = update_claim_info("1001", "invalid_field", "Some value")
    data = json.loads(result)

    assert "error" in data
    assert (
        "security" in data["error"].lower()
        or "cannot be updated" in data["error"].lower()
    )


def test_update_claim_info_estimate():
    result = update_claim_info("1003", "estimate", "2500.00")
    data = json.loads(result)

    assert data["estimate"] == pytest.approx(2500.0)


def test_update_claim_info_invalid_estimate():
    result = update_claim_info("1001", "estimate", "invalid_amount")
    data = json.loads(result)

    assert "error" in data
    assert (
        "security" in data["error"].lower()
        or "cannot be updated" in data["error"].lower()
    )


def test_update_claim_info_negative_estimate():
    result = update_claim_info("1001", "estimate", "-100")
    data = json.loads(result)

    assert "error" in data
    assert (
        "security" in data["error"].lower()
        or "cannot be updated" in data["error"].lower()
    )


def test_update_claim_info_date_field():
    result = update_claim_info("1003", "estimate_date", "2026-06-01")
    data = json.loads(result)

    assert data["estimate_date"] == "2026-06-01"


def test_update_claim_info_invalid_date():
    result = update_claim_info("1001", "estimate_date", "invalid_date")
    data = json.loads(result)

    assert "error" in data
    assert (
        "security" in data["error"].lower()
        or "cannot be updated" in data["error"].lower()
    )


def test_update_claim_info_outstanding_items():
    result = update_claim_info(
        "1001", "outstanding_items", "Photos, Police report, Medical records"
    )
    data = json.loads(result)

    assert "Photos" in data["outstanding_items"]
    assert "Police report" in data["outstanding_items"]
    assert "Medical records" in data["outstanding_items"]


def test_update_claim_info_empty_field():
    result = update_claim_info("1001", "", "Some value")
    data = json.loads(result)

    assert "error" in data


def test_update_claim_info_empty_value():
    result = update_claim_info("1001", "next_steps", "")
    data = json.loads(result)

    assert "error" in data


def test_claim_data_exception():
    exception = ClaimDataException("Test error", ErrorCategory.USER_INPUT)

    assert str(exception) == "Test error"
    assert exception.message == "Test error"
    assert exception.category == ErrorCategory.USER_INPUT


def test_field_validators_estimate():
    # Valid estimate
    valid, result = FieldValidators.validate_estimate("1500.50")
    assert valid is True
    assert result == pytest.approx(1500.50)

    # Invalid estimate
    valid, result = FieldValidators.validate_estimate("invalid")
    assert valid is False
    assert "valid number" in result

    # Negative estimate
    valid, result = FieldValidators.validate_estimate("-100")
    assert valid is False
    assert "negative" in result


def test_field_validators_date():
    # Valid date
    valid, result = FieldValidators.validate_date("2026-05-15")
    assert valid is True
    assert result == "2026-05-15"

    # Invalid date format
    valid, result = FieldValidators.validate_date("invalid_date")
    assert valid is False
    assert "YYYY-MM-DD" in result


def test_field_validators_items_list():
    # Valid list
    valid, result = FieldValidators.validate_items_list(
        "Photos, Police report, Medical records"
    )
    assert valid is True
    assert "Photos" in result
    assert "Police report" in result
    assert "Medical records" in result

    # Empty list
    valid, result = FieldValidators.validate_items_list("")
    assert valid is False
    assert "at least one item" in result


def test_field_validators_text():
    # Valid text
    valid, result = FieldValidators.validate_text("Valid text content")
    assert valid is True
    assert result == "Valid text content"

    # Empty text
    valid, result = FieldValidators.validate_text("")
    assert valid is False
    assert "empty" in result


def test_error_category_enum():
    assert ErrorCategory.USER_INPUT.value == "user_input"
    assert ErrorCategory.DATA_ACCESS.value == "data_access"
    assert ErrorCategory.SYSTEM.value == "system"
    assert ErrorCategory.MODEL.value == "model"
    assert ErrorCategory.SECURITY.value == "security"


def test_claims_database_structure():
    assert len(CLAIMS_DATABASE) >= 3

    for claim in CLAIMS_DATABASE:
        assert hasattr(claim, "claim_number")
        assert hasattr(claim, "policy_number")
        assert hasattr(claim, "status")
        assert hasattr(claim, "next_steps")
        assert hasattr(claim, "outstanding_items")


def test_claim_record_to_json():
    claim = CLAIMS_DATABASE[0]
    json_str = claim.to_json()
    data = json.loads(json_str)

    assert data["claim_number"] == claim.claim_number
    assert data["policy_number"] == claim.policy_number
    assert data["status"] == claim.status

import pytest

from agents.billing.dspy_modules.evaluation.metrics import (
    get_amount_score,
    get_billing_cycle_score,
    get_date_score,
    get_format_score,
    get_status_score,
    precision_metric,
)


def test_get_amount_score_exact_match():
    expected = "Your amount due is $120.50"
    actual = "Your amount due is $120.50"

    score, performed = get_amount_score(expected, actual)
    assert performed is True
    assert score == pytest.approx(1.0)


def test_get_amount_score_different_format():
    expected = "Your amount due is $120.50"
    actual = "You owe $120.5"

    score, performed = get_amount_score(expected, actual)
    assert performed is True
    assert score == pytest.approx(0.8)


def test_get_amount_score_wrong_amount():
    expected = "Your amount due is $120.50"
    actual = "Your amount due is $150.00"

    score, performed = get_amount_score(expected, actual)
    assert performed is True
    assert score == pytest.approx(0.0)


def test_get_amount_score_no_amount():
    expected = "Your status is active"
    actual = "Your amount due is $120.50"

    score, performed = get_amount_score(expected, actual)
    assert performed is False
    assert score == pytest.approx(0.0)


def test_get_date_score_exact_match():
    expected = "Due date is 2026-05-15"
    actual = "Due date is 2026-05-15"

    score, performed = get_date_score(expected, actual)
    assert performed is True
    assert score == pytest.approx(1.0)


def test_get_date_score_different_format():
    expected = "Due date is 2026-05-15"
    actual = "Due on May 15, 2026"

    score, performed = get_date_score(expected, actual)
    assert performed is True
    assert score == pytest.approx(0.7)


def test_get_date_score_wrong_date():
    expected = "Due date is 2026-05-15"
    actual = "Due date is 2026-06-15"

    score, performed = get_date_score(expected, actual)
    assert performed is True
    assert score == pytest.approx(0.0)


def test_get_date_score_no_date():
    expected = "Your status is active"
    actual = "Due date is 2026-05-15"

    score, performed = get_date_score(expected, actual)
    assert performed is False
    assert score == pytest.approx(0.0)


def test_get_status_score_exact_match():
    expected = "Your status is 'Paid'"
    actual = "Your status is 'Paid'"

    score, performed = get_status_score(expected, actual)
    assert performed is True
    assert score == pytest.approx(1.0)


def test_get_status_score_different_quotes():
    expected = "Your status is 'Paid'"
    actual = 'Your status is "Paid"'

    score, performed = get_status_score(expected, actual)
    assert performed is True
    assert score == pytest.approx(0.9)


def test_get_status_score_no_quotes():
    expected = "Your status is 'Paid'"
    actual = "Your status is Paid"

    score, performed = get_status_score(expected, actual)
    assert performed is True
    assert score == pytest.approx(0.8)


def test_get_status_score_wrong_status():
    expected = "Your status is 'Paid'"
    actual = "Your status is 'Pending'"

    score, performed = get_status_score(expected, actual)
    assert performed is True
    assert score == pytest.approx(0.0)


def test_get_status_score_no_status():
    expected = "Your amount is $120.50"
    actual = "Your status is 'Paid'"

    score, performed = get_status_score(expected, actual)
    assert performed is False
    assert score == pytest.approx(0.0)


def test_get_billing_cycle_score_exact_match():
    expected = "Your billing cycle is 'Monthly'"
    actual = "Your billing cycle is 'Monthly'"

    score, performed = get_billing_cycle_score(expected, actual)
    assert performed is True
    assert score == pytest.approx(1.0)


def test_get_billing_cycle_score_partial_match():
    expected = "Your billing cycle is 'Monthly'"
    actual = "Your billing cycle is different"

    score, performed = get_billing_cycle_score(expected, actual)
    assert performed is True
    assert score == pytest.approx(0.8)


def test_get_billing_cycle_score_cycle_only():
    expected = "Your billing cycle is 'Monthly'"
    actual = "Your cycle type is monthly"

    score, performed = get_billing_cycle_score(expected, actual)
    assert performed is True
    assert score == pytest.approx(0.7)


def test_get_billing_cycle_score_type_only():
    expected = "Your billing cycle is 'Monthly'"
    actual = "You are billed monthly"

    score, performed = get_billing_cycle_score(expected, actual)
    assert performed is True
    assert score == pytest.approx(0.3)


def test_get_billing_cycle_score_no_match():
    expected = "Your billing cycle is 'Monthly'"
    actual = "Your status is active"

    score, performed = get_billing_cycle_score(expected, actual)
    assert performed is True
    assert score == pytest.approx(0.0)


def test_get_billing_cycle_score_no_cycle():
    expected = "Your status is active"
    actual = "Your billing cycle is 'Monthly'"

    score, performed = get_billing_cycle_score(expected, actual)
    assert performed is False
    assert score == pytest.approx(0.0)


def test_get_format_score_perfect_format():
    expected = "Your current amount due is $120.50 with a due date of 2026-05-15. Your status is 'Paid'."
    actual = "Your current amount due is $120.50 with a due date of 2026-05-15. Your status is 'Paid'."

    score, performed = get_format_score(expected, actual)
    assert performed is True
    assert score == pytest.approx(1.0)


def test_get_format_score_partial_format():
    expected = "Your current amount due is $120.50 with a due date of 2026-05-15."
    actual = "Amount due: $120.50, due: 2026-05-15"

    score, performed = get_format_score(expected, actual)
    assert performed is True
    assert score > 0.0
    assert score < 1.0


def test_get_format_score_no_format():
    expected = "Hello there"
    actual = "Your current amount due is $120.50"

    score, performed = get_format_score(expected, actual)
    assert performed is False
    assert score == pytest.approx(0.0)


def test_precision_metric_all_evaluations():
    expected = "Your current amount due is $120.50 with a due date of 2026-05-15. Your status is 'Paid'. Your billing cycle is 'Monthly'."
    actual = "Your current amount due is $120.50 with a due date of 2026-05-15. Your status is 'Paid'. Your billing cycle is 'Monthly'."

    score = precision_metric(expected, actual)
    assert score == pytest.approx(1.0)


def test_precision_metric_partial_match():
    expected = "Your current amount due is $120.50 with a due date of 2026-05-15."
    actual = "Amount: $120.50, due: 2026-06-15"  # Wrong date

    score = precision_metric(expected, actual)
    assert 0.0 < score < 1.0


def test_precision_metric_no_evaluations():
    expected = "Hello there"
    actual = "Hi back"

    score = precision_metric(expected, actual)
    assert score == pytest.approx(0.0)


def test_precision_metric_mixed_results():
    expected = "Your amount due is $120.50 and status is 'Paid'"
    actual = "Your amount due is $150.00 and status is 'Paid'"  # Wrong amount, correct status

    score = precision_metric(expected, actual)
    assert 0.0 < score < 1.0


def test_amount_score_edge_cases():
    # Multiple amounts in text
    expected = "First amount $100.00 and second amount $200.00"
    actual = "First amount $100.00 and second amount $200.00"

    score, performed = get_amount_score(expected, actual)
    assert performed is True
    assert score > 0.0

    # Amount with commas
    expected = "Amount due is $1,234.56"
    actual = "Amount due is $1,234.56"

    score, performed = get_amount_score(expected, actual)
    assert performed is True
    assert score == pytest.approx(1.0)


def test_date_score_edge_cases():
    # Multiple dates
    expected = "Start date 2026-01-01 and end date 2026-12-31"
    actual = "Start date 2026-01-01 and end date 2026-12-31"

    score, performed = get_date_score(expected, actual)
    assert performed is True
    assert score > 0.0

    # Date with different separators
    expected = "Due date is 2026-05-15"
    actual = "Due date is 2026/05/15"

    score, performed = get_date_score(expected, actual)
    assert performed is True
    assert score > 0.0


def test_status_score_variations():
    statuses = ["Paid", "Pending", "Overdue", "Active", "Inactive"]

    for status in statuses:
        expected = f"Your status is '{status}'"
        actual = f"Your status is '{status}'"

        score, performed = get_status_score(expected, actual)
        assert performed is True
        assert score == pytest.approx(1.0)


def test_billing_cycle_variations():
    cycles = ["Monthly", "Quarterly", "Annual", "Bi-monthly", "Weekly"]

    for cycle in cycles:
        expected = f"Your billing cycle is '{cycle}'"
        actual = f"Your billing cycle is '{cycle}'"

        score, performed = get_billing_cycle_score(expected, actual)
        assert performed is True
        assert score == pytest.approx(1.0)

from pathlib import Path

import dspy
from sklearn.model_selection import train_test_split

from agents.billing.config import settings
from libraries.ml.dspy.language_model import dspy_set_language_model
from libraries.observability.logger import get_console_logger
from libraries.observability.tracing import TracedReAct

from .billing import BillingSignature
from .billing_dataset import (
    as_dspy_examples,
    create_billing_dataset,
)

logger = get_console_logger("billing_optimizer_simba")


def get_billing_info(policy_number: str):
    """Retrieve billing information for a policy number."""
    return (
        f'{{"policy_number": "{policy_number}", "name": "John Doe", "status": "Active", '
        f'"amount_due": 500, "amount_due_usd": "$500.00", "due_date": "2025-06-15", '
        f'"billing_cycle": "Monthly", "payment_status": "Current"}}'
    )


def update_billing_info(policy_number: str, update_data: str):
    """Update billing information for a policy number."""
    return f'{{"status": "success", "policy_number": "{policy_number}", "message": "Billing information updated successfully"}}'


def precision_metric(example, pred, trace=None) -> float:
    expected = getattr(example, "final_response", "").lower()
    predicted = getattr(pred, "final_response", "").lower()

    if not expected or not predicted:
        return 0.0

    if "i need your policy number" in expected:
        return 1.0 if "policy number" in predicted else 0.0

    score = 0.0
    total_checks = 0

    policy_numbers = [
        word
        for word in expected.split()
        if len(word) >= 2 and word[0].isalpha() and any(c.isdigit() for c in word)
    ]
    for policy in policy_numbers:
        total_checks += 1
        if policy in predicted:
            score += 1.0

    if "$" in expected:
        total_checks += 1
        try:
            amount_expected = expected.split("$")[1].split(" ")[0].strip().rstrip(".")
            if "$" in predicted and amount_expected in predicted:
                score += 1.0
        except (IndexError, ValueError):
            pass

    if "due on" in expected or "due date" in expected:
        total_checks += 1
        date_formats = ["YYYY-MM-DD", "2025-", "2024-"]
        for date_format in date_formats:
            if date_format in expected and date_format in predicted:
                score += 1.0
                break

    status_keywords = ["active", "pending", "current", "overdue"]
    for keyword in status_keywords:
        if keyword in expected:
            total_checks += 1
            if keyword in predicted:
                score += 1.0

    if total_checks == 0:
        common_words_expected = set(expected.split())
        common_words_predicted = set(predicted.split())
        intersection = len(common_words_expected.intersection(common_words_predicted))

        if intersection >= 0.6 * len(common_words_expected):
            return 1.0
        elif intersection >= 0.3 * len(common_words_expected):
            return 0.5
        return 0.0

    return score / total_checks


def main():
    dspy_set_language_model(settings)
    logger.info("Creating billing dataset...")
    examples = as_dspy_examples(create_billing_dataset())

    train_set, test_set = train_test_split(examples, test_size=0.2, random_state=42)

    max_train, max_test = 5, 3
    train_set = train_set[:max_train] if len(train_set) > max_train else train_set
    test_set = test_set[:max_test] if len(test_set) > max_test else test_set

    logger.info(
        f"Using {len(train_set)} training examples and {len(test_set)} test examples"
    )

    agent = TracedReAct(
        BillingSignature,
        tools=[get_billing_info, update_billing_info],
        name="billing_react",
        tracer=None,
        max_iters=5,
    )

    batch_size = min(len(train_set), 4)
    simba = dspy.SIMBA(
        metric=precision_metric,
        max_steps=3,
        max_demos=2,
        bsize=batch_size,
    )

    logger.info(
        f"Starting SIMBA optimization with minimal parameters (batch size: {batch_size})"
    )
    optimized_agent = simba.compile(agent, trainset=train_set, seed=42)

    output_path = Path(__file__).resolve().parent / "optimized_billing_simba.json"
    logger.info(f"Saving optimized model to {output_path}")
    optimized_agent.save(str(output_path))

    logger.info("Optimization complete!")


if __name__ == "__main__":
    main()

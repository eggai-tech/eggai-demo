from pathlib import Path

import dspy
from sklearn.model_selection import train_test_split

from agents.claims.config import settings
from agents.claims.dspy_modules.claims_dataset import (
    as_dspy_examples,
    create_claims_dataset,
)
from libraries.ml.dspy.language_model import dspy_set_language_model
from libraries.observability.logger import get_console_logger
from libraries.observability.tracing import TracedReAct

logger = get_console_logger("claims_optimizer_simba")

from agents.claims.dspy_modules.claims import ClaimsSignature


# Mock tools returning fixed data for optimization
def check_claim_status(claim_number: str):
    """Check the status of a claim by claim number."""
    return (
        '{"claim_number": "'
        + claim_number
        + '", "status": "In Review", "estimated_payout": 2300, "estimated_payout_usd": "$2,300.00", "estimated_completion_date": "2025-05-15", "missing_documentation": ["repair estimates", "photos"]}'
    )


def file_new_claim(policy_number: str, incident_description: str):
    """File a new claim for a policy."""
    return (
        '{"status": "success", "claim_number": "CL-123456", "policy_number": "'
        + policy_number
        + '", "next_steps": ["Email photos", "Submit police report"], "contact_email": "claims@example.com"}'
    )


def update_claim_details(claim_number: str, update_data: str):
    """Update details for an existing claim."""
    return (
        '{"status": "success", "claim_number": "'
        + claim_number
        + '", "message": "Claim details updated successfully"}'
    )


def precision_metric(example, pred, trace=None) -> float:
    expected = getattr(example, "final_response", "").lower()
    predicted = getattr(pred, "final_response", "").lower()

    if not expected or not predicted:
        return 0.0

    if "need a valid claim number" in expected:
        return 1.0 if "claim number" in predicted else 0.0

    score = 0.0
    total_checks = 0

    if "claim #" in expected or "claim number" in expected:
        total_checks += 1
        if "claim #" in predicted or "claim number" in predicted:
            score += 1.0

    status_keywords = ["in review", "approved", "pending", "processing"]
    for keyword in status_keywords:
        if keyword in expected:
            total_checks += 1
            if keyword in predicted:
                score += 1.0

    if "$" in expected:
        total_checks += 1
        try:
            amount_expected = expected.split("$")[1].split(" ")[0].strip().rstrip(".")
            if "$" in predicted and amount_expected in predicted:
                score += 1.0
        except (IndexError, ValueError):
            pass

    date_formats = ["YYYY-MM-DD", "2025-", "2024-"]
    for date_format in date_formats:
        if date_format in expected:
            total_checks += 1
            if date_format in predicted:
                score += 1.0
                break

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


if __name__ == "__main__":
    dspy_set_language_model(settings)

    logger.info("Creating claims dataset...")
    raw_examples = create_claims_dataset()
    examples = as_dspy_examples(raw_examples)

    logger.info(f"Created {len(examples)} examples, splitting into train/test...")
    train_set, test_set = train_test_split(examples, test_size=0.2, random_state=42)

    max_train = 5
    max_test = 3

    if len(train_set) > max_train:
        logger.info(f"Limiting training set to {max_train} examples")
        train_set = train_set[:max_train]

    if len(test_set) > max_test:
        logger.info(f"Limiting test set to {max_test} examples")
        test_set = test_set[:max_test]

    agent = TracedReAct(
        ClaimsSignature,
        tools=[check_claim_status, file_new_claim, update_claim_details],
        name="claims_react",
        tracer=None,  # No tracing during optimization
        max_iters=5,
    )

    output_path = Path(__file__).resolve().parent / "optimized_claims_simba.json"

    logger.info("Starting SIMBA optimization with minimal parameters...")
    batch_size = min(4, len(train_set))
    logger.info(f"Using batch size of {batch_size} for {len(train_set)} examples")
    simba = dspy.SIMBA(
        metric=precision_metric,
        max_steps=3,
        max_demos=2,
        bsize=batch_size,
    )
    optimized_agent = simba.compile(agent, trainset=train_set, seed=42)
    optimized_agent.save(str(output_path))

    logger.info(f"Optimization complete! Model saved to {output_path}")

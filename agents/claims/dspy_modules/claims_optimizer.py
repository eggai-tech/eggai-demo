import importlib.util
import inspect
from pathlib import Path

import dspy
import litellm

from agents.claims.config import settings
from agents.claims.dspy_modules.claims_dataset import (
    as_dspy_examples,
    create_claims_dataset,
)
from libraries.observability.tracing import TracedReAct

litellm.drop_params = True


def get_claims_signature_prompt() -> str:
    spec = importlib.util.spec_from_file_location(
        "claims", Path(__file__).resolve().parent / "claims.py"
    )
    claims_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(claims_module)
    return inspect.getdoc(claims_module.ClaimsSignature)


class ClaimsSignature(dspy.Signature):
    __doc__ = get_claims_signature_prompt()

    chat_history: str = dspy.InputField(desc="Full conversation context.")
    final_response: str = dspy.OutputField(desc="Claims response to the user.")


def get_claim_status(claim_number: str):
    """Retrieve claim status and details for a given claim_number."""
    return '{"claim_number": "1001", "policy_number": "A12345", "status": "In Review", "estimate": 2300.0, "estimate_date": "2026-05-15", "next_steps": "Submit repair estimates"}'


def file_claim(policy_number: str, claim_details: str):
    """File a new claim under the given policy with provided details."""
    return '{"claim_number": "1004", "policy_number": "A12345", "status": "Filed", "next_steps": "Provide documentation"}'


def update_claim_info(claim_number: str, field: str, new_value: str):
    """Update a given field in the claim record for the specified claim number."""
    return (
        '{"claim_number": "1001", "policy_number": "A12345", "status": "Updated", "field": "'
        + field
        + '", "new_value": "'
        + new_value
        + '"}'
    )


claims_program = TracedReAct(
    ClaimsSignature,
    tools=[get_claim_status, file_claim, update_claim_info],
    name="claims_react_optimizer",
    tracer=None,
    max_iters=5,
)


def precision_metric(example, pred, trace=None) -> float:
    expected = example.final_response.lower()
    predicted = pred.final_response.lower()

    if "claim #" in expected:
        try:
            claim_num_expected = (
                expected.split("claim #")[1].split(" ")[0].strip().replace(".", "")
            )
            if "claim #" not in predicted or claim_num_expected not in predicted:
                return 0.5
        except IndexError:
            pass

    if "currently" in expected and "'" in expected.split("currently")[1]:
        try:
            status_expected = expected.split("currently '")[1].split("'")[0].strip()
            if "currently" not in predicted or status_expected not in predicted:
                return 0.5
        except IndexError:
            pass

    if "payout of $" in expected:
        try:
            amount_expected = expected.split("payout of $")[1].split(" ")[0].strip()
            if "payout of $" not in predicted or amount_expected not in predicted:
                return 0.5
        except IndexError:
            pass

    if "policy" in expected:
        try:
            policy_expected = expected.split("policy")[1].split(".")[0].strip()
            if "policy" not in predicted or policy_expected not in predicted:
                return 0.5
        except IndexError:
            pass

    return 1.0


critical_privacy_text = "NEVER reveal ANY claim information unless the user has provided a specific, valid claim number"


def format_examples(examples, max_examples=3):
    formatted = []
    for i, example in enumerate(examples[:max_examples]):
        if hasattr(example, "chat_history") and hasattr(example, "final_response"):
            formatted.append(
                f"EXAMPLE {i + 1}:\nUser input: {example.chat_history}\nExpected response: {example.final_response}\n"
            )
    return "\n".join(formatted)


if __name__ == "__main__":
    from libraries.ml.dspy.optimizer_runner import OptimizerConfig, run_optimization

    config = OptimizerConfig(
        agent_name="claims",
        experiment_name="claims_agent_optimization",
        program=claims_program,
        metric_fn=precision_metric,
        dataset_creator=create_claims_dataset,
        dataset_converter=as_dspy_examples,
        critical_privacy_text=critical_privacy_text,
        output_filename=str(
            Path(__file__).resolve().parent / "optimized_claims.json"
        ),
        settings=settings,
        emergency_instruction="You are the Claims Agent for an insurance company. Your #1 responsibility is data privacy. NEVER reveal ANY claim information unless the user has provided a specific, valid claim number.",
    )
    run_optimization(config)

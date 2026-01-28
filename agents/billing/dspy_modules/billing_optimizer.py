import importlib.util
import inspect
from pathlib import Path

import dspy
import litellm

from agents.billing.config import settings
from libraries.observability.tracing import TracedReAct

from .billing_dataset import (
    as_dspy_examples,
    create_billing_dataset,
)

litellm.drop_params = True


def get_billing_signature_prompt() -> str:
    spec = importlib.util.spec_from_file_location(
        "billing", Path(__file__).resolve().parent / "billing.py"
    )
    billing_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(billing_module)
    return inspect.getdoc(billing_module.BillingSignature)


class BillingSignature(dspy.Signature):
    __doc__ = get_billing_signature_prompt()

    chat_history: str = dspy.InputField(desc="Full conversation context.")
    final_response: str = dspy.OutputField(desc="Billing response to the user.")


def mock_get_billing_info(policy_number: str):
    """Retrieve billing information for a policy number."""
    return '{"policy_number": "A12345", "billing_cycle": "Monthly", "amount_due": 120.0, "due_date": "2026-02-01", "status": "Paid"}'


def mock_update_billing_info(policy_number: str, field: str, new_value: str):
    """Update billing information for a policy record."""
    return '{"policy_number": "A12345", "billing_cycle": "Monthly", "amount_due": 120.0, "due_date": "2026-02-01", "status": "Updated"}'


billing_program = TracedReAct(
    BillingSignature,
    tools=[mock_get_billing_info, mock_update_billing_info],
    name="billing_react_optimizer",
    tracer=None,
    max_iters=5,
)


def precision_metric(example, pred, trace=None) -> float:
    expected = example.final_response.lower()
    predicted = pred.final_response.lower()

    if "$" in expected:
        amount_expected = expected.split("$")[1].split(" ")[0].strip()
        if "$" not in predicted or amount_expected not in predicted:
            return 0.5

    if "due on" in expected:
        date_expected = expected.split("due on")[1].split(",")[0].strip()
        if "due on" not in predicted or date_expected not in predicted:
            return 0.5

    if "status" in expected:
        status_expected = (
            expected.split("status is")[1].strip().replace("'", "").replace(".", "")
        )
        if "status" not in predicted or status_expected not in predicted:
            return 0.5

    return 1.0


critical_privacy_text = "NEVER reveal ANY billing information unless the user has provided a specific, valid policy number"


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
        agent_name="billing",
        experiment_name="billing_agent_optimization",
        program=billing_program,
        metric_fn=precision_metric,
        dataset_creator=create_billing_dataset,
        dataset_converter=as_dspy_examples,
        critical_privacy_text=critical_privacy_text,
        output_filename=str(
            Path(__file__).resolve().parent / "optimized_billing.json"
        ),
        settings=settings,
        emergency_instruction="You are the Billing Agent for an insurance company. Your #1 responsibility is data privacy. NEVER reveal ANY billing information unless the user has provided a specific, valid policy number.",
    )
    run_optimization(config)

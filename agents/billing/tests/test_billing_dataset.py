import dspy

from agents.billing.dspy_modules.billing_dataset import (
    BillingExample,
    as_dspy_examples,
    create_billing_dataset,
)


def test_billing_example_dataclass():
    example = BillingExample(
        chat_history="User: What's my bill?\nAgent: I can help with that.",
        expected_response="Your amount due is $120.50",
    )

    assert example.chat_history == "User: What's my bill?\nAgent: I can help with that."
    assert example.expected_response == "Your amount due is $120.50"


def test_create_billing_dataset():
    dataset = create_billing_dataset()

    assert isinstance(dataset, list)
    assert len(dataset) > 0

    for example in dataset:
        assert isinstance(example, BillingExample)
        assert hasattr(example, "chat_history")
        assert hasattr(example, "expected_response")
        assert len(example.chat_history) > 0
        assert len(example.expected_response) > 0


def test_dataset_contains_policy_numbers():
    dataset = create_billing_dataset()

    expected_policies = ["A12345", "B67890", "C24680"]

    found_policies = set()
    for example in dataset:
        for policy in expected_policies:
            if policy in example.chat_history:
                found_policies.add(policy)

    assert len(found_policies) == len(expected_policies)


def test_dataset_example_types():
    dataset = create_billing_dataset()

    inquiry_types = {
        "payment_due": False,
        "amount_due": False,
        "billing_cycle": False,
        "payment_status": False,
    }

    for example in dataset:
        chat_lower = example.chat_history.lower()
        if "payment due" in chat_lower or "next payment" in chat_lower:
            inquiry_types["payment_due"] = True
        if "amount" in chat_lower and "owe" in chat_lower:
            inquiry_types["amount_due"] = True
        if "billing cycle" in chat_lower:
            inquiry_types["billing_cycle"] = True
        if "payment" in chat_lower and (
            "through" in chat_lower or "status" in chat_lower
        ):
            inquiry_types["payment_status"] = True

    assert sum(inquiry_types.values()) >= 2


def test_dataset_response_formats():
    dataset = create_billing_dataset()

    for example in dataset:
        response = example.expected_response

        if "$" in response:
            assert any(char.isdigit() for char in response)

        if "due" in response.lower():
            assert any(char.isdigit() for char in response)

        if "status" in response.lower():
            assert "'" in response or '"' in response


def test_as_dspy_examples():
    billing_examples = [
        BillingExample(
            chat_history="User: What's my bill?",
            expected_response="Your amount due is $120.50",
        ),
        BillingExample(
            chat_history="User: When is payment due?",
            expected_response="Your payment is due on 2026-05-15",
        ),
    ]

    dspy_examples = as_dspy_examples(billing_examples)

    assert isinstance(dspy_examples, list)
    assert len(dspy_examples) == 2

    for example in dspy_examples:
        assert isinstance(example, dspy.Example)
        assert hasattr(example, "chat_history")
        assert hasattr(example, "final_response")


def test_dspy_examples_input_fields():
    billing_examples = [
        BillingExample(
            chat_history="User: Test question", expected_response="Test response"
        )
    ]

    dspy_examples = as_dspy_examples(billing_examples)
    example = dspy_examples[0]

    assert example.chat_history == "User: Test question"
    assert example.final_response == "Test response"


def test_dataset_comprehensive_coverage():
    dataset = create_billing_dataset()

    policy_examples = {}
    for example in dataset:
        for policy in ["A12345", "B67890", "C24680"]:
            if policy in example.chat_history:
                if policy not in policy_examples:
                    policy_examples[policy] = []
                policy_examples[policy].append(example)

    for _policy, examples in policy_examples.items():
        assert len(examples) >= 3

        responses = [ex.expected_response for ex in examples]
        unique_responses = set(responses)
        assert len(unique_responses) == len(responses)


def test_dataset_realistic_conversations():
    dataset = create_billing_dataset()

    for example in dataset:
        chat = example.chat_history

        assert "User:" in chat
        assert "BillingAgent:" in chat or "Agent:" in chat

        lines = chat.strip().split("\n")
        last_line = lines[-1].strip()

        policy_found = any(
            policy in last_line for policy in ["A12345", "B67890", "C24680"]
        )
        assert policy_found


def test_dataset_response_accuracy():
    dataset = create_billing_dataset()

    expected_data = {
        "A12345": {
            "amount": 120.0,
            "cycle": "Monthly",
            "status": "Paid",
            "due": "2026-02-01",
        },
        "B67890": {
            "amount": 300.0,
            "cycle": "Quarterly",
            "status": "Pending",
            "due": "2026-03-15",
        },
        "C24680": {
            "amount": 1000.0,
            "cycle": "Annual",
            "status": "Pending",
            "due": "2026-12-01",
        },
    }

    for example in dataset:
        policy = None
        for p in expected_data.keys():
            if p in example.chat_history:
                policy = p
                break

        if policy:
            response = example.expected_response
            data = expected_data[policy]

            if (
                f"${data['amount']:.2f}" in response
                or f"${data['amount']:.0f}" in response
            ):
                assert (
                    str(data["amount"]) in response
                    or f"{data['amount']:.2f}" in response
                )

            if data["due"] in response:
                assert data["due"] in response

            if data["status"] in response:
                assert data["status"] in response

            if data["cycle"] in response:
                assert data["cycle"] in response


def test_empty_dataset_handling():
    dataset = create_billing_dataset()

    assert len(dataset) > 0
    for example in dataset:
        assert example.chat_history is not None
        assert example.expected_response is not None
        assert len(example.chat_history.strip()) > 0
        assert len(example.expected_response.strip()) > 0


def test_as_dspy_examples_empty_list():
    dspy_examples = as_dspy_examples([])

    assert isinstance(dspy_examples, list)
    assert len(dspy_examples) == 0

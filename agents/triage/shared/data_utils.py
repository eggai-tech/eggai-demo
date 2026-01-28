import dspy
import numpy as np

from agents.triage.data_sets.loader import (
    load_dataset_triage_testing,
    load_dataset_triage_training,
)


def create_examples(sample_size: int = -1, phase: str = "train", seed: int = 42) -> list[dspy.Example]:
    if phase not in ["train", "test"]:
        raise ValueError(f"Invalid phase: {phase}. Expected 'train' or 'test'.")
    if phase == "test":
        dataset = load_dataset_triage_testing()
    else:
        dataset = load_dataset_triage_training()
    total_available = len(dataset)

    if sample_size == -1:
        actual_size = total_available
    elif sample_size > total_available:
        actual_size = total_available
    else:
        actual_size = sample_size
        rs = np.random.RandomState(seed)
        keys = rs.choice(len(dataset), size=actual_size, replace=False)
        dataset = [dataset[i] for i in keys]

    print(f"Using {actual_size}/{total_available} examples")

    examples = []
    for case in dataset:
        target_agent = case.target_agent
        if hasattr(target_agent, 'value'):
            target_agent = target_agent.value

        examples.append(dspy.Example(
            chat_history=case.conversation,
            target_agent=target_agent
        ).with_inputs("chat_history"))

    return examples

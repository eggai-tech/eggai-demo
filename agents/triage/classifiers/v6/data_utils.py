import dspy

from agents.triage.shared.data_utils import (
    create_examples as _create_training_examples,
)


def create_training_examples(sample_size: int = 20, seed: int = 42) -> list[dspy.Example]:
    return _create_training_examples(sample_size=sample_size, seed=seed)

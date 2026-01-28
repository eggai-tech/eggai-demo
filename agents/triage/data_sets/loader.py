import json
from dataclasses import dataclass
from pathlib import Path

import dspy

from agents.triage.models import TargetAgent


@dataclass
class DatasetRow:
    conversation: str
    target_agent: TargetAgent
    turns: int = 0
    temperature: float = 0.7
    index_batch: int = 0
    total_batch: int = 0
    special_case: str = None
    model: str = "openai/gpt-4o-mini"
    agent_distribution: str = None
    special_case_distribution: str = None

    def __post_init__(self):
        self.conversation = self.conversation.strip()
        self.target_agent = self.target_agent.strip()
        self.special_case = self.special_case.strip() if self.special_case else None
        self.agent_distribution = self.agent_distribution.strip() if self.agent_distribution else None
        self.special_case_distribution = self.special_case_distribution.strip() if self.special_case_distribution else None


def load_dataset(file_path: Path):
    dataset = []
    with open(file_path) as file:
        for line in file:
            data = json.loads(line)
            if data.get("target_agent") is None:
                continue
            data["target_agent"] = translate_agent_str_to_enum(data["target_agent"])
            dataset.append(DatasetRow(**data))
    return dataset


def translate_agent_str_to_enum(agent_str: str) -> TargetAgent:
    if agent_str == "BillingAgent":
        return TargetAgent.BillingAgent
    elif agent_str == "PolicyAgent":
        return TargetAgent.PolicyAgent
    elif agent_str == "ClaimsAgent":
        return TargetAgent.ClaimsAgent
    elif agent_str == "EscalationAgent":
        return TargetAgent.EscalationAgent
    elif agent_str == "ChattyAgent":
        return TargetAgent.ChattyAgent
    else:
        raise ValueError(f"Unknown agent string: {agent_str}")


LABEL2ID = {
    "BillingAgent": 0,
    "PolicyAgent": 1,
    "ClaimsAgent": 2,
    "EscalationAgent": 3,
    "ChattyAgent": 4,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def load_dataset_triage_testing():
    return load_dataset(Path(__file__).resolve().parent / "triage-testing-proofread.jsonl")


def load_dataset_triage_training():
    return load_dataset(Path(__file__).resolve().parent / "triage-training-proofread.jsonl")


def as_dspy_examples(dataset: list[DatasetRow]):
    examples = []
    for row in dataset:
        examples.append(
            dspy.Example(
                chat_history=row.conversation,
                target_agent=row.target_agent,
            ).with_inputs("chat_history")
        )
    return examples


if __name__ == "__main__":
    for r in as_dspy_examples(load_dataset_triage_testing()[0:10]):
        print(r)

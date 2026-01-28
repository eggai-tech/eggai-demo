import json
import logging
from typing import Any

import mlflow
import numpy as np
import pandas as pd

CATEGORY_LABEL_MAP = {
    "BILLING": 0,
    "POLICY": 1,
    "CLAIM": 2,
    "ESCALATION": 3,
    "CHATTY": 4,
}

AGENT_TO_CLASS = {
    "BillingAgent": 0,
    "PolicyAgent": 1,
    "ClaimsAgent": 2,
    "EscalationAgent": 3,
    "ChattyAgent": 4,
}


def sample_training_examples(
    dataset: dict[str, int], n_examples: int, seed: int
) -> tuple[list[str], list[int]]:
    if n_examples is None:
        return list(dataset.keys()), list(dataset.values())
    rs = np.random.RandomState(seed)
    X_sampled = []
    y_sampled = []
    for label in set(dataset.values()):
        examples = [key for key, value in dataset.items() if value == label]
        sampled_examples = rs.choice(examples, size=n_examples, replace=False)
        X_sampled.extend(sampled_examples)
        y_sampled.extend([label] * n_examples)
    return X_sampled, y_sampled


def load_json_dataset(file_path: str) -> dict[str, int]:
    with open(file_path) as file:
        data = file.readlines()
        dataset = [json.loads(line) for line in data]
        dataset = [item for item in dataset if item["target_agent"] is not None]
        return {
            item["conversation"]: AGENT_TO_CLASS[item["target_agent"]]
            for item in dataset
        }


def load_csv_dataset(file_path: str, version="v2") -> dict[str, int]:
    df = pd.read_csv(file_path)
    return {
        row["instruction"]: CATEGORY_LABEL_MAP[row["category"].strip()]
        for _, row in df.iterrows()
    }


def load_dataset(file_path: str, version="v2") -> dict[str, int]:
    if file_path.endswith(".jsonl"):
        return load_json_dataset(file_path)
    elif file_path.endswith(".csv"):
        return load_csv_dataset(file_path, version)
    else:
        raise ValueError(
            "Unsupported file format. Please provide a .jsonl or .csv file."
        )


def load_datasets(file_paths: list[str], version="v2") -> dict[str, int]:
    dataset = {}
    for file_path in file_paths:
        dataset.update(load_dataset(file_path, version))
    return dataset


def setup_logging(log_level: str = "INFO"):
    import sys
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout
    )


def init_mlflow(cfg: dict[str, Any]):
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])
    mlflow.start_run(run_name=cfg.get("run_name", None))
    mlflow.log_params(cfg)


def unroll_dataset(train_dataset: dict[str, int]) -> dict[str, int]:
    # Data augmentation: splits conversations into progressive subsequences
    # to capture intent changes mid-conversation (e.g. user, user-agent-user, ...)
    unrolled_dataset = {}
    for instruction, label in train_dataset.items():
        unrolled_dataset[instruction] = label

        instructions = instruction.split("\n")
        for i in range(1, len(instructions), 2):
            new_instruction = "\n".join(instructions[:i])
            lbl_instruction = instructions[i]
            actor = lbl_instruction.split(":")[0]
            if not actor.endswith("Agent"):
                continue
            label = AGENT_TO_CLASS[actor]
            unrolled_dataset[new_instruction] = label

    return unrolled_dataset

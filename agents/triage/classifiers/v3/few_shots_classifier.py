import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from agents.triage.classifiers.v3.utils import sample_training_examples


class FewShotsClassifier:
    def __init__(
        self,
        n_classes: int | None = None,
        n_examples: int | None = None,
        seeds: int | list[int] = 47,
        C: float = 1.0,
        st_model_name: str = "all-MiniLM-L12-v2",
    ):
        self.n_classes = n_classes
        self.n_examples = n_examples
        self.seeds = seeds
        self.sentence_transformer = SentenceTransformer(st_model_name)
        self.C = C
        self.classifiers = []

    def fit(self, dataset: dict[str, int]):
        if isinstance(self.seeds, int) or self.n_examples is None:
            seeds = [self.seeds]
        else:
            seeds = self.seeds

        for seed in seeds:
            instructions, y = sample_training_examples(dataset, self.n_examples, seed)
            X = self.sentence_transformer.encode(instructions)
            # Use balanced weights when training on full dataset (class imbalance),
            # None when sampling equal examples per class
            class_weight = "balanced" if self.n_examples is None else None
            logistic_regression = LogisticRegression(
                penalty="l2",
                C=self.C,
                solver="lbfgs",
                max_iter=1000,
                class_weight=class_weight,
            )
            logistic_regression.fit(X, y)
            self.classifiers.append(logistic_regression)

    def __call__(self, instructions: list[str]) -> np.array:
        X = self.sentence_transformer.encode(instructions)
        y_pred = np.stack([clf.predict_proba(X) for clf in self.classifiers])
        return y_pred.mean(axis=0)

    def save(self, checkpoint_dir: str, name: str | None = None) -> Path:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if name is None:
            name = f"fewshot_classifier_n_{self.n_examples}"
        output_path = checkpoint_dir / f"{name}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(self.classifiers, f)

        return output_path

    @staticmethod
    def load(input_path: str) -> "FewShotsClassifier":
        with open(input_path, "rb") as f:
            loaded = pickle.load(f)
            if not isinstance(loaded, FewShotsClassifier):
                assert isinstance(loaded, list), (
                    "Loaded model must be a list of classifiers"
                )
                result = FewShotsClassifier()
                result.classifiers = loaded
                return result

            return loaded

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_f1_score(y_true: np.array, y_pred: np.array, n_classes: int) -> float:
    y_true_binary = np.eye(n_classes)[y_true]
    y_pred_binary = np.eye(n_classes)[y_pred.argmax(axis=1)]

    tp = np.sum(y_true_binary * y_pred_binary, axis=0)
    fp = np.sum((1 - y_true_binary) * y_pred_binary, axis=0)
    fn = np.sum(y_true_binary * (1 - y_pred_binary), axis=0)

    # Epsilon to avoid division by zero
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)

    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    return np.mean(f1_score)


def compute_roc_auc(y_true: np.array, y_pred: np.array) -> float:
    return roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")

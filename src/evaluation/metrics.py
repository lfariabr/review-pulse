"""Metric helpers for evaluation."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute accuracy and F1 from binary prediction arrays."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
    }


_classification_metrics = compute_metrics

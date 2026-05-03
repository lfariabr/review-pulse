"""Plotting helpers for evaluation artifacts."""

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from src.config import OUTPUTS_DIR

CONFUSION_PNG = OUTPUTS_DIR / "confusion_matrix.png"


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "BiLSTM — Test Confusion Matrix",
    save_path: Optional[Path] = None,
) -> np.ndarray:
    """Plot and optionally save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    classes = ["Negative", "Positive"]
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=classes, yticklabels=classes,
        xlabel="Predicted", ylabel="True",
        title=title,
    )

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"plot_confusion_matrix: saved → {save_path}")
    plt.close()
    return cm

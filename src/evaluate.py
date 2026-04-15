"""Evaluation and error analysis for ReviewPulse.

Loads the saved BiLSTM checkpoint and TF-IDF baseline, runs both against
the held-out test split, prints a side-by-side comparison, saves a
confusion matrix PNG, and writes misclassified examples to CSV.

Usage:
    python -m src.evaluate
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.dataset import (
    OUTPUTS_DIR,
    load_vocab,
    make_dataloaders,
)
from src.model import BiLSTMSentiment
from src.train import evaluate_epoch

CHECKPOINT_PATH    = OUTPUTS_DIR / "bilstm.pt"
VOCAB_PATH         = OUTPUTS_DIR / "vocab.json"
CONFUSION_PNG      = OUTPUTS_DIR / "confusion_matrix.png"
ERROR_CSV          = OUTPUTS_DIR / "error_analysis.csv"
BASELINE_PATH      = OUTPUTS_DIR / "baseline.joblib"


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> tuple:
    """Load a saved BiLSTM checkpoint and reconstruct the model.

    Returns:
        (model, model_config, history) — model is in eval mode on device.
    """
    checkpoint_path = checkpoint_path or CHECKPOINT_PATH
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt["model_config"]

    model = BiLSTMSentiment(
        vocab_size=cfg["vocab_size"],
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"load_checkpoint: loaded epoch {ckpt['best_epoch']} "
          f"(val_f1={ckpt['best_val_f1']:.4f}) ← {checkpoint_path}")
    return model, cfg, ckpt.get("history", [])


# ---------------------------------------------------------------------------
# Predictions helper
# ---------------------------------------------------------------------------

def collect_predictions(
    model: BiLSTMSentiment,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple:
    """Return (y_true, y_pred) numpy arrays for the full dataloader."""
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for tokens, labels in loader:
            tokens = tokens.to(device)
            logits = model(tokens)
            preds  = (torch.sigmoid(logits) >= 0.5).long().cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    return np.array(all_labels), np.array(all_preds)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "BiLSTM — Test Confusion Matrix",
    save_path: Optional[Path] = None,
) -> np.ndarray:
    """Plot and optionally save a confusion matrix.

    Returns:
        The 2×2 confusion matrix array.
    """
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
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    save_path = save_path or CONFUSION_PNG
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"plot_confusion_matrix: saved → {save_path}")
    return cm


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def error_analysis(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    n_examples: int = 50,
) -> pd.DataFrame:
    """Collect misclassified examples and save to CSV.

    Returns:
        DataFrame of misclassified rows with predicted label and text.
    """
    errors = test_df.copy().reset_index(drop=True)
    errors["predicted"] = y_pred
    errors["true"]      = y_true
    errors = errors[errors["predicted"] != errors["true"]].copy()

    # Categorise error type
    errors["error_type"] = errors.apply(
        lambda r: "false_positive" if r["predicted"] == 1 else "false_negative",
        axis=1,
    )

    # Sample evenly across error types for the CSV
    fp = errors[errors["error_type"] == "false_positive"].head(n_examples // 2)
    fn = errors[errors["error_type"] == "false_negative"].head(n_examples // 2)
    sample = pd.concat([fp, fn]).sort_index()

    save_path = save_path or ERROR_CSV
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    sample[["text", "true", "predicted", "error_type"]].to_csv(save_path, index=False)
    print(f"error_analysis: {len(errors)} misclassified → {save_path} "
          f"({len(sample)} examples saved)")
    return errors


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    checkpoint_path: Optional[Path] = None,
    vocab_path: Optional[Path] = None,
    baseline_path: Optional[Path] = None,
    confusion_path: Optional[Path] = None,
    error_path: Optional[Path] = None,
) -> dict:
    """Full evaluation pipeline: load model, run on test set, compare to baseline.

    Returns:
        Dict with 'bilstm' and 'baseline' metric sub-dicts.
    """
    from src.parser import load_all_domains
    from src.preprocess import preprocess
    from src.baseline import load_baseline, evaluate_baseline

    # ── Data ────────────────────────────────────────────────────────────────
    raw = load_all_domains()
    _, _, test_df = preprocess(raw)

    # ── BiLSTM ──────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model, cfg, _ = load_checkpoint(checkpoint_path, device)
    vocab = load_vocab(vocab_path or VOCAB_PATH)

    _, _, test_loader = make_dataloaders(
        test_df, test_df, test_df,
        vocab=vocab,
        batch_size=64,
        max_len=256,
        seed=42,
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    bilstm_metrics = evaluate_epoch(model, test_loader, criterion, device)

    y_true, y_pred = collect_predictions(model, test_loader, device)

    print(f"\n=== BiLSTM — test ===")
    print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))

    plot_confusion_matrix(y_true, y_pred, save_path=confusion_path)
    error_analysis(test_df, y_true, y_pred, save_path=error_path)

    # ── Baseline ────────────────────────────────────────────────────────────
    baseline = load_baseline(baseline_path or BASELINE_PATH)
    baseline_metrics = evaluate_baseline(baseline, test_df, split_name="test")

    # ── Comparison table ────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"{'Model':<20} {'Accuracy':>10} {'F1':>10}")
    print("-" * 50)
    print(f"{'TF-IDF baseline':<20} {baseline_metrics['accuracy']:>10.4f} {baseline_metrics['f1']:>10.4f}")
    print(f"{'BiLSTM+GloVe':<20} {bilstm_metrics['accuracy']:>10.4f} {bilstm_metrics['f1']:>10.4f}")
    print("=" * 50)

    return {"bilstm": bilstm_metrics, "baseline": baseline_metrics}


if __name__ == "__main__":
    run_evaluation()

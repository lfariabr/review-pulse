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
from src.inference import load_checkpoint   # avoids pulling matplotlib into app startup
from src.model import BiLSTMSentiment
from src.train import evaluate_epoch

CHECKPOINT_PATH    = OUTPUTS_DIR / "bilstm.pt"
VOCAB_PATH         = OUTPUTS_DIR / "vocab.json"
CONFUSION_PNG      = OUTPUTS_DIR / "confusion_matrix.png"
ERROR_CSV          = OUTPUTS_DIR / "error_analysis.csv"
BASELINE_PATH      = OUTPUTS_DIR / "baseline.joblib"


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


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard rounded classification metrics from predictions."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
    }


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


# ---------------------------------------------------------------------------
# DistilBERT evaluation runner
# ---------------------------------------------------------------------------

def collect_bert_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return transformer-model predictions for a full dataloader."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch["labels"].long().tolist())

    return np.array(all_labels), np.array(all_preds)


def _run_single_distilbert_evaluation(
    *,
    label: str,
    model: torch.nn.Module,
    tokenizer,
    checkpoint: dict,
    device: torch.device,
    test_df: pd.DataFrame,
    confusion_path: Path,
    error_path: Path,
) -> dict:
    """Evaluate one DistilBERT-style model on the held-out test split."""
    from src.train_bert import make_bert_test_loader

    model_cfg = checkpoint.get("model_config", {})
    batch_size = int(model_cfg.get("batch_size", 64))
    max_len = int(model_cfg.get("max_len", 256))

    test_loader = make_bert_test_loader(
        test_df,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        seed=42,
    )

    y_true, y_pred = collect_bert_predictions(model, test_loader, device)
    metrics = _classification_metrics(y_true, y_pred)

    print(f"\n=== {label} — test ===")
    print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))

    plot_confusion_matrix(
        y_true,
        y_pred,
        title=f"{label} — Test Confusion Matrix",
        save_path=confusion_path,
    )
    error_analysis(test_df, y_true, y_pred, save_path=error_path)

    return {
        **metrics,
        "best_val_f1": checkpoint.get("best_val_f1"),
        "best_epoch": checkpoint.get("best_epoch"),
    }


def _load_distilbert_test_df() -> pd.DataFrame:
    """Load the held-out test split used by DistilBERT evaluation helpers."""
    from src.parser import load_all_domains
    from src.preprocess import preprocess

    raw = load_all_domains()
    _, _, test_df = preprocess(raw)
    return test_df


def run_evaluation_local_distilbert(
    local_checkpoint_path: Optional[Path] = None,
    local_vocab_path: Optional[Path] = None,
    local_confusion_path: Optional[Path] = None,
    local_error_path: Optional[Path] = None,
) -> dict:
    """Evaluate the saved local DistilBERT-style checkpoint."""
    from src.train_bert import load_local_bert_bundle

    test_df = _load_distilbert_test_df()
    local_model, local_tokenizer, local_ckpt, local_device = load_local_bert_bundle(
        checkpoint_path=local_checkpoint_path,
        vocab_path=local_vocab_path,
    )
    return _run_single_distilbert_evaluation(
        label="Local DistilBERT",
        model=local_model,
        tokenizer=local_tokenizer,
        checkpoint=local_ckpt,
        device=local_device,
        test_df=test_df,
        confusion_path=Path(
            local_confusion_path
            or OUTPUTS_DIR / "confusion_matrix_bert_local.png"
        ),
        error_path=Path(
            local_error_path
            or OUTPUTS_DIR / "error_analysis_bert_local.csv"
        ),
    )


def run_evaluation_pretrained_distilbert(
    pretrained_checkpoint_path: Optional[Path] = None,
    pretrained_confusion_path: Optional[Path] = None,
    pretrained_error_path: Optional[Path] = None,
) -> dict:
    """Evaluate the saved pretrained DistilBERT checkpoint."""
    from src.train_bert import load_pretrained_bert_bundle

    test_df = _load_distilbert_test_df()
    pretrained_model, pretrained_tokenizer, pretrained_ckpt, pretrained_device = (
        load_pretrained_bert_bundle(checkpoint_path=pretrained_checkpoint_path)
    )
    return _run_single_distilbert_evaluation(
        label="Pretrained DistilBERT",
        model=pretrained_model,
        tokenizer=pretrained_tokenizer,
        checkpoint=pretrained_ckpt,
        device=pretrained_device,
        test_df=test_df,
        confusion_path=Path(
            pretrained_confusion_path
            or OUTPUTS_DIR / "confusion_matrix_bert_pretrained.png"
        ),
        error_path=Path(
            pretrained_error_path
            or OUTPUTS_DIR / "error_analysis_bert_pretrained.csv"
        ),
    )


def print_distilbert_comparison_table(results: dict) -> None:
    """Print a compact comparison table for DistilBERT evaluation results."""
    if not results:
        return

    print("\n" + "=" * 50)
    print(f"{'Model':<24} {'Accuracy':>10} {'F1':>10}")
    print("-" * 50)
    if "local" in results:
        print(
            f"{'Local DistilBERT':<24} "
            f"{results['local']['accuracy']:>10.4f} "
            f"{results['local']['f1']:>10.4f}"
        )
    if "pretrained" in results:
        print(
            f"{'Pretrained DistilBERT':<24} "
            f"{results['pretrained']['accuracy']:>10.4f} "
            f"{results['pretrained']['f1']:>10.4f}"
        )
    print("=" * 50)


def run_evaluation_distilbert(
    local_checkpoint_path: Optional[Path] = None,
    local_vocab_path: Optional[Path] = None,
    pretrained_checkpoint_path: Optional[Path] = None,
    local_confusion_path: Optional[Path] = None,
    local_error_path: Optional[Path] = None,
    pretrained_confusion_path: Optional[Path] = None,
    pretrained_error_path: Optional[Path] = None,
    evaluate_local: bool = True,
    evaluate_pretrained: bool = True,
) -> dict:
    """Evaluate the saved local and/or pretrained DistilBERT checkpoints.

    This mirrors ``run_evaluation`` but targets the transformer checkpoints
    produced by ``src.train_bert``.
    """
    if not evaluate_local and not evaluate_pretrained:
        raise ValueError("At least one of evaluate_local or evaluate_pretrained must be True.")

    results = {}

    if evaluate_local:
        results["local"] = run_evaluation_local_distilbert(
            local_checkpoint_path=local_checkpoint_path,
            local_vocab_path=local_vocab_path,
            local_confusion_path=local_confusion_path,
            local_error_path=local_error_path,
        )

    if evaluate_pretrained:
        results["pretrained"] = run_evaluation_pretrained_distilbert(
            pretrained_checkpoint_path=pretrained_checkpoint_path,
            pretrained_confusion_path=pretrained_confusion_path,
            pretrained_error_path=pretrained_error_path,
        )

    print_distilbert_comparison_table(results)
    return results


if __name__ == "__main__":
    run_evaluation()
    from src.train_bert import (
        CHECKPOINT_PATH as LOCAL_BERT_CHECKPOINT_PATH,
        PRETRAINED_CHECKPOINT_PATH,
    )

    bert_results = {}

    if not LOCAL_BERT_CHECKPOINT_PATH.exists():
        print(
            "Skipping DistilBERT evaluation: "
            f"local checkpoint not found at {LOCAL_BERT_CHECKPOINT_PATH}"
        )
    else:
        bert_results["local"] = run_evaluation_local_distilbert(
            local_checkpoint_path=LOCAL_BERT_CHECKPOINT_PATH,
        )

    if PRETRAINED_CHECKPOINT_PATH.exists():
        bert_results["pretrained"] = run_evaluation_pretrained_distilbert(
            pretrained_checkpoint_path=PRETRAINED_CHECKPOINT_PATH,
        )
    else:
        print(
            "Skipping pretrained DistilBERT evaluation: "
            f"checkpoint not found at {PRETRAINED_CHECKPOINT_PATH}"
        )

    print_distilbert_comparison_table(bert_results)

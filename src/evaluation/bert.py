"""DistilBERT evaluation runners."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

from src.config import OUTPUTS_DIR, PRED_THRESHOLD

from .errors import error_analysis
from .metrics import _classification_metrics
from .plots import plot_confusion_matrix


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
            preds = (torch.sigmoid(logits) >= PRED_THRESHOLD).long().cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch["labels"].long().tolist())

    return np.array(all_labels), np.array(all_preds)


def _run_distilbert_evaluation(
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
    """Evaluate the Hugging Face DistilBERT model on the held-out test split."""
    from src.training.bert import make_bert_test_loader

    model_cfg = checkpoint.get("model_config", {})
    batch_size = int(model_cfg.get("batch_size", 64))
    max_len = int(model_cfg.get("max_len", 256))

    test_loader = make_bert_test_loader(
        test_df,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
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


def run_evaluation_distilbert(
    checkpoint_path: Optional[Path] = None,
    confusion_path: Optional[Path] = None,
    error_path: Optional[Path] = None,
    label: str = "DistilBERT",
) -> dict:
    """Evaluate the saved Hugging Face DistilBERT checkpoint."""
    from src.training.bert import load_pretrained_bert_bundle

    test_df = _load_distilbert_test_df()
    model, tokenizer, checkpoint, device = load_pretrained_bert_bundle(
        checkpoint_path=checkpoint_path,
    )
    metrics = _run_distilbert_evaluation(
        label=label,
        model=model,
        tokenizer=tokenizer,
        checkpoint=checkpoint,
        device=device,
        test_df=test_df,
        confusion_path=Path(
            confusion_path or OUTPUTS_DIR / "confusion_matrix_distilbert.png"
        ),
        error_path=Path(
            error_path or OUTPUTS_DIR / "error_analysis_distilbert.csv"
        ),
    )
    print("\n" + "=" * 50)
    print(f"{'Model':<20} {'Accuracy':>10} {'F1':>10}")
    print("-" * 50)
    print(f"{label:<20} {metrics['accuracy']:>10.4f} {metrics['f1']:>10.4f}")
    print("=" * 50)
    return metrics


def run_evaluation_distilbert_deploy(
    checkpoint_path: Optional[Path] = None,
    confusion_path: Optional[Path] = None,
    error_path: Optional[Path] = None,
) -> dict:
    """Evaluate the compact deployment DistilBERT bundle."""
    from src.config import DISTILBERT_PATH

    return run_evaluation_distilbert(
        checkpoint_path=checkpoint_path or DISTILBERT_PATH,
        confusion_path=confusion_path
        or OUTPUTS_DIR / "confusion_matrix_distilbert_deploy.png",
        error_path=error_path or OUTPUTS_DIR / "error_analysis_distilbert_deploy.csv",
        label="DistilBERT deploy",
    )

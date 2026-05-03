"""BiLSTM evaluation runner and prediction collection."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import classification_report

from src.config import BASELINE_PATH, BILSTM_CHECKPOINT_PATH, PRED_THRESHOLD, VOCAB_PATH
from src.inference import load_checkpoint
from src.models.bilstm import BiLSTMSentiment
from src.tokenization.sequence import make_dataloaders
from src.tokenization.vocab import load_vocab
from src.training.bilstm import evaluate_epoch

from .errors import ERROR_CSV, error_analysis
from .plots import CONFUSION_PNG, plot_confusion_matrix

CHECKPOINT_PATH = BILSTM_CHECKPOINT_PATH


def collect_predictions(
    model: BiLSTMSentiment,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple:
    """Return (y_true, y_pred) numpy arrays for the full dataloader."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for tokens, labels in loader:
            tokens = tokens.to(device)
            logits = model(tokens)
            preds = (torch.sigmoid(logits) >= PRED_THRESHOLD).long().cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    return np.array(all_labels), np.array(all_preds)


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_evaluation(
    checkpoint_path: Optional[Path] = None,
    vocab_path: Optional[Path] = None,
    baseline_path: Optional[Path] = None,
    confusion_path: Optional[Path] = None,
    error_path: Optional[Path] = None,
    save_outputs: bool = True,
) -> dict:
    """Full evaluation pipeline: load model, run on test set, compare to baseline."""
    from src.parser import load_all_domains
    from src.preprocess import preprocess
    from src.training.baseline import evaluate_baseline, load_baseline

    raw = load_all_domains()
    _, _, test_df = preprocess(raw)

    device = _resolve_device()
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

    plot_confusion_matrix(
        y_true, y_pred,
        save_path=(confusion_path or CONFUSION_PNG) if save_outputs else None,
    )
    error_analysis(
        test_df, y_true, y_pred,
        save_path=(error_path or ERROR_CSV) if save_outputs else None,
    )

    baseline = load_baseline(baseline_path or BASELINE_PATH)
    baseline_metrics = evaluate_baseline(baseline, test_df, split_name="test")

    print("\n" + "=" * 50)
    print(f"{'Model':<20} {'Accuracy':>10} {'F1':>10}")
    print("-" * 50)
    print(
        f"{'TF-IDF baseline':<20} "
        f"{baseline_metrics['accuracy']:>10.4f} {baseline_metrics['f1']:>10.4f}"
    )
    print(
        f"{'BiLSTM+GloVe':<20} "
        f"{bilstm_metrics['accuracy']:>10.4f} {bilstm_metrics['f1']:>10.4f}"
    )
    print("=" * 50)

    return {"bilstm": bilstm_metrics, "baseline": baseline_metrics}

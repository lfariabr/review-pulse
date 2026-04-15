# .venv/bin/pytest tests/test_evaluate.py -v

"""Tests for src/evaluate.py."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluate import (
    collect_predictions,
    error_analysis,
    load_checkpoint,
    plot_confusion_matrix,
)
from src.model import BiLSTMSentiment
from src.dataset import build_vocab, make_dataloaders

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE    = 100
EMBEDDING_DIM = 16
HIDDEN_DIM    = 16
BATCH         = 8
SEQ_LEN       = 20

TEXTS = [
    "great product love it highly recommend",
    "excellent quality exceeded expectations fantastic",
    "terrible product broke waste money",
    "do not buy garbage worst purchase",
    "best purchase ever works perfectly time",
    "very disappointed with quality not worth",
    "outstanding performance really impressed great",
    "stopped working week would not recommend",
]
LABELS = [1, 1, 0, 0, 1, 0, 1, 0]


def _small_df():
    return pd.DataFrame({"text": TEXTS, "label": LABELS})


def _tiny_model():
    m = BiLSTMSentiment(
        vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM, n_layers=1, dropout=0.0,
    )
    m.eval()
    return m


def _loader():
    df    = _small_df()
    vocab = build_vocab(df["text"], min_freq=1)
    _, _, loader = make_dataloaders(
        df, df, df, vocab=vocab, batch_size=4, max_len=SEQ_LEN, seed=42
    )
    return loader


# ---------------------------------------------------------------------------
# load_checkpoint
# ---------------------------------------------------------------------------

def test_load_checkpoint_returns_model_and_config(tmp_path):
    vocab_size = 50
    model = BiLSTMSentiment(vocab_size=vocab_size, embedding_dim=16, hidden_dim=16, n_layers=1)
    ckpt_path = tmp_path / "bilstm.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {
                "vocab_size": vocab_size, "embedding_dim": 16,
                "hidden_dim": 16, "n_layers": 1, "dropout": 0.5,
            },
            "vocab_path": str(tmp_path / "vocab.json"),
            "best_val_f1": 0.85,
            "best_epoch": 3,
            "history": [],
        },
        ckpt_path,
    )
    loaded_model, cfg, history = load_checkpoint(ckpt_path, device=torch.device("cpu"))
    assert isinstance(loaded_model, BiLSTMSentiment)
    assert cfg["vocab_size"] == vocab_size
    assert isinstance(history, list)


def test_load_checkpoint_model_is_in_eval_mode(tmp_path):
    vocab_size = 50
    model = BiLSTMSentiment(vocab_size=vocab_size, embedding_dim=16, hidden_dim=16, n_layers=1)
    ckpt_path = tmp_path / "bilstm.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {
                "vocab_size": vocab_size, "embedding_dim": 16,
                "hidden_dim": 16, "n_layers": 1, "dropout": 0.5,
            },
            "vocab_path": "", "best_val_f1": 0.0, "best_epoch": 1, "history": [],
        },
        ckpt_path,
    )
    loaded_model, _, _ = load_checkpoint(ckpt_path, device=torch.device("cpu"))
    assert not loaded_model.training


# ---------------------------------------------------------------------------
# collect_predictions
# ---------------------------------------------------------------------------

def test_collect_predictions_returns_arrays():
    loader = _loader()
    model  = _tiny_model()
    y_true, y_pred = collect_predictions(model, loader, torch.device("cpu"))
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_pred, np.ndarray)


def test_collect_predictions_same_length_as_dataset():
    df     = _small_df()
    loader = _loader()
    model  = _tiny_model()
    y_true, y_pred = collect_predictions(model, loader, torch.device("cpu"))
    assert len(y_true) == len(df)
    assert len(y_pred) == len(df)


def test_collect_predictions_binary():
    loader = _loader()
    model  = _tiny_model()
    _, y_pred = collect_predictions(model, loader, torch.device("cpu"))
    assert set(y_pred).issubset({0, 1})


# ---------------------------------------------------------------------------
# plot_confusion_matrix
# ---------------------------------------------------------------------------

def test_plot_confusion_matrix_saves_file(tmp_path):
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    save_path = tmp_path / "cm.png"
    plot_confusion_matrix(y_true, y_pred, save_path=save_path)
    assert save_path.exists()


def test_plot_confusion_matrix_returns_array(tmp_path):
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    cm = plot_confusion_matrix(y_true, y_pred, save_path=tmp_path / "cm.png")
    assert cm.shape == (2, 2)
    assert cm.sum() == len(y_true)


# ---------------------------------------------------------------------------
# error_analysis
# ---------------------------------------------------------------------------

def test_error_analysis_saves_csv(tmp_path):
    df     = _small_df()
    y_true = np.array(LABELS)
    y_pred = np.array([1, 1, 1, 0, 1, 1, 1, 0])   # some wrong
    save_path = tmp_path / "errors.csv"
    error_analysis(df, y_true, y_pred, save_path=save_path)
    assert save_path.exists()


def test_error_analysis_csv_has_expected_columns(tmp_path):
    df     = _small_df()
    y_true = np.array(LABELS)
    y_pred = np.array([1, 1, 1, 0, 1, 1, 1, 0])
    save_path = tmp_path / "errors.csv"
    error_analysis(df, y_true, y_pred, save_path=save_path)
    result = pd.read_csv(save_path)
    for col in ("text", "true", "predicted", "error_type"):
        assert col in result.columns


def test_error_analysis_returns_only_misclassified(tmp_path):
    df     = _small_df()
    y_true = np.array(LABELS)
    y_pred = y_true.copy()
    y_pred[0] = 1 - y_pred[0]   # flip exactly one
    errors = error_analysis(df, y_true, y_pred, save_path=tmp_path / "e.csv")
    assert len(errors) == 1


def test_error_analysis_perfect_predictions_empty(tmp_path):
    df     = _small_df()
    y_true = np.array(LABELS)
    errors = error_analysis(df, y_true, y_true.copy(), save_path=tmp_path / "e.csv")
    assert len(errors) == 0


# ---------------------------------------------------------------------------
# integration — real data
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_run_evaluation_returns_metrics():
    """Full evaluation pipeline on real data."""
    import tempfile
    from src.evaluate import run_evaluation

    with tempfile.TemporaryDirectory() as tmp:
        result = run_evaluation(
            confusion_path=tmp + "/cm.png",
            error_path=tmp + "/errors.csv",
        )

    assert "bilstm"   in result
    assert "baseline" in result
    assert result["bilstm"]["accuracy"] >= 0.75

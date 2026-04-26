# .venv/bin/pytest tests/test_evaluate.py -v
# .venv/bin/pytest tests/test_evaluate.py -v -m "not slow"

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
    run_evaluation_distilbert_deploy,
    run_evaluation_distilbert,
)
from src.model import BiLSTMSentiment
from src.dataset import build_vocab, make_dataloaders
from tiny_tokenizer import TinyTokenizer

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


def test_run_evaluation_distilbert_returns_metrics(monkeypatch, tmp_path):
    """DistilBERT evaluation should return metrics for the HF checkpoint."""
    import src.parser as parser_module
    import src.preprocess as preprocess_module
    import src.train_bert as train_bert_module

    test_df = _small_df()
    vocab = build_vocab(test_df["text"], min_freq=1)
    tokenizer = TinyTokenizer(vocab)

    class TinyBertClassifier(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            del attention_mask
            return (input_ids[:, 0] % 2).float() - 0.5

    checkpoint = {
        "model_config": {"batch_size": 4, "max_len": 16},
        "best_val_f1": 0.8,
        "best_epoch": 2,
    }

    monkeypatch.setattr(parser_module, "load_all_domains", lambda: pd.DataFrame())
    monkeypatch.setattr(preprocess_module, "preprocess", lambda raw: (None, None, test_df))
    monkeypatch.setattr(
        train_bert_module,
        "load_pretrained_bert_bundle",
        lambda checkpoint_path=None: (
            TinyBertClassifier(),
            tokenizer,
            checkpoint,
            torch.device("cpu"),
        ),
    )

    result = run_evaluation_distilbert(
        confusion_path=tmp_path / "cm_distilbert.png",
        error_path=tmp_path / "errors_distilbert.csv",
    )

    assert "accuracy" in result
    assert "f1" in result
    assert result["best_val_f1"] == 0.8
    assert (tmp_path / "cm_distilbert.png").exists()
    assert (tmp_path / "errors_distilbert.csv").exists()


def test_run_evaluation_distilbert_deploy_uses_deploy_checkpoint(monkeypatch, tmp_path):
    """Deployment DistilBERT evaluation should load the compact deploy bundle."""
    import src.parser as parser_module
    import src.preprocess as preprocess_module
    import src.evaluate as evaluate_module
    import src.train_bert as train_bert_module

    test_df = _small_df()
    deploy_checkpoint = tmp_path / "distilbert_deploy" / "metadata.pt"
    seen = {}
    tokenizer = TinyTokenizer()

    class TinyBertClassifier(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            del attention_mask
            return torch.ones(input_ids.shape[0])

    checkpoint = {
        "model_config": {"batch_size": 4, "max_len": 16},
        "best_val_f1": 0.8452,
        "best_epoch": 4,
    }

    def fake_load_bundle(checkpoint_path=None):
        seen["checkpoint_path"] = checkpoint_path
        return TinyBertClassifier(), tokenizer, checkpoint, torch.device("cpu")

    monkeypatch.setattr(parser_module, "load_all_domains", lambda: pd.DataFrame())
    monkeypatch.setattr(preprocess_module, "preprocess", lambda raw: (None, None, test_df))
    monkeypatch.setattr(
        evaluate_module,
        "DISTILBERT_DEPLOY_CHECKPOINT_PATH",
        deploy_checkpoint,
    )
    monkeypatch.setattr(
        train_bert_module,
        "load_pretrained_bert_bundle",
        fake_load_bundle,
    )

    result = run_evaluation_distilbert_deploy(
        confusion_path=tmp_path / "cm_deploy.png",
        error_path=tmp_path / "errors_deploy.csv",
    )

    assert seen["checkpoint_path"] == deploy_checkpoint
    assert result["best_val_f1"] == 0.8452
    assert (tmp_path / "cm_deploy.png").exists()
    assert (tmp_path / "errors_deploy.csv").exists()

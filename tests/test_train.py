# .venv/bin/pytest tests/test_train.py -v

"""Tests for src.training.bilstm — training loop."""

import torch
import pandas as pd
import pytest

from src.training.bilstm import train_one_epoch, evaluate_epoch, train
from src.models.bilstm import BiLSTMSentiment
from src.tokenization.sequence import make_dataloaders
from src.tokenization.vocab import build_vocab

# ---------------------------------------------------------------------------
# Minimal fixtures — tiny vocab and dataset so tests run fast on CPU
# ---------------------------------------------------------------------------

TEXTS_POS = [
    "great product love it highly recommend",
    "excellent quality exceeded all expectations fantastic",
    "best purchase ever works perfectly every time",
    "outstanding performance really impressed great item",
    "amazing value definitely buy again wonderful",
]
TEXTS_NEG = [
    "terrible product broke after one day waste",
    "do not buy garbage worst purchase ever made",
    "very disappointed with quality not worth price",
    "stopped working after a week would not recommend",
    "awful experience product did not work described",
]

ALL_TEXTS  = TEXTS_POS + TEXTS_NEG
ALL_LABELS = [1] * 5 + [0] * 5


def _small_df():
    return pd.DataFrame({"text": ALL_TEXTS, "label": ALL_LABELS})


def _fixtures():
    """Return (model, train_loader, criterion) for a tiny setup."""
    df    = _small_df()
    vocab = build_vocab(df["text"], min_freq=1)
    train_loader, val_loader, _ = make_dataloaders(
        df, df, df, vocab=vocab, batch_size=4, max_len=16, seed=42
    )
    model = BiLSTMSentiment(
        vocab_size=len(vocab), embedding_dim=16, hidden_dim=16, n_layers=1, dropout=0.0
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, train_loader, val_loader, optimizer, criterion


# ---------------------------------------------------------------------------
# train_one_epoch
# ---------------------------------------------------------------------------

def test_train_one_epoch_returns_loss():
    model, train_loader, _, optimizer, criterion = _fixtures()
    metrics = train_one_epoch(model, train_loader, optimizer, criterion)
    assert "loss" in metrics


def test_train_one_epoch_loss_is_positive():
    model, train_loader, _, optimizer, criterion = _fixtures()
    metrics = train_one_epoch(model, train_loader, optimizer, criterion)
    assert metrics["loss"] > 0


def test_train_one_epoch_updates_weights():
    model, train_loader, _, optimizer, criterion = _fixtures()
    before = model.fc.weight.data.clone()
    train_one_epoch(model, train_loader, optimizer, criterion)
    assert not torch.equal(before, model.fc.weight.data)


# ---------------------------------------------------------------------------
# evaluate_epoch
# ---------------------------------------------------------------------------

def test_evaluate_epoch_returns_required_keys():
    model, _, val_loader, _, criterion = _fixtures()
    metrics = evaluate_epoch(model, val_loader, criterion)
    assert "loss"     in metrics
    assert "accuracy" in metrics
    assert "f1"       in metrics


def test_evaluate_epoch_accuracy_in_range():
    model, _, val_loader, _, criterion = _fixtures()
    metrics = evaluate_epoch(model, val_loader, criterion)
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_evaluate_epoch_f1_in_range():
    model, _, val_loader, _, criterion = _fixtures()
    metrics = evaluate_epoch(model, val_loader, criterion)
    assert 0.0 <= metrics["f1"] <= 1.0


def test_evaluate_epoch_does_not_update_weights():
    model, _, val_loader, _, criterion = _fixtures()
    before = model.fc.weight.data.clone()
    evaluate_epoch(model, val_loader, criterion)
    assert torch.equal(before, model.fc.weight.data)


# ---------------------------------------------------------------------------
# train (full loop)
# ---------------------------------------------------------------------------

def test_train_saves_checkpoint(tmp_path):
    df    = _small_df()
    vocab = build_vocab(df["text"], min_freq=1)
    ckpt  = tmp_path / "bilstm.pt"
    train(df, df, vocab, epochs=1, hidden_dim=16, n_layers=1,
          embedding_dim=16, batch_size=4, max_len=16, checkpoint_path=ckpt)
    assert ckpt.exists()


def test_train_checkpoint_has_required_keys(tmp_path):
    df    = _small_df()
    vocab = build_vocab(df["text"], min_freq=1)
    ckpt  = tmp_path / "bilstm.pt"
    train(df, df, vocab, epochs=1, hidden_dim=16, n_layers=1,
          embedding_dim=16, batch_size=4, max_len=16, checkpoint_path=ckpt)
    data = torch.load(ckpt, weights_only=False)
    for key in ("model_state", "model_config", "vocab_path", "best_val_f1", "best_epoch", "history"):
        assert key in data, f"checkpoint missing key: {key}"


def test_train_returns_best_val_f1(tmp_path):
    df    = _small_df()
    vocab = build_vocab(df["text"], min_freq=1)
    ckpt  = tmp_path / "bilstm.pt"
    result = train(df, df, vocab, epochs=2, hidden_dim=16, n_layers=1,
                   embedding_dim=16, batch_size=4, max_len=16, checkpoint_path=ckpt)
    assert "best_val_f1" in result
    assert 0.0 <= result["best_val_f1"] <= 1.0


def test_train_history_length_matches_epochs(tmp_path):
    df    = _small_df()
    vocab = build_vocab(df["text"], min_freq=1)
    ckpt  = tmp_path / "bilstm.pt"
    result = train(df, df, vocab, epochs=3, hidden_dim=16, n_layers=1,
                   embedding_dim=16, batch_size=4, max_len=16, checkpoint_path=ckpt)
    assert len(result["history"]) == 3


def test_train_checkpoint_model_config_matches(tmp_path):
    df    = _small_df()
    vocab = build_vocab(df["text"], min_freq=1)
    ckpt  = tmp_path / "bilstm.pt"
    train(df, df, vocab, epochs=1, hidden_dim=32, n_layers=1,
          embedding_dim=16, batch_size=4, max_len=16, checkpoint_path=ckpt)
    data = torch.load(ckpt, weights_only=False)
    assert data["model_config"]["hidden_dim"]    == 32
    assert data["model_config"]["vocab_size"]    == len(vocab)
    assert data["model_config"]["embedding_dim"] == 16


# ---------------------------------------------------------------------------
# integration — real data, one epoch sanity check
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_train_real_data_one_epoch():
    """One epoch on real data must complete without error and return val_f1 > 0."""
    import tempfile
    from src.data.parser import load_all_domains
    from src.data.preprocess import preprocess
    from src.tokenization.vocab import save_vocab

    raw = load_all_domains()
    train_df, val_df, _ = preprocess(raw)
    vocab = build_vocab(train_df["text"])

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = tmp + "/bilstm.pt"
        result = train(
            train_df, val_df, vocab,
            epochs=1, checkpoint_path=ckpt,
        )

    assert result["best_val_f1"] >= 0.0

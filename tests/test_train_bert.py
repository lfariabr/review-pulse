# .venv/bin/pytest tests/test_train_bert.py -v

"""Tests for src/train_bert.py — local transformer training loop."""

import pandas as pd
import torch

from src.dataset import build_vocab
from src.model_bert import DistilBERTSentiment
from src.train_bert import (
    build_local_pretraining_corpus,
    evaluate_epoch_bert,
    load_tokenizer,
    make_bert_dataloaders,
    pretrain_local_bert,
    train_bert,
    train_one_epoch_bert,
)

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

ALL_TEXTS = TEXTS_POS + TEXTS_NEG
ALL_LABELS = [1] * 5 + [0] * 5


def _small_df():
    return pd.DataFrame({"text": ALL_TEXTS, "label": ALL_LABELS})


def _fixtures():
    df = _small_df()
    vocab = build_vocab(df["text"], min_freq=1)
    tokenizer = load_tokenizer(vocab=vocab)
    train_loader, val_loader, _ = make_bert_dataloaders(
        df,
        df,
        df,
        tokenizer=tokenizer,
        batch_size=4,
        max_len=16,
        seed=42,
    )
    model = DistilBERTSentiment(
        vocab_size=len(vocab),
        embedding_dim=16,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        max_len=16,
        dropout=0.0,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, train_loader, val_loader, optimizer, criterion


def test_load_tokenizer_returns_local_encodings():
    vocab = build_vocab(_small_df()["text"], min_freq=1)
    tokenizer = load_tokenizer(vocab=vocab)
    encodings = tokenizer(["great product", "terrible product"], max_length=8)
    assert set(encodings.keys()) == {"input_ids", "attention_mask"}
    assert encodings["input_ids"].shape == (2, 8)
    assert encodings["attention_mask"].shape == (2, 8)


def test_train_one_epoch_bert_returns_positive_loss():
    model, train_loader, _, optimizer, criterion = _fixtures()
    metrics = train_one_epoch_bert(model, train_loader, optimizer, criterion)
    assert metrics["loss"] > 0


def test_evaluate_epoch_bert_returns_required_keys():
    model, _, val_loader, _, criterion = _fixtures()
    metrics = evaluate_epoch_bert(model, val_loader, criterion)
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "f1" in metrics


def test_train_bert_saves_checkpoint_and_vocab(tmp_path):
    df = _small_df()
    ckpt = tmp_path / "distilbert.pt"
    vocab_path = tmp_path / "distilbert_vocab.json"

    result = train_bert(
        df,
        df,
        epochs=1,
        pretrain_epochs=0,
        batch_size=4,
        max_len=16,
        checkpoint_path=ckpt,
        vocab_path=vocab_path,
        embedding_dim=16,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
    )

    assert ckpt.exists()
    assert vocab_path.exists()
    assert result["best_val_f1"] >= 0.0


def test_train_bert_checkpoint_has_required_keys(tmp_path):
    df = _small_df()
    ckpt = tmp_path / "distilbert.pt"
    vocab_path = tmp_path / "distilbert_vocab.json"

    train_bert(
        df,
        df,
        epochs=1,
        pretrain_epochs=0,
        batch_size=4,
        max_len=16,
        checkpoint_path=ckpt,
        vocab_path=vocab_path,
        embedding_dim=16,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
    )

    data = torch.load(ckpt, weights_only=False)
    for key in (
        "model_state",
        "model_config",
        "tokenizer_name",
        "vocab_path",
        "best_val_f1",
        "best_epoch",
        "history",
    ):
        assert key in data, f"checkpoint missing key: {key}"

    assert data["model_config"]["vocab_size"] >= 2
    assert data["model_config"]["embedding_dim"] == 16
    assert data["model_config"]["model_type"] == "local"
    assert data["tokenizer_name"] == str(vocab_path)


def test_train_bert_checkpoint_can_be_reloaded_locally(tmp_path):
    df = _small_df()
    ckpt = tmp_path / "distilbert.pt"
    vocab_path = tmp_path / "distilbert_vocab.json"

    train_bert(
        df,
        df,
        epochs=1,
        pretrain_epochs=0,
        batch_size=4,
        max_len=16,
        checkpoint_path=ckpt,
        vocab_path=vocab_path,
        embedding_dim=16,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
    )

    data = torch.load(ckpt, weights_only=False)
    cfg = data["model_config"]
    model = DistilBERTSentiment(
        model_name=cfg["model_name"],
        dropout=cfg["dropout"],
        freeze_encoder=cfg["freeze_encoder"],
        vocab_size=cfg["vocab_size"],
        embedding_dim=cfg["embedding_dim"],
        token_embedding_dim=cfg.get("token_embedding_dim"),
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        ff_dim=cfg["ff_dim"],
        max_len=cfg["max_len"],
    )
    model.load_state_dict(data["model_state"])

    tokenizer = load_tokenizer(data["tokenizer_name"])
    encodings = tokenizer(["great product"], max_length=16)

    with torch.no_grad():
        logits = model(
            encodings["input_ids"],
            encodings["attention_mask"],
        )

    assert logits.shape == (1,)


def test_train_bert_can_disable_glove(tmp_path):
    df = _small_df()
    ckpt = tmp_path / "distilbert_local.pt"
    vocab_path = tmp_path / "distilbert_vocab.json"

    train_bert(
        df,
        df,
        epochs=1,
        pretrain_epochs=0,
        batch_size=4,
        max_len=16,
        checkpoint_path=ckpt,
        vocab_path=vocab_path,
        embedding_dim=16,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        use_glove=False,
    )

    data = torch.load(ckpt, weights_only=False)
    assert data["model_config"]["use_glove"] is False


def test_build_local_pretraining_corpus_uses_train_texts_when_unlabeled_disabled():
    corpus = build_local_pretraining_corpus(_small_df(), include_unlabeled=False)
    assert len(corpus) == len(_small_df())
    assert set(corpus) == set(_small_df()["text"])


def test_pretrain_local_bert_returns_history():
    df = _small_df()
    vocab = build_vocab(df["text"], min_freq=1)
    model = DistilBERTSentiment(
        vocab_size=len(vocab),
        embedding_dim=16,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        max_len=16,
        dropout=0.0,
    )

    result = pretrain_local_bert(
        model,
        df,
        vocab=vocab,
        epochs=1,
        batch_size=4,
        max_len=16,
        include_unlabeled=False,
    )

    assert result["corpus_size"] == len(df)
    assert len(result["history"]) == 1
    assert result["history"][0]["loss"] > 0

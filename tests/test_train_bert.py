# .venv/bin/pytest tests/test_train_bert.py -v

"""Tests for src/train_bert.py — Hugging Face DistilBERT training loop."""

import pandas as pd
import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification

import src.model_bert as model_bert_module
from src.model_bert import DistilBERTSentiment
from tiny_tokenizer import TinyTokenizer
from src.train_bert import (
    evaluate_epoch_bert,
    load_pretrained_bert_bundle,
    load_tokenizer,
    make_bert_dataloaders,
    train_bert,
    train_one_epoch_bert,
)
import src.train_bert as train_bert_module

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
VOCAB = {"[PAD]": 0, "[UNK]": 1}
for text in ALL_TEXTS:
    for token in text.split():
        VOCAB.setdefault(token, len(VOCAB))


def _small_df():
    return pd.DataFrame({"text": ALL_TEXTS, "label": ALL_LABELS})


def _tiny_hf_model() -> DistilBertForSequenceClassification:
    config = DistilBertConfig(
        vocab_size=len(VOCAB),
        n_layers=1,
        dim=32,
        hidden_dim=64,
        n_heads=4,
        dropout=0.0,
        attention_dropout=0.0,
        seq_classif_dropout=0.0,
        num_labels=1,
    )
    return DistilBertForSequenceClassification(config)


def _patch_hf(monkeypatch):
    def model_from_pretrained(cls, *args, **kwargs):
        return _tiny_hf_model()

    def tokenizer_from_pretrained(cls, *args, **kwargs):
        return TinyTokenizer(VOCAB)

    monkeypatch.setattr(
        model_bert_module.DistilBertForSequenceClassification,
        "from_pretrained",
        classmethod(model_from_pretrained),
    )
    monkeypatch.setattr(
        train_bert_module.AutoTokenizer,
        "from_pretrained",
        classmethod(tokenizer_from_pretrained),
    )


def _fixtures(monkeypatch):
    _patch_hf(monkeypatch)
    df = _small_df()
    tokenizer = load_tokenizer()
    train_loader, val_loader, _ = make_bert_dataloaders(
        df,
        df,
        df,
        tokenizer=tokenizer,
        batch_size=4,
        max_len=16,
        seed=42,
    )
    model = DistilBERTSentiment(dropout=0.0)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=1e-3,
    )
    return model, train_loader, val_loader, optimizer, criterion


def test_load_tokenizer_returns_hf_style_encodings(monkeypatch):
    _patch_hf(monkeypatch)
    tokenizer = load_tokenizer()
    encodings = tokenizer(["great product", "terrible product"], max_length=8)
    assert set(encodings.keys()) == {"input_ids", "attention_mask"}
    assert encodings["input_ids"].shape == (2, 8)
    assert encodings["attention_mask"].shape == (2, 8)


def test_make_bert_dataloaders_returns_labelled_batches(monkeypatch):
    _patch_hf(monkeypatch)
    df = _small_df()
    tokenizer = load_tokenizer()
    train_loader, val_loader, test_loader = make_bert_dataloaders(
        df,
        df,
        df,
        tokenizer=tokenizer,
        batch_size=4,
        max_len=16,
    )
    batch = next(iter(train_loader))
    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}
    assert len(val_loader.dataset) == len(df)
    assert len(test_loader.dataset) == len(df)


def test_train_one_epoch_bert_returns_positive_loss(monkeypatch):
    model, train_loader, _, optimizer, criterion = _fixtures(monkeypatch)
    metrics = train_one_epoch_bert(model, train_loader, optimizer, criterion)
    assert metrics["loss"] > 0


def test_evaluate_epoch_bert_returns_required_keys(monkeypatch):
    model, _, val_loader, _, criterion = _fixtures(monkeypatch)
    metrics = evaluate_epoch_bert(model, val_loader, criterion)
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "f1" in metrics


def test_train_bert_saves_hugging_face_checkpoint(monkeypatch, tmp_path):
    _patch_hf(monkeypatch)
    df = _small_df()
    ckpt = tmp_path / "distilbert.pt"

    result = train_bert(
        df,
        df,
        epochs=1,
        batch_size=4,
        max_len=16,
        checkpoint_path=ckpt,
    )

    assert ckpt.exists()
    assert result["best_val_f1"] >= 0.0


def test_train_bert_checkpoint_has_required_keys(monkeypatch, tmp_path):
    _patch_hf(monkeypatch)
    df = _small_df()
    ckpt = tmp_path / "distilbert.pt"

    train_bert(
        df,
        df,
        epochs=1,
        batch_size=4,
        max_len=16,
        checkpoint_path=ckpt,
    )

    data = torch.load(ckpt, weights_only=False)
    for key in (
        "model_config",
        "tokenizer_name",
        "best_val_f1",
        "best_epoch",
        "history",
        "model_state",
        "weights_format",
        "weights_dtype",
        "save_strategy",
    ):
        assert key in data, f"checkpoint missing key: {key}"

    assert data["model_config"]["model_type"] == "pretrained"
    assert data["model_config"]["freeze_encoder"] is True
    assert data["model_config"]["num_labels"] == 1
    assert data["weights_format"] == "torch_state_dict"
    assert data["weights_dtype"] == "float16"
    assert data["save_strategy"] == "head_only"
    assert data["tokenizer_files"]
    assert not (tmp_path / "distilbert.safetensors").exists()


def test_head_only_checkpoint_saves_only_classifier_safetensors(monkeypatch, tmp_path):
    _patch_hf(monkeypatch)
    df = _small_df()
    ckpt = tmp_path / "distilbert.pt"

    train_bert(
        df,
        df,
        epochs=1,
        batch_size=4,
        max_len=16,
        checkpoint_path=ckpt,
    )

    data = torch.load(ckpt, weights_only=False)
    weights = data["model_state"]

    assert data["save_strategy"] == "head_only"
    assert weights
    assert all(not key.startswith("model.distilbert.") for key in weights)
    assert all(not key.startswith("encoder.") for key in weights)
    assert all(tensor.dtype == torch.float16 for tensor in weights.values())


def test_full_finetune_checkpoint_saves_full_fp16_safetensors(monkeypatch, tmp_path):
    _patch_hf(monkeypatch)
    df = _small_df()
    ckpt = tmp_path / "distilbert.pt"

    train_bert(
        df,
        df,
        epochs=1,
        head_epochs=0,
        freeze_encoder=False,
        batch_size=4,
        max_len=16,
        checkpoint_path=ckpt,
    )

    data = torch.load(ckpt, weights_only=False)
    weights = data["model_state"]

    assert data["save_strategy"] == "full"
    assert any(key.startswith("model.distilbert.") for key in weights)
    assert all(not key.startswith("encoder.") for key in weights)
    assert all(tensor.dtype == torch.float16 for tensor in weights.values())


def test_partial_finetune_checkpoint_saves_last_layers_and_head(monkeypatch, tmp_path):
    _patch_hf(monkeypatch)
    df = _small_df()
    ckpt = tmp_path / "distilbert.pt"

    result = train_bert(
        df,
        df,
        epochs=1,
        head_epochs=0,
        fine_tune_last_n_layers=1,
        batch_size=4,
        max_len=16,
        checkpoint_path=ckpt,
    )

    data = torch.load(ckpt, weights_only=False)
    weights = data["model_state"]

    assert result["history"][0]["stage"] == "fine_tune_last_1"
    assert data["save_strategy"] == "partial_encoder"
    assert data["model_config"]["fine_tune_last_n_layers"] == 1
    assert data["trainable_encoder_layers"] == [0]
    assert any(key.startswith("model.distilbert.transformer.layer.0.") for key in weights)
    assert any(key.startswith("model.classifier.") for key in weights)
    assert any(key.startswith("model.pre_classifier.") for key in weights)
    assert all(not key.startswith("encoder.") for key in weights)
    assert all(not key.startswith("model.distilbert.embeddings.") for key in weights)
    assert all(tensor.dtype == torch.float16 for tensor in weights.values())


def test_train_bert_runs_head_then_fine_tune_stages(monkeypatch, tmp_path):
    _patch_hf(monkeypatch)
    df = _small_df()
    ckpt = tmp_path / "distilbert.pt"

    result = train_bert(
        df,
        df,
        epochs=2,
        head_epochs=1,
        batch_size=4,
        max_len=16,
        checkpoint_path=ckpt,
    )

    stages = [row["stage"] for row in result["history"]]
    data = torch.load(ckpt, weights_only=False)

    assert stages == ["head", "fine_tune"]
    assert data["model_config"]["head_epochs"] == 1
    assert data["model_config"]["fine_tune_epochs"] == 1
    assert "encoder_lr" in data["model_config"]
    assert "classifier_lr" in data["model_config"]


def test_train_bert_checkpoint_can_be_reloaded(monkeypatch, tmp_path):
    _patch_hf(monkeypatch)
    df = _small_df()
    ckpt = tmp_path / "distilbert.pt"

    train_bert(
        df,
        df,
        epochs=1,
        batch_size=4,
        max_len=16,
        checkpoint_path=ckpt,
    )

    model, tokenizer, data, device = load_pretrained_bert_bundle(ckpt)
    encodings = tokenizer(["great product"], max_length=16)
    encodings = {key: value.to(device) for key, value in encodings.items()}

    with torch.no_grad():
        logits = model(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"],
        )

    assert logits.shape == (1,)
    assert data["model_config"]["model_type"] == "pretrained"

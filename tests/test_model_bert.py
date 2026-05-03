# .venv/bin/pytest tests/test_model_bert.py -v

"""Tests for src.models.bert — Hugging Face DistilBERT sentiment model."""

import pytest
pytest.importorskip("transformers")

import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification

import src.models.bert as model_bert
from src.models.bert import DistilBERTSentiment, PretrainedDistilBERTSentiment

VOCAB_SIZE = 100
BATCH = 4
SEQ_LEN = 12


def _tiny_hf_model() -> DistilBertForSequenceClassification:
    config = DistilBertConfig(
        vocab_size=VOCAB_SIZE,
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


def _patch_hf_model(monkeypatch):
    def from_pretrained(cls, *args, **kwargs):
        return _tiny_hf_model()

    monkeypatch.setattr(
        model_bert.DistilBertForSequenceClassification,
        "from_pretrained",
        classmethod(from_pretrained),
    )


def _model(monkeypatch, **kwargs) -> DistilBERTSentiment:
    _patch_hf_model(monkeypatch)
    defaults = dict(dropout=0.0)
    defaults.update(kwargs)
    return DistilBERTSentiment(**defaults)


def _batch(batch=BATCH, seq_len=SEQ_LEN) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(1, VOCAB_SIZE, (batch, seq_len))
    attention_mask = torch.ones_like(tokens)
    return tokens, attention_mask


def test_model_is_nn_module(monkeypatch):
    assert isinstance(_model(monkeypatch), torch.nn.Module)


def test_model_uses_hugging_face_distilbert_classifier(monkeypatch):
    m = _model(monkeypatch)
    assert isinstance(m.model, DistilBertForSequenceClassification)
    assert m.encoder is m.model.distilbert


def test_forward_output_shape(monkeypatch):
    m = _model(monkeypatch)
    m.eval()
    tokens, attention_mask = _batch()
    with torch.no_grad():
        out = m(tokens, attention_mask)
    assert out.shape == (BATCH,)


def test_forward_output_is_finite(monkeypatch):
    m = _model(monkeypatch)
    m.eval()
    tokens, attention_mask = _batch()
    with torch.no_grad():
        out = m(tokens, attention_mask)
    assert torch.all(torch.isfinite(out))


def test_encoder_is_frozen_by_default_and_classifier_trainable(monkeypatch):
    m = _model(monkeypatch)
    assert m.freeze_encoder is True
    assert not any(param.requires_grad for param in m.encoder.parameters())
    assert all(param.requires_grad for param in m.pre_classifier.parameters())
    assert all(param.requires_grad for param in m.classifier.parameters())


def test_can_unfreeze_encoder(monkeypatch):
    m = _model(monkeypatch)
    m.unfreeze_distilbert_encoder()
    assert m.freeze_encoder is False
    assert all(param.requires_grad for param in m.encoder.parameters())


def test_can_refreeze_encoder(monkeypatch):
    m = _model(monkeypatch, freeze_encoder=False)
    m.freeze_distilbert_encoder()
    assert m.freeze_encoder is True
    assert not any(param.requires_grad for param in m.encoder.parameters())


def test_can_unfreeze_only_last_encoder_layers(monkeypatch):
    m = _model(monkeypatch)
    trainable_layers = m.unfreeze_last_encoder_layers(1)

    assert m.freeze_encoder is False
    assert trainable_layers == [0]
    assert any(param.requires_grad for param in m.encoder.transformer.layer[0].parameters())
    assert not any(param.requires_grad for param in m.encoder.embeddings.parameters())


def test_can_initialize_with_encoder_unfrozen(monkeypatch):
    m = _model(monkeypatch, freeze_encoder=False)
    assert any(param.requires_grad for param in m.encoder.parameters())


def test_pretrained_alias_uses_same_implementation(monkeypatch):
    _patch_hf_model(monkeypatch)
    assert PretrainedDistilBERTSentiment is DistilBERTSentiment
    m = PretrainedDistilBERTSentiment(dropout=0.0)
    assert isinstance(m, DistilBERTSentiment)


def test_unexpected_constructor_keyword_raises(monkeypatch):
    _patch_hf_model(monkeypatch)
    try:
        DistilBERTSentiment(freez_encoder=True)
    except TypeError as exc:
        assert "freez_encoder" in str(exc)
    else:
        raise AssertionError("unexpected kwargs should not be accepted")

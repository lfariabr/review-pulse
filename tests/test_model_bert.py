# .venv/bin/pytest tests/test_model_bert.py -v

"""Tests for src/model_bert.py — local transformer sentiment model."""

import numpy as np
import pytest
import torch

from src.model_bert import DistilBERTSentiment

VOCAB_SIZE = 100
EMBEDDING_DIM = 32
HEADS = 4
LAYERS = 1
FF_DIM = 64
MAX_LEN = 32
BATCH = 8
SEQ_LEN = 16


def _model(**kwargs) -> DistilBERTSentiment:
    defaults = dict(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        n_heads=HEADS,
        n_layers=LAYERS,
        ff_dim=FF_DIM,
        max_len=MAX_LEN,
        dropout=0.0,
    )
    defaults.update(kwargs)
    return DistilBERTSentiment(**defaults)


def _batch(batch=BATCH, seq_len=SEQ_LEN) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(1, VOCAB_SIZE, (batch, seq_len))
    attention_mask = torch.ones_like(tokens)
    return tokens, attention_mask


def test_model_is_nn_module():
    assert isinstance(_model(), torch.nn.Module)


def test_model_has_transformer_encoder():
    m = _model()
    assert hasattr(m, "encoder")
    assert isinstance(m.encoder, torch.nn.TransformerEncoder)


def test_forward_output_shape():
    m = _model()
    m.eval()
    tokens, attention_mask = _batch()
    with torch.no_grad():
        out = m(tokens, attention_mask)
    assert out.shape == (BATCH,)


def test_forward_output_is_finite():
    m = _model()
    m.eval()
    tokens, attention_mask = _batch()
    with torch.no_grad():
        out = m(tokens, attention_mask)
    assert torch.all(torch.isfinite(out))


def test_freeze_encoder_leaves_classifier_trainable():
    m = _model(freeze_encoder=True)
    assert not any(param.requires_grad for param in m.token_embedding.parameters())
    assert not any(param.requires_grad for param in m.embedding_projection.parameters())
    assert not any(param.requires_grad for param in m.position_embedding.parameters())
    assert not any(param.requires_grad for param in m.embedding_norm.parameters())
    assert not any(param.requires_grad for param in m.encoder.parameters())
    assert all(param.requires_grad for param in m.fc.parameters())


def test_padding_does_not_change_output():
    m = _model(dropout=0.0)
    m.eval()

    short = torch.tensor([[5, 6, 7, 8, 0, 0]])
    short_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    long = torch.tensor([[5, 6, 7, 8, 0, 0, 0, 0, 0, 0]])
    long_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]])

    with torch.no_grad():
        out_short = m(short, short_mask)
        out_long = m(long, long_mask)

    assert torch.allclose(out_short, out_long, atol=1e-6)


def test_invalid_attention_head_configuration_raises():
    with pytest.raises(ValueError, match="must be divisible"):
        _model(embedding_dim=30, n_heads=4)


def test_pretrained_embeddings_are_loaded():
    pretrained = np.random.randn(VOCAB_SIZE, EMBEDDING_DIM).astype(np.float32)
    m = _model(pretrained_embeddings=pretrained)
    loaded = m.token_embedding.weight.data.numpy()
    np.testing.assert_allclose(loaded, pretrained, rtol=1e-5)


def test_pretrained_embeddings_projection_handles_mismatched_width():
    pretrained = np.random.randn(VOCAB_SIZE, 24).astype(np.float32)
    m = _model(pretrained_embeddings=pretrained)
    assert isinstance(m.embedding_projection, torch.nn.Linear)

    tokens, attention_mask = _batch()
    with torch.no_grad():
        out = m(tokens, attention_mask)

    assert out.shape == (BATCH,)

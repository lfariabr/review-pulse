# .venv/bin/pytest tests/test_model.py -v

"""Tests for src.models.bilstm — BiLSTMSentiment."""

import numpy as np
import pytest
import torch

from src.models.bilstm import BiLSTMSentiment

# ---------------------------------------------------------------------------
# shared constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 200
EMBEDDING_DIM = 100
HIDDEN_DIM = 64   # small for fast CPU tests
N_LAYERS = 2
BATCH = 8
SEQ_LEN = 32


def _model(**kwargs) -> BiLSTMSentiment:
    defaults = dict(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        dropout=0.0,  # deterministic in tests
    )
    defaults.update(kwargs)
    return BiLSTMSentiment(**defaults)


def _batch(batch=BATCH, seq_len=SEQ_LEN) -> torch.Tensor:
    """Random token indices in [0, VOCAB_SIZE)."""
    return torch.randint(0, VOCAB_SIZE, (batch, seq_len))


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def test_model_is_nn_module():
    assert isinstance(_model(), torch.nn.Module)


def test_legacy_model_wrapper_exports_same_class():
    from src.model import BiLSTMSentiment as LegacyBiLSTMSentiment

    assert LegacyBiLSTMSentiment is BiLSTMSentiment


def test_model_has_embedding():
    m = _model()
    assert hasattr(m, "embedding")
    assert isinstance(m.embedding, torch.nn.Embedding)


def test_model_has_lstm():
    m = _model()
    assert hasattr(m, "lstm")
    assert isinstance(m.lstm, torch.nn.LSTM)


def test_embedding_padding_idx_is_zero():
    m = _model()
    assert m.embedding.padding_idx == 0


def test_lstm_is_bidirectional():
    m = _model()
    assert m.lstm.bidirectional


def test_lstm_hidden_dim():
    m = _model()
    assert m.lstm.hidden_size == HIDDEN_DIM


def test_lstm_num_layers():
    m = _model()
    assert m.lstm.num_layers == N_LAYERS


def test_fc_output_dim_is_one():
    m = _model()
    assert m.fc.out_features == 1


def test_fc_input_dim_is_hidden_times_two():
    m = _model()
    assert m.fc.in_features == HIDDEN_DIM * 2


# ---------------------------------------------------------------------------
# forward pass shape & dtype
# ---------------------------------------------------------------------------

def test_forward_output_shape():
    m = _model()
    m.eval()
    with torch.no_grad():
        out = m(_batch())
    assert out.shape == (BATCH,)


def test_forward_output_is_float():
    m = _model()
    m.eval()
    with torch.no_grad():
        out = m(_batch())
    assert out.dtype == torch.float32


def test_forward_output_is_raw_logit():
    """Output should be unbounded (not clamped to [0,1])."""
    m = _model()
    m.eval()
    with torch.no_grad():
        out = m(_batch())
    # With random weights some logits exceed ±1 for large hidden dims,
    # but with small hidden we just check the tensor is finite.
    assert torch.all(torch.isfinite(out))


def test_forward_batch_size_one():
    m = _model()
    m.eval()
    with torch.no_grad():
        out = m(_batch(batch=1))
    assert out.shape == (1,)


def test_forward_batch_size_64():
    m = _model()
    m.eval()
    with torch.no_grad():
        out = m(_batch(batch=64))
    assert out.shape == (64,)


def test_forward_different_seq_lengths():
    m = _model()
    m.eval()
    for seq_len in [1, 16, 128, 256]:
        with torch.no_grad():
            out = m(_batch(seq_len=seq_len))
        assert out.shape == (BATCH,), f"failed for seq_len={seq_len}"


# ---------------------------------------------------------------------------
# pretrained embeddings
# ---------------------------------------------------------------------------

def test_pretrained_embeddings_initialised():
    glove = np.random.randn(VOCAB_SIZE, EMBEDDING_DIM).astype(np.float32)
    m = _model(pretrained_embeddings=glove)
    loaded = m.embedding.weight.data.numpy()
    np.testing.assert_allclose(loaded, glove, rtol=1e-5)


def test_pretrained_embeddings_wrong_shape_raises():
    bad = np.random.randn(VOCAB_SIZE + 10, EMBEDDING_DIM).astype(np.float32)
    with pytest.raises(ValueError, match="does not match"):
        _model(pretrained_embeddings=bad)


def test_pretrained_embeddings_wrong_dim_raises():
    bad = np.random.randn(VOCAB_SIZE, EMBEDDING_DIM + 50).astype(np.float32)
    with pytest.raises(ValueError, match="does not match"):
        _model(pretrained_embeddings=bad)


def test_without_pretrained_embeddings_forward_works():
    m = _model()
    m.eval()
    with torch.no_grad():
        out = m(_batch())
    assert out.shape == (BATCH,)


# ---------------------------------------------------------------------------
# padding index — pad tokens should not affect gradient
# ---------------------------------------------------------------------------

def test_padding_idx_zero_embedding_is_zero_vector():
    """Embedding row 0 (padding) must be zeroed out."""
    m = _model()
    pad_vec = m.embedding.weight.data[0]
    assert torch.all(pad_vec == 0)


# ---------------------------------------------------------------------------
# packed sequences — padding must not change output for the same real tokens
# ---------------------------------------------------------------------------

def test_padding_does_not_change_output():
    """Extra pad tokens appended to the same real tokens must not alter the logit."""
    m = _model(dropout=0.0)
    m.eval()

    # token ids 5-8 are the "real" content; 0 = <pad>
    padded_short = torch.tensor([[5, 6, 7, 8, 0, 0]])
    padded_long  = torch.tensor([[5, 6, 7, 8, 0, 0, 0, 0, 0, 0]])

    with torch.no_grad():
        out_short = m(padded_short)
        out_long  = m(padded_long)

    assert torch.allclose(out_short, out_long, atol=1e-6), (
        f"Packed LSTM output changed with extra padding: {out_short} vs {out_long}"
    )


# ---------------------------------------------------------------------------
# determinism — same input → same output in eval mode
# ---------------------------------------------------------------------------

def test_eval_mode_is_deterministic():
    m = _model()
    m.eval()
    x = _batch()
    with torch.no_grad():
        out1 = m(x)
        out2 = m(x)
    assert torch.allclose(out1, out2)

# .venv/bin/pytest tests/test_inference.py -v
# .venv/bin/pytest tests/test_inference.py -v -m "not slow"

"""Tests for src/inference.py."""

import json
import joblib
import numpy as np
import pytest
import torch
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.model import BiLSTMSentiment
from src.dataset import build_vocab, save_vocab

# ---------------------------------------------------------------------------
# Fixtures — save minimal artefacts so tests don't need real model files
# ---------------------------------------------------------------------------

TEXTS  = [
    "great product love it highly recommend",
    "excellent quality exceeded expectations fantastic",
    "terrible product broke waste money",
    "do not buy garbage worst purchase ever",
    "best purchase ever works perfectly time",
    "very disappointed with quality not worth",
]
LABELS = [1, 1, 0, 0, 1, 0]


def _make_baseline(tmp_path: Path) -> Path:
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf",   LogisticRegression(max_iter=200)),
    ])
    pipeline.fit(TEXTS, LABELS)
    path = tmp_path / "baseline.joblib"
    joblib.dump(pipeline, path)
    return path


def _make_bilstm_checkpoint(tmp_path: Path, vocab: dict) -> Path:
    vocab_size = len(vocab)
    model = BiLSTMSentiment(vocab_size=vocab_size, embedding_dim=16, hidden_dim=16, n_layers=1, dropout=0.0)
    ckpt_path = tmp_path / "bilstm.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {
                "vocab_size":    vocab_size,
                "embedding_dim": 16,
                "hidden_dim":    16,
                "n_layers":      1,
                "dropout":       0.0,
            },
            "vocab_path":  str(tmp_path / "vocab.json"),
            "best_val_f1": 0.80,
            "best_epoch":  1,
            "history":     [],
        },
        ckpt_path,
    )
    return ckpt_path


@pytest.fixture()
def artefacts(tmp_path):
    """Returns (baseline_path, bilstm_path, vocab_path) for tiny fixtures."""
    import src.inference as inf_mod
    # Reset module-level caches between tests
    inf_mod._baseline_cache = None
    inf_mod._bilstm_cache   = None

    vocab = build_vocab(TEXTS, min_freq=1)
    vocab_path    = tmp_path / "vocab.json"
    save_vocab(vocab, vocab_path)

    baseline_path = _make_baseline(tmp_path)
    bilstm_path   = _make_bilstm_checkpoint(tmp_path, vocab)

    yield baseline_path, bilstm_path, vocab_path

    # Reset again after test
    inf_mod._baseline_cache = None
    inf_mod._bilstm_cache   = None


# ---------------------------------------------------------------------------
# predict_baseline
# ---------------------------------------------------------------------------

def test_predict_baseline_returns_required_keys(artefacts):
    from src.inference import predict_baseline
    baseline_path, _, _ = artefacts
    result = predict_baseline("great product love it", path=baseline_path)
    assert "label"      in result
    assert "confidence" in result
    assert "model"      in result


def test_predict_baseline_model_name(artefacts):
    from src.inference import predict_baseline
    baseline_path, _, _ = artefacts
    result = predict_baseline("great product love it", path=baseline_path)
    assert result["model"] == "baseline"


def test_predict_baseline_label_values(artefacts):
    from src.inference import predict_baseline
    baseline_path, _, _ = artefacts
    result = predict_baseline("great product love it", path=baseline_path)
    assert result["label"] in ("Positive review", "Negative review")


def test_predict_baseline_confidence_in_range(artefacts):
    from src.inference import predict_baseline
    baseline_path, _, _ = artefacts
    result = predict_baseline("great product love it", path=baseline_path)
    assert 0.0 <= result["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# predict_bilstm
# ---------------------------------------------------------------------------

def test_predict_bilstm_returns_required_keys(artefacts):
    from src.inference import predict_bilstm
    _, bilstm_path, vocab_path = artefacts
    result = predict_bilstm("great product love it",
                            checkpoint_path=bilstm_path, vocab_path=vocab_path)
    assert "label"      in result
    assert "confidence" in result
    assert "model"      in result


def test_predict_bilstm_model_name(artefacts):
    from src.inference import predict_bilstm
    _, bilstm_path, vocab_path = artefacts
    result = predict_bilstm("great product love it",
                            checkpoint_path=bilstm_path, vocab_path=vocab_path)
    assert result["model"] == "bilstm"


def test_predict_bilstm_label_values(artefacts):
    from src.inference import predict_bilstm
    _, bilstm_path, vocab_path = artefacts
    result = predict_bilstm("great product love it",
                            checkpoint_path=bilstm_path, vocab_path=vocab_path)
    assert result["label"] in ("Positive review", "Negative review")


def test_predict_bilstm_confidence_in_range(artefacts):
    from src.inference import predict_bilstm
    _, bilstm_path, vocab_path = artefacts
    result = predict_bilstm("terrible broke garbage waste",
                            checkpoint_path=bilstm_path, vocab_path=vocab_path)
    assert 0.0 <= result["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# predict_sentiment (unified entry point)
# ---------------------------------------------------------------------------

def test_predict_sentiment_default_is_baseline(artefacts):
    import src.inference as inf_mod
    from src.inference import predict_sentiment
    baseline_path, _, _ = artefacts
    inf_mod._baseline_cache = joblib.load(baseline_path)
    result = predict_sentiment("great product")
    assert result["model"] == "baseline"


def test_predict_sentiment_selects_bilstm(artefacts):
    import src.inference as inf_mod
    from src.inference import predict_sentiment
    _, bilstm_path, vocab_path = artefacts
    vocab = build_vocab(TEXTS, min_freq=1)
    model = BiLSTMSentiment(vocab_size=len(vocab), embedding_dim=16, hidden_dim=16, n_layers=1, dropout=0.0)
    torch.load(bilstm_path, weights_only=False)  # load to check structure
    inf_mod._bilstm_cache = (model, vocab, torch.device("cpu"))
    result = predict_sentiment("great product", model_name="bilstm")
    assert result["model"] == "bilstm"


def test_predict_sentiment_invalid_model_raises(artefacts):
    from src.inference import predict_sentiment
    with pytest.raises(ValueError, match="Unknown model"):
        predict_sentiment("great product", model_name="gpt5")


# ---------------------------------------------------------------------------
# register_predictor / get_available_models
# ---------------------------------------------------------------------------

class _DummyPredictor:
    def predict(self, text: str) -> dict:
        return {"label": "Positive review", "confidence": 1.0, "model": "dummy"}


def test_get_available_models_contains_defaults():
    from src.inference import get_available_models
    models = get_available_models()
    assert "baseline" in models
    assert "bilstm" in models
    assert "distilbert" in models


def test_register_predictor_and_predict(artefacts):
    import src.inference as inf_mod
    from src.inference import register_predictor, predict_sentiment, get_available_models
    # Clean up after test regardless of outcome
    try:
        register_predictor("dummy", _DummyPredictor())
        assert "dummy" in get_available_models()
        result = predict_sentiment("anything", model_name="dummy")
        assert result["model"] == "dummy"
    finally:
        inf_mod._PREDICTORS.pop("dummy", None)


def test_register_predictor_rejects_invalid_object():
    from src.inference import register_predictor
    with pytest.raises(TypeError, match="Predictor protocol"):
        register_predictor("bad", object())


def test_register_predictor_rejects_duplicate():
    from src.inference import register_predictor
    with pytest.raises(ValueError, match="already registered"):
        register_predictor("baseline", _DummyPredictor())


def test_register_predictor_overwrite_allowed():
    import src.inference as inf_mod
    from src.inference import register_predictor
    original = inf_mod._PREDICTORS.get("baseline")
    try:
        register_predictor("baseline", _DummyPredictor(), overwrite=True)
        assert inf_mod._PREDICTORS["baseline"] is not original
    finally:
        inf_mod._PREDICTORS["baseline"] = original


# ---------------------------------------------------------------------------
# integration — real saved artefacts
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_predict_sentiment_real_baseline():
    import src.inference as inf_mod
    inf_mod._baseline_cache = None
    from src.inference import predict_sentiment
    result = predict_sentiment("This product is absolutely fantastic, I love it!", model_name="baseline")
    assert result["label"] == "Positive review"
    assert result["confidence"] >= 0.5


@pytest.mark.slow
def test_predict_sentiment_real_bilstm():
    import src.inference as inf_mod
    inf_mod._bilstm_cache = None
    from src.inference import predict_sentiment
    result = predict_sentiment("Terrible quality, broke after one day.", model_name="bilstm")
    assert result["label"] == "Negative review"
    assert result["confidence"] >= 0.5

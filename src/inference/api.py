"""Compatibility-facing inference API."""

from pathlib import Path
from typing import Optional

from src.config import MODEL_BASELINE

from . import loaders
from .registry import _BILSTM_PREDICTOR, _DISTILBERT_PREDICTOR, _PREDICTORS


def _sync_loader_caches_from_package() -> None:
    """Reflect legacy ``src.inference._*_cache`` assignments into loaders."""
    import src.inference as package

    loaders._baseline_cache = getattr(package, "_baseline_cache", loaders._baseline_cache)
    loaders._bilstm_cache = getattr(package, "_bilstm_cache", loaders._bilstm_cache)
    loaders._distilbert_cache = getattr(
        package, "_distilbert_cache", loaders._distilbert_cache
    )


def _sync_loader_caches_to_package() -> None:
    """Expose loader cache mutations through legacy package attributes."""
    import src.inference as package

    package._baseline_cache = loaders._baseline_cache
    package._bilstm_cache = loaders._bilstm_cache
    package._distilbert_cache = loaders._distilbert_cache


def _get_baseline_predictor():
    import src.inference as package

    return getattr(package, "_BASELINE_PREDICTOR")


def get_available_models() -> tuple[str, ...]:
    """Return the names of all currently registered models."""
    return tuple(_PREDICTORS.keys())


def predict_baseline(text: str, path: Optional[Path] = None) -> dict:
    """Predict sentiment using the TF-IDF baseline."""
    _sync_loader_caches_from_package()
    result = _get_baseline_predictor().predict(text, path=path)
    _sync_loader_caches_to_package()
    return result


def predict_bilstm(
    text: str,
    checkpoint_path: Optional[Path] = None,
    vocab_path: Optional[Path] = None,
) -> dict:
    """Predict sentiment using the BiLSTM model."""
    _sync_loader_caches_from_package()
    result = _BILSTM_PREDICTOR.predict(
        text, checkpoint_path=checkpoint_path, vocab_path=vocab_path
    )
    _sync_loader_caches_to_package()
    return result


def predict_distilbert(text: str, checkpoint_path: Optional[Path] = None) -> dict:
    """Predict sentiment using the deployed DistilBERT model."""
    _sync_loader_caches_from_package()
    result = _DISTILBERT_PREDICTOR.predict(text, checkpoint_path=checkpoint_path)
    _sync_loader_caches_to_package()
    return result


def predict_sentiment(text: str, model_name: str = MODEL_BASELINE) -> dict:
    """Predict the sentiment of a review."""
    if model_name not in _PREDICTORS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {get_available_models()}."
        )

    if model_name == MODEL_BASELINE:
        return predict_baseline(text)

    _sync_loader_caches_from_package()
    result = _PREDICTORS[model_name].predict(text)
    _sync_loader_caches_to_package()
    return result

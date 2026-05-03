"""Predictor registry for inference models."""

from src.config import MODEL_BASELINE, MODEL_BILSTM, MODEL_DISTILBERT

from .predictors import (
    BaselinePredictor,
    BiLSTMPredictor,
    DistilBERTPredictor,
    Predictor,
)

_BASELINE_PREDICTOR = BaselinePredictor()
_BILSTM_PREDICTOR = BiLSTMPredictor()
_DISTILBERT_PREDICTOR = DistilBERTPredictor()

_PREDICTORS: dict[str, Predictor] = {
    MODEL_BASELINE: _BASELINE_PREDICTOR,
    MODEL_BILSTM: _BILSTM_PREDICTOR,
    MODEL_DISTILBERT: _DISTILBERT_PREDICTOR,
}


def register_predictor(
    name: str, predictor: Predictor, *, overwrite: bool = False
) -> None:
    """Register a predictor under the given model name."""
    if not isinstance(predictor, Predictor):
        raise TypeError(
            "predictor must implement the Predictor protocol "
            f"(needs a predict() method), got {type(predictor).__name__!r}"
        )
    if name in _PREDICTORS and not overwrite:
        raise ValueError(
            f"Predictor '{name}' is already registered. "
            "Pass overwrite=True to replace it."
        )
    _PREDICTORS[name] = predictor


def get_available_models() -> tuple[str, ...]:
    """Return the names of all currently registered models."""
    return tuple(_PREDICTORS.keys())

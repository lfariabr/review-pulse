"""Public compatibility surface for ``src.inference``.

The implementation lives in responsibility-focused modules under this package,
while this module keeps the historical ``from src.inference import ...`` API.
"""

from . import loaders as _loaders
from .api import (
    predict_baseline,
    predict_bilstm,
    predict_distilbert,
    predict_sentiment,
)
from .loaders import (
    CHECKPOINT_PATH,
    DEPLOY_CHECKPOINT_PATH,
    load_baseline_model,
    load_bilstm_model,
    load_checkpoint,
    load_distilbert_model,
)
from .predictors import (
    BaselinePredictor,
    BiLSTMPredictor,
    DistilBERTPredictor,
    Predictor,
)
from .registry import (
    _BASELINE_PREDICTOR,
    _BILSTM_PREDICTOR,
    _DISTILBERT_PREDICTOR,
    _PREDICTORS,
    get_available_models,
    register_predictor,
)

_baseline_cache = _loaders._baseline_cache
_bilstm_cache = _loaders._bilstm_cache
_distilbert_cache = _loaders._distilbert_cache

__all__ = [
    "CHECKPOINT_PATH",
    "DEPLOY_CHECKPOINT_PATH",
    "Predictor",
    "BaselinePredictor",
    "BiLSTMPredictor",
    "DistilBERTPredictor",
    "_BASELINE_PREDICTOR",
    "_BILSTM_PREDICTOR",
    "_DISTILBERT_PREDICTOR",
    "_PREDICTORS",
    "_baseline_cache",
    "_bilstm_cache",
    "_distilbert_cache",
    "load_checkpoint",
    "load_baseline_model",
    "load_bilstm_model",
    "load_distilbert_model",
    "register_predictor",
    "get_available_models",
    "predict_baseline",
    "predict_bilstm",
    "predict_distilbert",
    "predict_sentiment",
]

"""Compatibility wrapper for the ``src.inference`` package.

The implementation now lives in:

- ``src.inference.loaders``
- ``src.inference.predictors``
- ``src.inference.registry``
- ``src.inference.api``
"""

from src.inference import (
    CHECKPOINT_PATH,
    DEPLOY_CHECKPOINT_PATH,
    BaselinePredictor,
    BiLSTMPredictor,
    DistilBERTPredictor,
    Predictor,
    _BASELINE_PREDICTOR,
    _BILSTM_PREDICTOR,
    _DISTILBERT_PREDICTOR,
    _PREDICTORS,
    _baseline_cache,
    _bilstm_cache,
    _distilbert_cache,
    get_available_models,
    load_baseline_model,
    load_bilstm_model,
    load_checkpoint,
    load_distilbert_model,
    predict_baseline,
    predict_bilstm,
    predict_distilbert,
    predict_sentiment,
    register_predictor,
)

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

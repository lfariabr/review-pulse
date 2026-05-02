"""Inference module for ReviewPulse.

Provides predict_sentiment() — a single entry point used by both app.py
and evaluate.py. Supports three models:

    "baseline"   TF-IDF + Logistic Regression  (default — better test F1)
    "bilstm"     GloVe + BiLSTM                (satisfies neural-net requirement)
    "distilbert" Hugging Face DistilBERT        (deployment transformer)

Result shape:
    {
        "label":      "Positive review" | "Negative review",
        "confidence": float,   # probability of the predicted class
        "model":      "baseline" | "bilstm" | "distilbert",
    }
"""

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import torch

from src.config import (
    ALL_MODELS,
    BASELINE_PATH,
    BILSTM_CHECKPOINT_PATH,
    DISTILBERT_PATH,
    MODEL_BASELINE,
    MODEL_BILSTM,
    MODEL_DISTILBERT,
    PRED_THRESHOLD,
    VOCAB_PATH,
)
from src.dataset import load_vocab
from src.models.bilstm import BiLSTMSentiment
from src.preprocess import clean_text

CHECKPOINT_PATH        = BILSTM_CHECKPOINT_PATH
DEPLOY_CHECKPOINT_PATH = DISTILBERT_PATH


# ---------------------------------------------------------------------------
# Checkpoint loading (lives here so evaluate.py doesn't pull in matplotlib)
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> tuple:
    """Load a saved BiLSTM checkpoint and reconstruct the model.

    Returns:
        (model, model_config, history) — model is in eval mode on device.
    """
    checkpoint_path = checkpoint_path or CHECKPOINT_PATH
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt["model_config"]

    model = BiLSTMSentiment(
        vocab_size=cfg["vocab_size"],
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"load_checkpoint: loaded epoch {ckpt['best_epoch']} "
          f"(val_f1={ckpt['best_val_f1']:.4f}) ← {checkpoint_path}")
    return model, cfg, ckpt.get("history", [])


# ---------------------------------------------------------------------------
# Model loaders (cached at module level after first call)
# ---------------------------------------------------------------------------

_baseline_cache   = None
_bilstm_cache     = None   # (model, vocab, device)
_distilbert_cache = None   # (model, tokenizer, checkpoint, device)


def load_baseline_model(path: Optional[Path] = None):
    """Load and cache the TF-IDF + LogReg pipeline."""
    global _baseline_cache
    if _baseline_cache is None:
        from src.models.baseline import load_baseline
        _baseline_cache = load_baseline(path or BASELINE_PATH)
    return _baseline_cache


def load_bilstm_model(
    checkpoint_path: Optional[Path] = None,
    vocab_path: Optional[Path] = None,
):
    """Load and cache the BiLSTM model + vocab."""
    global _bilstm_cache
    if _bilstm_cache is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        model, _, _ = load_checkpoint(checkpoint_path or CHECKPOINT_PATH, device)
        vocab = load_vocab(vocab_path or VOCAB_PATH)
        _bilstm_cache = (model, vocab, device)
    return _bilstm_cache


def load_distilbert_model(checkpoint_path: Optional[Path] = None):
    """Load and cache the deployed Hugging Face DistilBERT bundle."""
    global _distilbert_cache
    if _distilbert_cache is None:
        from src.train_bert import load_pretrained_bert_bundle

        _distilbert_cache = load_pretrained_bert_bundle(
            checkpoint_path or DEPLOY_CHECKPOINT_PATH
        )
    return _distilbert_cache


# ---------------------------------------------------------------------------
# Predictor protocol + per-model implementations
# ---------------------------------------------------------------------------

@runtime_checkable
class Predictor(Protocol):
    """Single-text sentiment predictor interface."""

    def predict(self, text: str) -> dict:
        """Return {"label": str, "confidence": float, "model": str}."""
        ...


class BaselinePredictor:
    """TF-IDF + Logistic Regression predictor."""

    def predict(self, text: str, path: Optional[Path] = None) -> dict:
        pipeline  = load_baseline_model(path)
        cleaned   = clean_text(text)
        proba     = pipeline.predict_proba([cleaned])[0]
        pred_idx  = int(proba.argmax())
        confidence = float(proba[pred_idx])
        label     = "Positive review" if pred_idx == 1 else "Negative review"
        return {"label": label, "confidence": round(confidence, 4), "model": MODEL_BASELINE}


class BiLSTMPredictor:
    """BiLSTM + GloVe predictor."""

    def predict(
        self,
        text: str,
        checkpoint_path: Optional[Path] = None,
        vocab_path: Optional[Path] = None,
    ) -> dict:
        from src.dataset import tokenize_and_pad, MAX_LEN

        model, vocab, device = load_bilstm_model(checkpoint_path, vocab_path)
        cleaned = clean_text(text)
        tokens  = tokenize_and_pad([cleaned], vocab, max_len=MAX_LEN).to(device)

        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(tokens)).item()

        pred_idx   = int(prob >= PRED_THRESHOLD)
        confidence = prob if pred_idx == 1 else 1.0 - prob
        label      = "Positive review" if pred_idx == 1 else "Negative review"
        return {"label": label, "confidence": round(confidence, 4), "model": MODEL_BILSTM}


class DistilBERTPredictor:
    """Hugging Face DistilBERT predictor."""

    def predict(self, text: str, checkpoint_path: Optional[Path] = None) -> dict:
        model, tokenizer, checkpoint, device = load_distilbert_model(checkpoint_path)
        cleaned = clean_text(text)
        max_len = int(checkpoint.get("model_config", {}).get("max_len", 256))

        encoded = tokenizer(
            [cleaned],
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(
                model(input_ids=encoded["input_ids"],
                      attention_mask=encoded["attention_mask"])
            ).item()

        pred_idx   = int(prob >= PRED_THRESHOLD)
        confidence = prob if pred_idx == 1 else 1.0 - prob
        label      = "Positive review" if pred_idx == 1 else "Negative review"
        return {"label": label, "confidence": round(confidence, 4), "model": MODEL_DISTILBERT}


# Named concrete instances — used by both the registry and the compat delegates
_BASELINE_PREDICTOR   = BaselinePredictor()
_BILSTM_PREDICTOR     = BiLSTMPredictor()
_DISTILBERT_PREDICTOR = DistilBERTPredictor()

# Registry — typed purely against the Protocol's single-text interface
_PREDICTORS: dict[str, Predictor] = {
    MODEL_BASELINE:   _BASELINE_PREDICTOR,
    MODEL_BILSTM:     _BILSTM_PREDICTOR,
    MODEL_DISTILBERT: _DISTILBERT_PREDICTOR,
}


def register_predictor(
    name: str, predictor: Predictor, *, overwrite: bool = False
) -> None:
    """Register a new predictor under the given model name.

    Allows adding future models (e.g. RoBERTa) without editing this module:

        from src.inference import register_predictor
        register_predictor("roberta", RoBERTaPredictor())

    Raises:
        TypeError:  If predictor does not implement the Predictor protocol.
        ValueError: If name is already registered and overwrite=False.
    """
    if not isinstance(predictor, Predictor):
        raise TypeError(
            f"predictor must implement the Predictor protocol (needs a predict() method), "
            f"got {type(predictor).__name__!r}"
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


# ---------------------------------------------------------------------------
# Backward-compat flat functions (call concrete instances directly)
# ---------------------------------------------------------------------------

def predict_baseline(text: str, path: Optional[Path] = None) -> dict:
    """Predict sentiment using the TF-IDF baseline."""
    return _BASELINE_PREDICTOR.predict(text, path=path)


def predict_bilstm(
    text: str,
    checkpoint_path: Optional[Path] = None,
    vocab_path: Optional[Path] = None,
) -> dict:
    """Predict sentiment using the BiLSTM model."""
    return _BILSTM_PREDICTOR.predict(
        text, checkpoint_path=checkpoint_path, vocab_path=vocab_path
    )


def predict_distilbert(text: str, checkpoint_path: Optional[Path] = None) -> dict:
    """Predict sentiment using the deployed DistilBERT model."""
    return _DISTILBERT_PREDICTOR.predict(text, checkpoint_path=checkpoint_path)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def predict_sentiment(text: str, model_name: str = MODEL_BASELINE) -> dict:
    """Predict the sentiment of a review.

    Args:
        text:       Raw review text.
        model_name: "baseline" (default), "bilstm", or "distilbert".

    Returns:
        {"label": "Positive review" | "Negative review",
         "confidence": float, "model": str}

    Raises:
        ValueError: If model_name is not recognised.
    """
    if model_name not in _PREDICTORS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {get_available_models()}."
        )
    return _PREDICTORS[model_name].predict(text)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples = [
        "This product is absolutely fantastic, I love it!",
        "Terrible quality, broke after one day. Complete waste of money.",
        "Not bad, but not great either.",
    ]

    for text in samples:
        for model_name in ALL_MODELS:
            result = predict_sentiment(text, model_name=model_name)
            print(f"[{result['model']:>8}] {result['label']} "
                  f"({result['confidence']:.1%}) | {text[:60]}")
        print()

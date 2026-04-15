"""Inference module for ReviewPulse.

Provides predict_sentiment() — a single entry point used by both app.py
and evaluate.py. Supports two models:

    "baseline"  TF-IDF + Logistic Regression  (default — better test F1)
    "bilstm"    GloVe + BiLSTM                (satisfies neural-net requirement)

Result shape:
    {
        "label":      "Positive review" | "Negative review",
        "confidence": float,   # probability of the predicted class
        "model":      "baseline" | "bilstm",
    }
"""

from pathlib import Path
from typing import Optional

import torch

from src.dataset import OUTPUTS_DIR, load_vocab
from src.model import BiLSTMSentiment
from src.preprocess import clean_text

CHECKPOINT_PATH = OUTPUTS_DIR / "bilstm.pt"
VOCAB_PATH      = OUTPUTS_DIR / "vocab.json"
BASELINE_PATH   = OUTPUTS_DIR / "baseline.joblib"


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

_baseline_cache = None
_bilstm_cache   = None   # (model, vocab, device)


def load_baseline_model(path: Optional[Path] = None):
    """Load and cache the TF-IDF + LogReg pipeline."""
    global _baseline_cache
    if _baseline_cache is None:
        from src.baseline import load_baseline
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


# ---------------------------------------------------------------------------
# Per-model prediction helpers
# ---------------------------------------------------------------------------

def predict_baseline(text: str, path: Optional[Path] = None) -> dict:
    """Predict sentiment using the TF-IDF baseline.

    Args:
        text: Raw review text (cleaning applied internally).
        path: Optional path to a saved baseline.joblib.

    Returns:
        {"label": str, "confidence": float, "model": "baseline"}
    """
    pipeline = load_baseline_model(path)
    cleaned  = clean_text(text)

    proba     = pipeline.predict_proba([cleaned])[0]   # [neg_prob, pos_prob]
    pred_idx  = int(proba.argmax())
    confidence = float(proba[pred_idx])
    label      = "Positive review" if pred_idx == 1 else "Negative review"

    return {"label": label, "confidence": round(confidence, 4), "model": "baseline"}


def predict_bilstm(
    text: str,
    checkpoint_path: Optional[Path] = None,
    vocab_path: Optional[Path] = None,
) -> dict:
    """Predict sentiment using the BiLSTM model.

    Args:
        text:            Raw review text (cleaning applied internally).
        checkpoint_path: Optional path to bilstm.pt.
        vocab_path:      Optional path to vocab.json.

    Returns:
        {"label": str, "confidence": float, "model": "bilstm"}
    """
    from src.dataset import tokenize_and_pad, MAX_LEN

    model, vocab, device = load_bilstm_model(checkpoint_path, vocab_path)
    cleaned = clean_text(text)

    tokens = tokenize_and_pad([cleaned], vocab, max_len=MAX_LEN).to(device)

    model.eval()
    with torch.no_grad():
        logit = model(tokens)                          # (1,)
        prob  = torch.sigmoid(logit).item()

    pred_idx   = int(prob >= 0.5)
    confidence = prob if pred_idx == 1 else 1.0 - prob
    label      = "Positive review" if pred_idx == 1 else "Negative review"

    return {"label": label, "confidence": round(confidence, 4), "model": "bilstm"}


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def predict_sentiment(text: str, model_name: str = "baseline") -> dict:
    """Predict the sentiment of a review.

    Args:
        text:       Raw review text.
        model_name: "baseline" (default) or "bilstm".

    Returns:
        {"label": "Positive review" | "Negative review",
         "confidence": float, "model": str}

    Raises:
        ValueError: If model_name is not recognised.
    """
    if model_name == "baseline":
        return predict_baseline(text)
    elif model_name == "bilstm":
        return predict_bilstm(text)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose 'baseline' or 'bilstm'.")


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
        for model_name in ("baseline", "bilstm"):
            result = predict_sentiment(text, model_name=model_name)
            print(f"[{result['model']:>8}] {result['label']} "
                  f"({result['confidence']:.1%}) | {text[:60]}")
        print()

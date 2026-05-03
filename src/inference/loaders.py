"""Artifact and model loading for ReviewPulse inference."""

from pathlib import Path
from typing import Optional

import joblib
import torch

from src.checkpoint_bert import load_pretrained_bert_bundle
from src.config import (
    BASELINE_PATH,
    BILSTM_CHECKPOINT_PATH,
    DISTILBERT_PATH,
    VOCAB_PATH,
)
from src.models.bilstm import BiLSTMSentiment
from src.tokenization.vocab import load_vocab

CHECKPOINT_PATH = BILSTM_CHECKPOINT_PATH
DEPLOY_CHECKPOINT_PATH = DISTILBERT_PATH

_baseline_cache = None
_bilstm_cache = None
_distilbert_cache = None


def resolve_device() -> torch.device:
    """Return the best available torch device for local inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(
    checkpoint_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> tuple:
    """Load a saved BiLSTM checkpoint and reconstruct the model.

    Returns:
        (model, model_config, history) with the model in eval mode on device.
    """
    checkpoint_path = checkpoint_path or CHECKPOINT_PATH
    device = device or resolve_device()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]

    model = BiLSTMSentiment(
        vocab_size=cfg["vocab_size"],
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(
        f"load_checkpoint: loaded epoch {ckpt['best_epoch']} "
        f"(val_f1={ckpt['best_val_f1']:.4f}) ← {checkpoint_path}"
    )
    return model, cfg, ckpt.get("history", [])


def load_baseline_model(path: Optional[Path] = None):
    """Load and cache the TF-IDF + LogReg pipeline."""
    global _baseline_cache
    if _baseline_cache is None:
        model_path = path or BASELINE_PATH
        _baseline_cache = joblib.load(model_path)
        print(f"load_baseline: loaded ← {model_path}")
    return _baseline_cache


def load_bilstm_model(
    checkpoint_path: Optional[Path] = None,
    vocab_path: Optional[Path] = None,
):
    """Load and cache the BiLSTM model + vocab."""
    global _bilstm_cache
    if _bilstm_cache is None:
        device = resolve_device()
        model, _, _ = load_checkpoint(checkpoint_path or CHECKPOINT_PATH, device)
        vocab = load_vocab(vocab_path or VOCAB_PATH)
        _bilstm_cache = (model, vocab, device)
    return _bilstm_cache


def load_distilbert_model(checkpoint_path: Optional[Path] = None):
    """Load and cache the deployed Hugging Face DistilBERT bundle."""
    global _distilbert_cache
    if _distilbert_cache is None:
        _distilbert_cache = load_pretrained_bert_bundle(
            checkpoint_path or DEPLOY_CHECKPOINT_PATH
        )
    return _distilbert_cache

"""DistilBERT checkpoint save and load helpers for ReviewPulse.

Handles FP16 checkpoint serialization for deployment and bundle loading
used by both inference.py and evaluate.py.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch

from src.config import DISTILBERT_PATH
from src.dataset_bert import LOCAL_FILES_ONLY, resolve_device
from src.model_bert import (
    BERT_DROPOUT,
    PRETRAINED_DISTILBERT_MODEL_NAME,
    DistilBERTSentiment,
)

LOGGER = logging.getLogger(__name__)

DEPLOY_CHECKPOINT_PATH = DISTILBERT_PATH
PRETRAINED_MODEL_NAME  = PRETRAINED_DISTILBERT_MODEL_NAME
DROPOUT                = BERT_DROPOUT
FREEZE_ENCODER         = True


# ---------------------------------------------------------------------------
# Tokenizer serialization helpers
# ---------------------------------------------------------------------------

def _serialize_tokenizer(tokenizer: Any) -> dict[str, bytes] | None:
    """Serialize Hugging Face tokenizer files into a checkpoint-safe payload."""
    if tokenizer is None or not hasattr(tokenizer, "save_pretrained"):
        return None

    with tempfile.TemporaryDirectory() as tmp:
        tokenizer.save_pretrained(tmp)
        tmp_path = Path(tmp)
        return {
            path.name: path.read_bytes()
            for path in tmp_path.iterdir()
            if path.is_file()
        }


def _load_tokenizer_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    model_name: str,
    local_files_only: bool,
):
    """Load an embedded tokenizer payload, falling back to Hugging Face."""
    from src.dataset_bert import load_tokenizer

    tokenizer_files = checkpoint.get("tokenizer_files")
    if tokenizer_files:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for name, data in tokenizer_files.items():
                (tmp_path / name).write_bytes(data)
            return load_tokenizer(model_name=str(tmp_path), local_files_only=True)

    return load_tokenizer(
        model_name=checkpoint.get("tokenizer_name", model_name),
        local_files_only=local_files_only,
    )


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------

def _trainable_encoder_layer_indexes(model: DistilBERTSentiment) -> list[int]:
    """Return DistilBERT transformer layer indexes with trainable parameters."""
    return [
        idx
        for idx, layer in enumerate(model.encoder.transformer.layer)
        if any(param.requires_grad for param in layer.parameters())
    ]


def _save_checkpoint(
    *,
    checkpoint_path: Path,
    model: DistilBERTSentiment,
    tokenizer_files: dict[str, bytes] | None,
    model_config: dict[str, Any],
    tokenizer_name: str,
    history: list[dict[str, Any]],
    best_val_f1: float,
    best_epoch: int,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Save fp16_state_dict for deployment inference, not resumable training.

    Casts weights to FP16 and omits optimizer state. Not suitable for
    resume_from-style training — use only for inference deployment.
    """
    checkpoint_path = Path(checkpoint_path)
    state_dict = model.state_dict()
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("encoder.")
    }

    trainable_encoder_layer_indexes = _trainable_encoder_layer_indexes(model)
    encoder_is_frozen    = not any(param.requires_grad for param in model.encoder.parameters())
    all_encoder_trainable = all(param.requires_grad for param in model.encoder.parameters())

    if encoder_is_frozen:
        save_strategy = "head_only"
    elif all_encoder_trainable:
        save_strategy = "full"
    else:
        save_strategy = "partial_encoder"

    if save_strategy == "head_only":
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("model.distilbert.")
        }
    elif save_strategy == "partial_encoder":
        trainable_prefixes = tuple(
            f"model.distilbert.transformer.layer.{idx}."
            for idx in trainable_encoder_layer_indexes
        )
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("model.distilbert.")
            or key.startswith(trainable_prefixes)
        }

    fp16_state_dict = {
        key: value.detach().cpu().to(torch.float16)
        for key, value in state_dict.items()
    }

    payload = {
        "model_config":              model_config,
        "tokenizer_name":            tokenizer_name,
        "tokenizer_files":           tokenizer_files,
        "best_val_f1":               best_val_f1,
        "best_epoch":                best_epoch,
        "history":                   history,
        "model_state":               fp16_state_dict,
        "weights_format":            "torch_state_dict",
        "weights_dtype":             "float16",
        "save_strategy":             save_strategy,
        "trainable_encoder_layers":  trainable_encoder_layer_indexes,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, checkpoint_path)


# ---------------------------------------------------------------------------
# Checkpoint load (used by inference.py and evaluate.py)
# ---------------------------------------------------------------------------

def load_pretrained_bert_bundle(
    checkpoint_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
):
    """Load a saved Hugging Face DistilBERT checkpoint and tokenizer."""
    device = resolve_device(device)
    checkpoint_path = Path(checkpoint_path or DEPLOY_CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"DistilBERT checkpoint not found at {checkpoint_path}. "
            "Run `train_bert(...)` first."
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt["model_config"]
    local_files_only = cfg.get("local_files_only", LOCAL_FILES_ONLY)
    model_name       = cfg.get("model_name", PRETRAINED_MODEL_NAME)

    model = DistilBERTSentiment(
        model_name=model_name,
        dropout=cfg.get("dropout", DROPOUT),
        freeze_encoder=cfg.get("freeze_encoder", FREEZE_ENCODER),
        local_files_only=local_files_only,
    ).to(device)

    # head_only / partial_encoder checkpoints omit frozen encoder weights;
    # the HF model supplies them. Full checkpoints must match exactly.
    strict = ckpt.get("save_strategy") not in ("head_only", "partial_encoder")
    missing_keys, unexpected_keys = model.load_state_dict(
        ckpt["model_state"], strict=strict
    )
    if missing_keys or unexpected_keys:
        LOGGER.debug(
            "load_pretrained_bert_bundle diagnostics: "
            "missing_keys=%s unexpected_keys=%s",
            missing_keys, unexpected_keys,
        )
    model.eval()

    tokenizer = _load_tokenizer_from_checkpoint(
        ckpt,
        model_name=model_name,
        local_files_only=local_files_only,
    )
    return model, tokenizer, ckpt, device

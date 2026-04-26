"""
Train Hugging Face DistilBERT for Amazon review sentiment analysis.

By default, training freezes the DistilBERT encoder and trains only the classifier head, 
then un-freezes the encoder for the last 2 layers and fine-tunes the full model,
the trained model optimized for deployment is saved to outputs/distilbert.pt.
additionally, the full model checkpoint with the best validation F1 is saved to outputs/distilbert_pretrained.pt

Usage:
    python -m src.train_bert
"""

from pathlib import Path
from typing import Any, Optional
import tempfile
import time
import traceback

import pandas as pd
import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from src.dataset import MAX_LEN, OUTPUTS_DIR
from src.model_bert import (
    BERT_DROPOUT,
    DISTILBERT_MODEL_NAME,
    PRETRAINED_DISTILBERT_MODEL_NAME,
    DistilBERTSentiment,
)

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover - optional dependency
    AutoTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


EPOCHS = 5
# Set to 5 for better performance, best f1 observed at 4 epochs in initial tests
# beyond 10 epochs, the model starts to overfit on the small dataset, with val F1 plateauing or declining
LR = 2e-5
HEAD_EPOCHS = 2
HEAD_LR = 5e-4
ENCODER_LR = LR
CLASSIFIER_LR = 5e-5
FINE_TUNE_LAST_N_LAYERS: Optional[int] = None
CLIP = 1.0
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16
DROPOUT = BERT_DROPOUT
MODEL_NAME = DISTILBERT_MODEL_NAME
SEED = 42
FREEZE_ENCODER = True
LOCAL_FILES_ONLY = False

CHECKPOINT_PATH = OUTPUTS_DIR / "distilbert_pretrained.pt"
DEPLOY_CHECKPOINT_PATH = OUTPUTS_DIR / "distilbert.pt"
PRETRAINED_MODEL_NAME = PRETRAINED_DISTILBERT_MODEL_NAME
PRETRAINED_LOCAL_FILES_ONLY = LOCAL_FILES_ONLY
PRETRAINED_CHECKPOINT_PATH = CHECKPOINT_PATH


def resolve_device(device: Optional[torch.device] = None) -> torch.device:
    """Resolve the best available device for training or evaluation."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(
    model_name: str = MODEL_NAME,
    local_files_only: bool = LOCAL_FILES_ONLY,
    **legacy_kwargs,
):
    """Load the Hugging Face tokenizer used by DistilBERT."""
    del legacy_kwargs
    if AutoTokenizer is None:
        raise ImportError(
            "transformers is required for DistilBERT tokenization."
        ) from _TRANSFORMERS_IMPORT_ERROR
    return AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )


def load_pretrained_tokenizer(
    model_name: str = PRETRAINED_MODEL_NAME,
    local_files_only: bool = PRETRAINED_LOCAL_FILES_ONLY,
):
    """Backward-compatible alias for the Hugging Face tokenizer loader."""
    return load_tokenizer(model_name=model_name, local_files_only=local_files_only)


class BertReviewDataset(Dataset):
    """Dataset containing tokenized reviews and binary labels."""

    def __init__(self, encodings: dict[str, torch.Tensor], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def encode_texts(
    texts: list[str],
    tokenizer: Any,
    max_len: int = MAX_LEN,
) -> dict[str, torch.Tensor]:
    """Tokenize and pad review texts for DistilBERT."""
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )


def make_bert_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: Any,
    batch_size: int = BATCH_SIZE,
    max_len: int = MAX_LEN,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap train / val / test DataFrames in DistilBERT-ready DataLoaders."""

    def _make(df: pd.DataFrame, shuffle: bool) -> DataLoader:
        encodings = encode_texts(df["text"].tolist(), tokenizer, max_len=max_len)
        generator = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(
            BertReviewDataset(encodings, df["label"].tolist()),
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    train_loader = _make(train_df, shuffle=True)
    val_loader = _make(val_df, shuffle=False)
    test_loader = _make(test_df, shuffle=False)

    print(
        f"make_bert_dataloaders: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)} "
        f"(batch_size={batch_size}, max_len={max_len})",
        flush=True,
    )
    return train_loader, val_loader, test_loader


def make_bert_test_loader(
    test_df: pd.DataFrame,
    tokenizer: Any,
    batch_size: int = BATCH_SIZE,
    max_len: int = MAX_LEN,
    seed: int = SEED,
) -> DataLoader:
    """Create the DistilBERT test DataLoader with one tokenization pass."""
    del seed
    encodings = encode_texts(test_df["text"].tolist(), tokenizer, max_len=max_len)
    return DataLoader(
        BertReviewDataset(encodings, test_df["label"].tolist()),
        batch_size=batch_size,
        shuffle=False,
    )


def train_one_epoch_bert(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    clip: float = CLIP,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run one training epoch for a DistilBERT classifier."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].float().to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item()

    return {"loss": total_loss / len(loader)}


def evaluate_epoch_bert(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Evaluate a DistilBERT classifier on a dataloader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].float().to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) >= 0.5).long().cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch["labels"].long().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {
        "loss": total_loss / len(loader),
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
    }


def _trainable_encoder_layer_indexes(model: DistilBERTSentiment) -> list[int]:
    """Return DistilBERT transformer layer indexes with trainable parameters."""
    return [
        idx
        for idx, layer in enumerate(model.encoder.transformer.layer)
        if any(param.requires_grad for param in layer.parameters())
    ]


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


def _save_checkpoint(
    *,
    checkpoint_path: Path,
    model: DistilBERTSentiment,
    tokenizer: Any,
    model_config: dict[str, Any],
    tokenizer_name: str,
    history: list[dict[str, Any]],
    best_val_f1: float,
    best_epoch: int,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    state_dict = model.state_dict()
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("encoder.")
    }

    trainable_encoder_layer_indexes = _trainable_encoder_layer_indexes(model)
    encoder_is_frozen = not any(
        param.requires_grad for param in model.encoder.parameters()
    )
    all_encoder_trainable = all(
        param.requires_grad for param in model.encoder.parameters()
    )
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
        "model_config": model_config,
        "tokenizer_name": tokenizer_name,
        "tokenizer_files": _serialize_tokenizer(tokenizer),
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "history": history,
        "model_state": fp16_state_dict,
        "weights_format": "torch_state_dict",
        "weights_dtype": "float16",
        "save_strategy": save_strategy,
        "trainable_encoder_layers": trainable_encoder_layer_indexes,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, checkpoint_path)


def _trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def _split_encoder_head_parameters(
    model: DistilBERTSentiment,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    encoder_param_ids = {id(param) for param in model.encoder.parameters()}
    encoder_params = [
        param for param in model.encoder.parameters() if param.requires_grad
    ]
    head_params = [
        param
        for param in model.parameters()
        if id(param) not in encoder_param_ids and param.requires_grad
    ]
    return encoder_params, head_params


def _make_finetune_optimizer(
    model: DistilBERTSentiment,
    *,
    encoder_lr: float,
    classifier_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    encoder_params, head_params = _split_encoder_head_parameters(model)
    param_groups = []
    if encoder_params:
        param_groups.append({
            "params": encoder_params,
            "lr": encoder_lr,
            "name": "encoder",
        })
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": classifier_lr,
            "name": "classifier",
        })
    if not param_groups:
        raise RuntimeError("No trainable parameters found for DistilBERT fine-tuning.")
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def _optimizer_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        str(group.get("name", f"group_{idx}")): float(group["lr"])
        for idx, group in enumerate(optimizer.param_groups)
    }


def train_bert(
    train_df,
    val_df,
    epochs: int = EPOCHS,
    lr: float = LR,
    head_epochs: int = HEAD_EPOCHS,
    head_lr: float = HEAD_LR,
    encoder_lr: Optional[float] = None,
    classifier_lr: float = CLASSIFIER_LR,
    clip: float = CLIP,
    weight_decay: float = WEIGHT_DECAY,
    batch_size: int = BATCH_SIZE,
    max_len: int = MAX_LEN,
    model_name: str = MODEL_NAME,
    dropout: float = DROPOUT,
    freeze_encoder: bool = FREEZE_ENCODER,
    fine_tune_last_n_layers: Optional[int] = FINE_TUNE_LAST_N_LAYERS,
    local_files_only: bool = LOCAL_FILES_ONLY,
    checkpoint_path: Optional[Path] = None,
    seed: int = SEED,
    device: Optional[torch.device] = None,
    **legacy_kwargs,
) -> dict:
    """Train Hugging Face DistilBERT and save the best checkpoint by val F1.

    By default, stage 1 freezes the DistilBERT encoder and trains only the
    classifier head. Stage 2 unfreezes the encoder and fine-tunes the full model
    with separate encoder/head learning rates. Set ``freeze_encoder=False`` to
    skip the frozen-head warmup.
    """
    del legacy_kwargs
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if head_epochs < 0:
        raise ValueError("head_epochs must be non-negative")
    if fine_tune_last_n_layers is not None and fine_tune_last_n_layers < 1:
        raise ValueError("fine_tune_last_n_layers must be None or at least 1")

    encoder_lr = lr if encoder_lr is None else encoder_lr

    device = resolve_device(device)
    checkpoint_path = Path(checkpoint_path or CHECKPOINT_PATH)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = load_tokenizer(model_name=model_name, local_files_only=local_files_only)
    train_loader, val_loader, _ = make_bert_dataloaders(
        train_df,
        val_df,
        val_df,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        seed=seed,
    )

    model = DistilBERTSentiment(
        model_name=model_name,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
        local_files_only=local_files_only,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    best_epoch = 0
    history = []
    total_count = sum(param.numel() for param in model.parameters())
    planned_head_epochs = min(head_epochs, epochs) if freeze_encoder else 0
    planned_finetune_epochs = epochs - planned_head_epochs
    print(
        f"Training Hugging Face DistilBERT on {device} | epochs={epochs} | "
        f"head_epochs={planned_head_epochs} | fine_tune_epochs={planned_finetune_epochs} | "
        f"batch_size={batch_size}",
        flush=True,
    )
    print("-" * 60, flush=True)

    def _run_stage(
        *,
        stage_name: str,
        stage_epochs: int,
        optimizer: torch.optim.Optimizer,
        start_epoch: int,
    ) -> int:
        nonlocal best_val_f1, best_epoch

        if stage_epochs <= 0:
            return start_epoch

        trainable_count = sum(
            param.numel() for param in model.parameters() if param.requires_grad
        )
        print(
            f"{stage_name}: trainable_params={trainable_count:,}/{total_count:,} | "
            f"lrs={_optimizer_lrs(optimizer)}",
            flush=True,
        )

        current_epoch = start_epoch
        for _ in range(stage_epochs):
            current_epoch += 1
            train_metrics = train_one_epoch_bert(
                model,
                train_loader,
                optimizer,
                criterion,
                clip,
                device,
            )
            val_metrics = evaluate_epoch_bert(model, val_loader, criterion, device)

            history.append({
                "epoch": current_epoch,
                "stage": stage_name,
                "train_loss": train_metrics["loss"],
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "lrs": _optimizer_lrs(optimizer),
            })

            print(
                f"Epoch {current_epoch:>2}/{epochs} [{stage_name}] | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f}",
                flush=True,
            )

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_epoch = current_epoch
                _save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    tokenizer=tokenizer,
                    model_config={
                        "model_type": "pretrained",
                        "model_name": model_name,
                        "dropout": dropout,
                        "freeze_encoder": not any(
                            param.requires_grad for param in model.encoder.parameters()
                        ),
                        "local_files_only": local_files_only,
                        "max_len": max_len,
                        "batch_size": batch_size,
                        "num_labels": 1,
                        "head_epochs": planned_head_epochs,
                        "fine_tune_epochs": planned_finetune_epochs,
                        "fine_tune_last_n_layers": fine_tune_last_n_layers,
                        "head_lr": head_lr,
                        "encoder_lr": encoder_lr,
                        "classifier_lr": classifier_lr,
                    },
                    tokenizer_name=model_name,
                    history=history,
                    best_val_f1=best_val_f1,
                    best_epoch=best_epoch,
                )
                print(f"  checkpoint saved (val_f1={best_val_f1:.4f})", flush=True)

        return current_epoch

    current_epoch = 0
    if planned_head_epochs > 0:
        head_params = _trainable_parameters(model)
        if not head_params:
            raise RuntimeError("No trainable classifier parameters found for head training.")
        head_optimizer = torch.optim.AdamW(
            head_params,
            lr=head_lr,
            weight_decay=weight_decay,
        )
        head_optimizer.param_groups[0]["name"] = "classifier"
        current_epoch = _run_stage(
            stage_name="head",
            stage_epochs=planned_head_epochs,
            optimizer=head_optimizer,
            start_epoch=current_epoch,
        )

    if planned_finetune_epochs > 0:
        if fine_tune_last_n_layers is None:
            model.unfreeze_distilbert_encoder()
            fine_tune_stage_name = "fine_tune"
        else:
            trainable_layers = model.unfreeze_last_encoder_layers(
                fine_tune_last_n_layers
            )
            fine_tune_stage_name = f"fine_tune_last_{len(trainable_layers)}"
        finetune_optimizer = _make_finetune_optimizer(
            model,
            encoder_lr=encoder_lr,
            classifier_lr=classifier_lr,
            weight_decay=weight_decay,
        )
        current_epoch = _run_stage(
            stage_name=fine_tune_stage_name,
            stage_epochs=planned_finetune_epochs,
            optimizer=finetune_optimizer,
            start_epoch=current_epoch,
        )

    print("-" * 60, flush=True)
    print(f"Best val F1: {best_val_f1:.4f} at epoch {best_epoch}", flush=True)
    return {"best_val_f1": best_val_f1, "best_epoch": best_epoch, "history": history}


def load_pretrained_bert_bundle(
    checkpoint_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
):
    """Load a saved Hugging Face DistilBERT checkpoint and tokenizer."""
    device = resolve_device(device)
    checkpoint_path = Path(checkpoint_path or PRETRAINED_CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"DistilBERT checkpoint not found at {checkpoint_path}. "
            "Run `train_bert(...)` first."
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    local_files_only = cfg.get("local_files_only", LOCAL_FILES_ONLY)
    model_name = cfg.get("model_name", PRETRAINED_MODEL_NAME)

    model = DistilBERTSentiment(
        model_name=model_name,
        dropout=cfg.get("dropout", DROPOUT),
        freeze_encoder=cfg.get("freeze_encoder", FREEZE_ENCODER),
        local_files_only=local_files_only,
    ).to(device)

    if ckpt.get("weights_format") == "safetensors":
        weights_path = Path(ckpt["weights_path"])
        if not weights_path.is_absolute():
            weights_path = checkpoint_path.parent / weights_path
        state_dict = load_safetensors(weights_path)
        strict = False
        model.load_state_dict(state_dict, strict=strict)
    else:
        strict = ckpt.get("weights_format") != "torch_state_dict"
        model.load_state_dict(ckpt["model_state"], strict=strict)
    model.eval()

    tokenizer = _load_tokenizer_from_checkpoint(
        ckpt,
        model_name=model_name,
        local_files_only=local_files_only,
    )
    return model, tokenizer, ckpt, device


if __name__ == "__main__":
    start_time = time.perf_counter()

    from src.parser import load_all_domains
    from src.preprocess import preprocess

    raw = load_all_domains()
    train_df, val_df, _ = preprocess(raw)
    load_time = time.perf_counter()

    print(f"Data loaded and preprocessed in {load_time - start_time:.2f} seconds", flush=True)
    try:
        train_bert(train_df, val_df, head_epochs=10,head_lr=1e-4, epochs=12)
        #Looks like magic numbers were used here, but they were actually the result of careful experimentation
    except (ImportError, RuntimeError) as exc:
        print(f"Skipping DistilBERT training due to recoverable failure: {exc}")
        traceback.print_exc()

    end_time = time.perf_counter()
    seconds = end_time - load_time
    minutes, remaining_seconds = divmod(seconds, 60)
    print(f"Training time: {int(minutes)} minutes and {remaining_seconds:.2f} seconds", flush=True)

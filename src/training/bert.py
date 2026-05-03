"""Train Hugging Face DistilBERT for Amazon review sentiment analysis.

By default, training freezes the DistilBERT encoder and trains only the
classifier head, then un-freezes the last 2 encoder layers for partial
fine-tuning. The trained model is saved to outputs/distilbert.pt.

Usage:
    python -m src.training.bert

Module layout
─────────────
  src/tokenization/bert.py — device resolution, tokenizer, BertReviewDataset,
                             DataLoader factories
  src/checkpoint_bert.py  — checkpoint serialization and bundle loading
  src/training/bert.py    — training loop, stage orchestration, CLI
                            (this file; re-exports from the above two)
"""

import logging
import traceback
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.config import OUTPUTS_DIR, PRED_THRESHOLD
from src.tokenization.bert import (     # noqa: F401 - re-exported for callers
    BATCH_SIZE,
    LOCAL_FILES_ONLY,
    MODEL_NAME,
    SEED,
    BertReviewDataset,
    encode_texts,
    load_tokenizer,
    make_bert_dataloaders,
    make_bert_test_loader,
    resolve_device,
)
from src.checkpoint_bert import (       # noqa: F401 — re-exported for callers
    DEPLOY_CHECKPOINT_PATH,
    PRETRAINED_MODEL_NAME,
    _load_tokenizer_from_checkpoint,
    _save_checkpoint,
    _serialize_tokenizer,
    _trainable_encoder_layer_indexes,
    load_pretrained_bert_bundle,
)
from src.models.bert import (
    BERT_DROPOUT,
    DISTILBERT_MODEL_NAME,
    PRETRAINED_DISTILBERT_MODEL_NAME,
    DistilBERTSentiment,
)
from src.tokenization.sequence import MAX_LEN

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

EPOCHS               = 5
LR                   = 2e-5
HEAD_EPOCHS          = 2
HEAD_LR              = 5e-4
CLASSIFIER_LR        = 5e-5
FINE_TUNE_LAST_N_LAYERS = 2
CLIP                 = 1.0
WEIGHT_DECAY         = 0.01
DROPOUT              = BERT_DROPOUT
FREEZE_ENCODER       = True


# ---------------------------------------------------------------------------
# Per-epoch training helpers
# ---------------------------------------------------------------------------

def train_one_epoch_bert(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    clip: float = CLIP,
    device: Optional[torch.device] = None,
) -> dict:
    """Run one training epoch for a DistilBERT classifier."""
    device = resolve_device(device)
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].float().to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item()

    return {"loss": total_loss / len(loader)}


def evaluate_epoch_bert(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: Optional[torch.device] = None,
) -> dict:
    """Evaluate a DistilBERT classifier on a dataloader."""
    device = resolve_device(device)
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].float().to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss   = criterion(logits, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) >= PRED_THRESHOLD).long().cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch["labels"].long().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)

    return {
        "loss":     total_loss / len(loader),
        "accuracy": round(acc, 4),
        "f1":       round(f1, 4),
    }


# ---------------------------------------------------------------------------
# Optimizer helpers
# ---------------------------------------------------------------------------

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
        param_groups.append({"params": encoder_params, "lr": encoder_lr,    "name": "encoder"})
    if head_params:
        param_groups.append({"params": head_params,    "lr": classifier_lr, "name": "classifier"})
    if not param_groups:
        raise RuntimeError("No trainable parameters found for DistilBERT fine-tuning.")
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def _optimizer_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        str(group.get("name", f"group_{idx}")): float(group["lr"])
        for idx, group in enumerate(optimizer.param_groups)
    }


# ---------------------------------------------------------------------------
# Main training orchestration
# ---------------------------------------------------------------------------

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
) -> dict:
    """Train Hugging Face DistilBERT and save the best checkpoint by val F1.

    Stage 1 freezes the DistilBERT encoder and trains only the classifier
    head. Stage 2 fine-tunes the final ``fine_tune_last_n_layers`` encoder
    layers with separate encoder/head learning rates. Pass
    ``fine_tune_last_n_layers=None`` to unfreeze the full encoder.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if head_epochs < 0:
        raise ValueError("head_epochs must be non-negative")
    if fine_tune_last_n_layers is not None and fine_tune_last_n_layers < 1:
        raise ValueError("fine_tune_last_n_layers must be None or at least 1")

    encoder_lr     = lr if encoder_lr is None else encoder_lr
    device         = resolve_device(device)
    checkpoint_path = Path(checkpoint_path or DEPLOY_CHECKPOINT_PATH)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer      = load_tokenizer(model_name=model_name, local_files_only=local_files_only)
    tokenizer_files = _serialize_tokenizer(tokenizer)
    train_loader, val_loader, _ = make_bert_dataloaders(
        train_df, val_df, val_df,
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

    best_val_f1  = -1.0
    best_epoch   = 0
    history      = []
    total_count  = sum(param.numel() for param in model.parameters())
    planned_head_epochs     = min(head_epochs, epochs) if freeze_encoder else 0
    planned_finetune_epochs = epochs - planned_head_epochs

    print(
        f"Training Hugging Face DistilBERT on {device} | epochs={epochs} | "
        f"head_epochs={planned_head_epochs} | fine_tune_epochs={planned_finetune_epochs} | "
        f"batch_size={batch_size}",
        flush=True,
    )
    print("-" * 60, flush=True)

    def _run_stage(*, stage_name, stage_epochs, optimizer, start_epoch):
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
            train_metrics = train_one_epoch_bert(model, train_loader, optimizer, criterion, clip, device)
            val_metrics   = evaluate_epoch_bert(model, val_loader, criterion, device)

            history.append({
                "epoch":      current_epoch,
                "stage":      stage_name,
                "train_loss": train_metrics["loss"],
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "lrs":        _optimizer_lrs(optimizer),
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
                best_val_f1  = val_metrics["f1"]
                best_epoch   = current_epoch
                _save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    tokenizer_files=tokenizer_files,
                    model_config={
                        "model_type":              "pretrained",
                        "model_name":              model_name,
                        "dropout":                 dropout,
                        "freeze_encoder":          not any(
                            param.requires_grad for param in model.encoder.parameters()
                        ),
                        "local_files_only":        local_files_only,
                        "max_len":                 max_len,
                        "batch_size":              batch_size,
                        "num_labels":              1,
                        "head_epochs":             planned_head_epochs,
                        "fine_tune_epochs":        planned_finetune_epochs,
                        "fine_tune_last_n_layers": fine_tune_last_n_layers,
                        "head_lr":                 head_lr,
                        "encoder_lr":              encoder_lr,
                        "classifier_lr":           classifier_lr,
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
        head_optimizer = torch.optim.AdamW(head_params, lr=head_lr, weight_decay=weight_decay)
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
            trainable_layers = model.unfreeze_last_encoder_layers(fine_tune_last_n_layers)
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


def main() -> None:
    """Train DistilBERT from the CLI."""
    start_time = time.perf_counter()

    from src.data.parser import load_all_domains
    from src.data.preprocess import preprocess

    raw = load_all_domains()
    train_df, val_df, _ = preprocess(raw)
    load_time = time.perf_counter()

    print(f"Data loaded and preprocessed in {load_time - start_time:.2f} seconds", flush=True)
    try:
        train_bert(
            train_df,
            val_df,
            epochs=12,
            head_epochs=10,
            fine_tune_last_n_layers=2,
            checkpoint_path=DEPLOY_CHECKPOINT_PATH,
            head_lr=1e-4,
        )
    except ImportError as exc:
        print(f"Skipping DistilBERT training: missing dependency — {exc}")
        traceback.print_exc()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(f"Skipping DistilBERT training: GPU OOM — {exc}")
            traceback.print_exc()
        else:
            raise

    end_time = time.perf_counter()
    seconds = end_time - load_time
    minutes, remaining_seconds = divmod(seconds, 60)
    print(f"Training time: {int(minutes)} minutes and {remaining_seconds:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()

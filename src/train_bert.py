"""Training loops for local and pretrained transformer sentiment models.

This module exposes two DistilBERT-style tracks:

1. ``train_bert`` trains the local non-Hugging-Face transformer from scratch,
   optionally initialised from local GloVe vectors.
2. ``train_pretrained_bert`` fine-tunes a cached pretrained DistilBERT model
   when ``transformers`` and local weights are available.
"""

from pathlib import Path
from typing import Any, Optional
import time

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from src.dataset import (
    EMBEDDINGS_DIR,
    PAD_TOKEN,
    UNK_TOKEN,
    MAX_LEN,
    OUTPUTS_DIR,
    build_vocab,
    load_glove,
    load_vocab,
    save_vocab,
    tokenize_and_pad,
)
from src.model_bert import (
    BERT_DROPOUT,
    BERT_EMBEDDING_DIM,
    BERT_FF_DIM,
    BERT_HEADS,
    BERT_LAYERS,
    DISTILBERT_MODEL_NAME,
    PRETRAINED_DISTILBERT_MODEL_NAME,
    DistilBERTSentiment,
    PretrainedDistilBERTSentiment,
)
from src.parser import load_unlabeled_domains
from src.preprocess import clean_text

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover - optional dependency
    AutoTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Local-model hyperparameters
# ---------------------------------------------------------------------------

EPOCHS = 10
LR = 2e-4
CLIP = 1.0
WEIGHT_DECAY = 0.01
BATCH_SIZE = 64
DROPOUT = BERT_DROPOUT
MODEL_NAME = DISTILBERT_MODEL_NAME
SEED = 42
MAX_VOCAB = 30_000
MIN_FREQ = 2
USE_GLOVE = True
PRETRAIN_EPOCHS = 4
PRETRAIN_LR = 5e-4
PRETRAIN_BATCH_SIZE = 16
PRETRAIN_MASK_PROB = 0.15
USE_UNLABELED_PRETRAINING = True
PRETRAIN_MAX_LEN = 128
PRETRAIN_MAX_TEXTS = 20_000

CHECKPOINT_PATH = OUTPUTS_DIR / "distilbert_local.pt"
VOCAB_PATH = OUTPUTS_DIR / "distilbert_vocab.json"
GLOVE_PATH = EMBEDDINGS_DIR / "glove.6B.100d.txt"


# ---------------------------------------------------------------------------
# Pretrained-model hyperparameters
# ---------------------------------------------------------------------------

PRETRAINED_EPOCHS = 4
PRETRAINED_LR = 2e-5
PRETRAINED_WEIGHT_DECAY = 0.01
PRETRAINED_BATCH_SIZE = 16
PRETRAINED_MODEL_NAME = PRETRAINED_DISTILBERT_MODEL_NAME
PRETRAINED_LOCAL_FILES_ONLY = True
PRETRAINED_CHECKPOINT_PATH = OUTPUTS_DIR / "distilbert_pretrained.pt"


def resolve_device(device: Optional[torch.device] = None) -> torch.device:
    """Resolve the best available device for training or evaluation."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LocalReviewTokenizer:
    """Minimal tokenizer wrapper matching the subset of the HF API we use."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab
        self.pad_token_id = vocab["<pad>"]

    def __call__(
        self,
        texts: list[str],
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int = MAX_LEN,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        if padding != "max_length":
            raise ValueError("LocalReviewTokenizer only supports padding='max_length'")
        if not truncation:
            raise ValueError("LocalReviewTokenizer requires truncation=True")
        if return_tensors != "pt":
            raise ValueError("LocalReviewTokenizer only supports return_tensors='pt'")

        input_ids = tokenize_and_pad(texts, self.vocab, max_len=max_length)
        attention_mask = (input_ids != self.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def load_tokenizer(
    model_name: str = MODEL_NAME,
    local_files_only: bool = False,
    vocab: Optional[dict[str, int]] = None,
    vocab_path: Optional[Path] = None,
) -> LocalReviewTokenizer:
    """Load a local tokenizer backed by a saved project vocabulary."""
    del local_files_only

    if vocab is not None:
        return LocalReviewTokenizer(vocab)

    if vocab_path is not None:
        path = Path(vocab_path)
    else:
        candidate = Path(model_name)
        path = candidate if candidate.exists() else VOCAB_PATH

    return LocalReviewTokenizer(load_vocab(path))


def load_pretrained_tokenizer(
    model_name: str = PRETRAINED_MODEL_NAME,
    local_files_only: bool = PRETRAINED_LOCAL_FILES_ONLY,
):
    """Load the pretrained DistilBERT tokenizer from local cache."""
    if AutoTokenizer is None:
        raise ImportError(
            "transformers is required for pretrained DistilBERT support."
        ) from _TRANSFORMERS_IMPORT_ERROR
    return AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )


class BertReviewDataset(Dataset):
    """Dataset containing tokenised reviews and binary labels."""

    def __init__(self, encodings: dict[str, torch.Tensor], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class MaskedLanguageModelingDataset(Dataset):
    """Dataset of token sequences for local masked-language-model pretraining."""

    def __init__(self, tokens: torch.Tensor) -> None:
        self.tokens = tokens

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tokens[idx]


def build_local_pretraining_corpus(
    train_df: pd.DataFrame,
    include_unlabeled: bool = USE_UNLABELED_PRETRAINING,
) -> list[str]:
    """Build a deduplicated text corpus for local unsupervised pretraining.

    Uses the cleaned training split plus any ``unlabeled.review`` files found in
    the repo. The held-out validation and test texts are intentionally excluded.
    """
    texts = [text for text in train_df["text"].tolist() if text.strip()]

    if include_unlabeled:
        unlabeled_df = load_unlabeled_domains()
        if not unlabeled_df.empty:
            unlabeled_texts = [
                clean_text(text)
                for text in unlabeled_df["text"].tolist()
            ]
            texts.extend(text for text in unlabeled_texts if text.strip())

    deduped = list(dict.fromkeys(texts))
    return [text for text in deduped if len(text.split()) >= 3]


def build_local_vocab_texts(
    train_df: pd.DataFrame,
    include_unlabeled: bool = USE_UNLABELED_PRETRAINING,
) -> list[str]:
    """Return the text corpus used to build the local-transformer vocabulary."""
    return build_local_pretraining_corpus(
        train_df,
        include_unlabeled=include_unlabeled,
    )


def make_mlm_dataloader(
    texts: list[str],
    vocab: dict[str, int],
    batch_size: int = PRETRAIN_BATCH_SIZE,
    max_len: int = MAX_LEN,
    seed: int = SEED,
) -> DataLoader:
    """Create a dataloader for masked-language-model pretraining."""
    tokens = tokenize_and_pad(texts, vocab, max_len=max_len)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        MaskedLanguageModelingDataset(tokens),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )


def mask_tokens_for_mlm(
    input_ids: torch.Tensor,
    *,
    pad_token_id: int,
    mask_token_id: int,
    vocab_size: int,
    mask_prob: float = PRETRAIN_MASK_PROB,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply dynamic MLM masking to a batch of token ids."""
    labels = input_ids.clone()
    masked_inputs = input_ids.clone()

    probability_matrix = torch.full(
        labels.shape,
        mask_prob,
        device=input_ids.device,
    )
    probability_matrix.masked_fill_(input_ids == pad_token_id, 0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    replacement_probs = torch.rand(input_ids.shape, device=input_ids.device)

    indices_replaced = masked_indices & (replacement_probs < 0.8)
    masked_inputs[indices_replaced] = mask_token_id

    indices_random = masked_indices & (replacement_probs >= 0.8) & (replacement_probs < 0.9)
    if indices_random.any() and vocab_size > 2:
        random_words = torch.randint(
            low=2,
            high=vocab_size,
            size=input_ids.shape,
            device=input_ids.device,
        )
        masked_inputs[indices_random] = random_words[indices_random]

    return masked_inputs, labels


def encode_texts(
    texts: list[str],
    tokenizer: Any,
    max_len: int = MAX_LEN,
) -> dict[str, torch.Tensor]:
    """Tokenise and pad review texts for either local or pretrained models."""
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
    """Wrap train / val / test DataFrames in transformer-ready DataLoaders."""

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
        f"make_bert_dataloaders: "
        f"train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, "
        f"test={len(test_loader.dataset)}  "
        f"(batch_size={batch_size}, max_len={max_len})"
    )
    return train_loader, val_loader, test_loader


def train_one_epoch_bert(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    clip: float = CLIP,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run one training epoch for a transformer classifier."""
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


def pretrain_one_epoch_local_bert(
    model: DistilBERTSentiment,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    vocab: dict[str, int],
    mask_prob: float = PRETRAIN_MASK_PROB,
    clip: float = CLIP,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run one masked-language-model pretraining epoch for the local transformer."""
    model.train()
    total_loss = 0.0
    total_masked = 0
    total_tokens = 0

    for tokens in loader:
        tokens = tokens.to(device)
        masked_inputs, labels = mask_tokens_for_mlm(
            tokens,
            pad_token_id=vocab[PAD_TOKEN],
            mask_token_id=vocab[UNK_TOKEN],
            vocab_size=len(vocab),
            mask_prob=mask_prob,
        )
        attention_mask = (masked_inputs != vocab[PAD_TOKEN]).long()

        optimizer.zero_grad()
        logits = model.forward_mlm(
            input_ids=masked_inputs,
            attention_mask=attention_mask,
        )
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item()
        total_masked += int((labels != -100).sum().item())
        total_tokens += int((tokens != vocab[PAD_TOKEN]).sum().item())

    return {
        "loss": total_loss / len(loader),
        "masked_fraction": round(total_masked / max(total_tokens, 1), 4),
    }


def pretrain_local_bert(
    model: DistilBERTSentiment,
    train_df: pd.DataFrame,
    vocab: dict[str, int],
    *,
    epochs: int = PRETRAIN_EPOCHS,
    batch_size: int = PRETRAIN_BATCH_SIZE,
    lr: float = PRETRAIN_LR,
    weight_decay: float = WEIGHT_DECAY,
    max_len: int = PRETRAIN_MAX_LEN,
    seed: int = SEED,
    mask_prob: float = PRETRAIN_MASK_PROB,
    include_unlabeled: bool = USE_UNLABELED_PRETRAINING,
    max_texts: Optional[int] = PRETRAIN_MAX_TEXTS,
    clip: float = CLIP,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Pretrain the local transformer on unlabeled text with an MLM objective."""
    if epochs <= 0:
        return {"history": [], "corpus_size": 0}

    texts = build_local_pretraining_corpus(
        train_df,
        include_unlabeled=include_unlabeled,
    )
    if not texts:
        return {"history": [], "corpus_size": 0}

    if max_texts is not None and len(texts) > max_texts:
        sample_indices = torch.randperm(len(texts), generator=torch.Generator().manual_seed(seed))
        texts = [texts[idx] for idx in sample_indices[:max_texts].tolist()]

    current_batch_size = batch_size
    while True:
        loader = make_mlm_dataloader(
            texts,
            vocab=vocab,
            batch_size=current_batch_size,
            max_len=max_len,
            seed=seed,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        history = []

        print(
            f"Local MLM pretraining on {device} | epochs={epochs} | "
            f"corpus={len(texts):,} | batch_size={current_batch_size} | max_len={max_len}",
            flush=True,
        )
        print("-" * 60, flush=True)

        try:
            for epoch in range(1, epochs + 1):
                metrics = pretrain_one_epoch_local_bert(
                    model,
                    loader,
                    optimizer,
                    criterion,
                    vocab=vocab,
                    mask_prob=mask_prob,
                    clip=clip,
                    device=device,
                )
                metrics["epoch"] = epoch
                history.append(metrics)
                print(
                    f"Pretrain {epoch:>2}/{epochs} | "
                    f"mlm_loss={metrics['loss']:.4f} | "
                    f"masked_fraction={metrics['masked_fraction']:.4f}",
                    flush=True,
                )
            print("-" * 60, flush=True)
            return {
                "history": history,
                "corpus_size": len(texts),
                "batch_size": current_batch_size,
                "max_len": max_len,
            }
        except torch.OutOfMemoryError:
            if device.type != "cuda" or current_batch_size <= 4:
                raise
            torch.cuda.empty_cache()
            current_batch_size = max(current_batch_size // 2, 4)
            print(
                f"CUDA OOM during MLM pretraining; retrying with batch_size={current_batch_size}",
                flush=True,
            )


def evaluate_epoch_bert(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Evaluate a transformer classifier on a dataloader."""
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


def _save_checkpoint(
    *,
    checkpoint_path: Path,
    model: nn.Module,
    model_config: dict[str, Any],
    tokenizer_name: str,
    history: list[dict[str, Any]],
    best_val_f1: float,
    best_epoch: int,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "model_config": model_config,
        "tokenizer_name": tokenizer_name,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "history": history,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, checkpoint_path)


def train_bert(
    train_df,
    val_df,
    epochs: int = EPOCHS,
    pretrain_epochs: int = PRETRAIN_EPOCHS,
    lr: float = LR,
    pretrain_lr: float = PRETRAIN_LR,
    clip: float = CLIP,
    weight_decay: float = WEIGHT_DECAY,
    batch_size: int = BATCH_SIZE,
    pretrain_batch_size: int = PRETRAIN_BATCH_SIZE,
    max_len: int = MAX_LEN,
    pretrain_max_len: int = PRETRAIN_MAX_LEN,
    model_name: str = MODEL_NAME,
    dropout: float = DROPOUT,
    freeze_encoder: bool = False,
    local_files_only: bool = False,
    checkpoint_path: Optional[Path] = None,
    seed: int = SEED,
    device: Optional[torch.device] = None,
    embedding_dim: int = BERT_EMBEDDING_DIM,
    n_heads: int = BERT_HEADS,
    n_layers: int = BERT_LAYERS,
    ff_dim: int = BERT_FF_DIM,
    vocab_path: Optional[Path] = None,
    max_vocab: int = MAX_VOCAB,
    min_freq: int = MIN_FREQ,
    use_glove: bool = USE_GLOVE,
    glove_path: Optional[Path] = None,
    pretrain_mask_prob: float = PRETRAIN_MASK_PROB,
    use_unlabeled_pretraining: bool = USE_UNLABELED_PRETRAINING,
    pretrain_max_texts: Optional[int] = PRETRAIN_MAX_TEXTS,
) -> dict:
    """Train the local transformer and save the best checkpoint by val F1."""
    del local_files_only

    device = resolve_device(device)
    checkpoint_path = Path(checkpoint_path or CHECKPOINT_PATH)
    vocab_path = Path(vocab_path or VOCAB_PATH)
    glove_path = Path(glove_path or GLOVE_PATH)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    vocab_texts = build_local_vocab_texts(
        train_df,
        include_unlabeled=use_unlabeled_pretraining,
    )
    vocab = build_vocab(
        vocab_texts,
        max_vocab=max_vocab,
        min_freq=min_freq,
    )
    save_vocab(vocab, vocab_path)
    tokenizer = load_tokenizer(vocab=vocab)

    train_loader, val_loader, _ = make_bert_dataloaders(
        train_df,
        val_df,
        val_df,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        seed=seed,
    )

    pretrained_embeddings = None
    glove_used = False
    token_embedding_dim = embedding_dim
    if use_glove and glove_path.exists():
        pretrained_embeddings = load_glove(vocab, glove_path=glove_path)
        glove_used = True
        token_embedding_dim = int(pretrained_embeddings.shape[1])

    model = DistilBERTSentiment(
        model_name=model_name,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
        local_files_only=False,
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        token_embedding_dim=token_embedding_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_dim=ff_dim,
        max_len=max_len,
        pretrained_embeddings=pretrained_embeddings,
    ).to(device)

    pretraining_result = pretrain_local_bert(
        model,
        train_df,
        vocab=vocab,
        epochs=pretrain_epochs,
        batch_size=pretrain_batch_size,
        lr=pretrain_lr,
        weight_decay=weight_decay,
        max_len=pretrain_max_len,
        seed=seed,
        mask_prob=pretrain_mask_prob,
        include_unlabeled=use_unlabeled_pretraining,
        max_texts=pretrain_max_texts,
        clip=clip,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    best_epoch = 0
    history = []

    print(
        f"Training local transformer on {device} | epochs={epochs} | "
        f"vocab={len(vocab):,} | batch_size={batch_size} | glove={glove_used} | "
        f"pretrain_epochs={pretrain_epochs}",
        flush=True,
    )
    print("-" * 60, flush=True)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch_bert(
            model,
            train_loader,
            optimizer,
            criterion,
            clip,
            device,
        )
        val_metrics = evaluate_epoch_bert(model, val_loader, criterion, device)
        scheduler.step(val_metrics["f1"])

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(
            f"Epoch {epoch:>2}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                model_config={
                    "model_type": "local",
                    "model_name": model_name,
                    "dropout": dropout,
                    "freeze_encoder": freeze_encoder,
                    "local_files_only": False,
                    "max_len": max_len,
                    "batch_size": batch_size,
                    "vocab_size": len(vocab),
                    "embedding_dim": embedding_dim,
                    "token_embedding_dim": token_embedding_dim,
                    "n_heads": n_heads,
                    "n_layers": n_layers,
                    "ff_dim": ff_dim,
                    "max_vocab": max_vocab,
                    "min_freq": min_freq,
                    "use_glove": glove_used,
                    "glove_path": str(glove_path),
                    "pretrain_epochs": pretrain_epochs,
                    "pretrain_lr": pretrain_lr,
                    "pretrain_batch_size": pretrain_batch_size,
                    "pretrain_max_len": pretrain_max_len,
                    "pretrain_mask_prob": pretrain_mask_prob,
                    "pretrain_max_texts": pretrain_max_texts,
                    "use_unlabeled_pretraining": use_unlabeled_pretraining,
                },
                tokenizer_name=str(vocab_path),
                history=history,
                best_val_f1=best_val_f1,
                best_epoch=best_epoch,
                extra={
                    "vocab_path": str(vocab_path),
                    "pretraining_history": pretraining_result["history"],
                    "pretraining_corpus_size": pretraining_result["corpus_size"],
                    "pretraining_batch_size": pretraining_result.get("batch_size"),
                    "pretraining_max_len": pretraining_result.get("max_len"),
                },
            )
            print(f"  ✓ checkpoint saved (val_f1={best_val_f1:.4f})", flush=True)

    print("-" * 60, flush=True)
    print(f"Best val F1: {best_val_f1:.4f} at epoch {best_epoch}", flush=True)
    return {
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "history": history,
        "pretraining_history": pretraining_result["history"],
        "pretraining_corpus_size": pretraining_result["corpus_size"],
        "pretraining_batch_size": pretraining_result.get("batch_size"),
        "pretraining_max_len": pretraining_result.get("max_len"),
    }


def train_pretrained_bert(
    train_df,
    val_df,
    epochs: int = PRETRAINED_EPOCHS,
    lr: float = PRETRAINED_LR,
    clip: float = CLIP,
    weight_decay: float = PRETRAINED_WEIGHT_DECAY,
    batch_size: int = PRETRAINED_BATCH_SIZE,
    max_len: int = MAX_LEN,
    model_name: str = PRETRAINED_MODEL_NAME,
    dropout: float = DROPOUT,
    freeze_encoder: bool = False,
    local_files_only: bool = PRETRAINED_LOCAL_FILES_ONLY,
    checkpoint_path: Optional[Path] = None,
    seed: int = SEED,
    device: Optional[torch.device] = None,
) -> dict:
    """Fine-tune a cached pretrained DistilBERT checkpoint."""
    device = resolve_device(device)
    checkpoint_path = Path(checkpoint_path or PRETRAINED_CHECKPOINT_PATH)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = load_pretrained_tokenizer(
        model_name=model_name,
        local_files_only=local_files_only,
    )
    train_loader, val_loader, _ = make_bert_dataloaders(
        train_df,
        val_df,
        val_df,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        seed=seed,
    )

    model = PretrainedDistilBERTSentiment(
        model_name=model_name,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
        local_files_only=local_files_only,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    best_epoch = 0
    history = []

    print(
        f"Training pretrained DistilBERT on {device} | epochs={epochs} | "
        f"batch_size={batch_size} | local_files_only={local_files_only}",
        flush=True,
    )
    print("-" * 60, flush=True)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch_bert(
            model,
            train_loader,
            optimizer,
            criterion,
            clip,
            device,
        )
        val_metrics = evaluate_epoch_bert(model, val_loader, criterion, device)
        scheduler.step(val_metrics["f1"])

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(
            f"Epoch {epoch:>2}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                model_config={
                    "model_type": "pretrained",
                    "model_name": model_name,
                    "dropout": dropout,
                    "freeze_encoder": freeze_encoder,
                    "local_files_only": local_files_only,
                    "max_len": max_len,
                    "batch_size": batch_size,
                },
                tokenizer_name=model_name,
                history=history,
                best_val_f1=best_val_f1,
                best_epoch=best_epoch,
            )
            print(f"  ✓ checkpoint saved (val_f1={best_val_f1:.4f})", flush=True)

    print("-" * 60, flush=True)
    print(f"Best val F1: {best_val_f1:.4f} at epoch {best_epoch}", flush=True)
    return {"best_val_f1": best_val_f1, "best_epoch": best_epoch, "history": history}


def load_local_bert_bundle(
    checkpoint_path: Optional[Path] = None,
    vocab_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
):
    """Load a saved local-transformer checkpoint and its tokenizer."""
    device = resolve_device(device)
    checkpoint_path = Path(checkpoint_path or CHECKPOINT_PATH)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    resolved_vocab_path = Path(vocab_path or ckpt.get("vocab_path") or VOCAB_PATH)

    model = DistilBERTSentiment(
        model_name=cfg.get("model_name", MODEL_NAME),
        dropout=cfg.get("dropout", DROPOUT),
        freeze_encoder=cfg.get("freeze_encoder", False),
        local_files_only=False,
        vocab_size=cfg.get("vocab_size", 30_000),
        embedding_dim=cfg.get("embedding_dim", BERT_EMBEDDING_DIM),
        token_embedding_dim=cfg.get("token_embedding_dim"),
        n_heads=cfg.get("n_heads", BERT_HEADS),
        n_layers=cfg.get("n_layers", BERT_LAYERS),
        ff_dim=cfg.get("ff_dim", BERT_FF_DIM),
        max_len=cfg.get("max_len", MAX_LEN),
    ).to(device)
    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"Local checkpoint at {checkpoint_path} is incompatible with the current "
            "local transformer architecture. Retrain it with `python -m src.train_bert`."
        ) from exc
    model.eval()

    tokenizer = load_tokenizer(vocab_path=resolved_vocab_path)
    return model, tokenizer, ckpt, device


def load_pretrained_bert_bundle(
    checkpoint_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
):
    """Load a saved pretrained DistilBERT checkpoint and tokenizer."""
    device = resolve_device(device)
    checkpoint_path = Path(checkpoint_path or PRETRAINED_CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Pretrained DistilBERT checkpoint not found at {checkpoint_path}. "
            "Run `train_pretrained_bert(...)` first."
        )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]

    model = PretrainedDistilBERTSentiment(
        model_name=cfg.get("model_name", PRETRAINED_MODEL_NAME),
        dropout=cfg.get("dropout", DROPOUT),
        freeze_encoder=cfg.get("freeze_encoder", False),
        local_files_only=cfg.get("local_files_only", PRETRAINED_LOCAL_FILES_ONLY),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tokenizer = load_pretrained_tokenizer(
        model_name=ckpt.get("tokenizer_name", cfg.get("model_name", PRETRAINED_MODEL_NAME)),
        local_files_only=cfg.get("local_files_only", PRETRAINED_LOCAL_FILES_ONLY),
    )
    return model, tokenizer, ckpt, device


if __name__ == "__main__":
    start_time = time.perf_counter()

    from src.parser import load_all_domains
    from src.preprocess import preprocess

    raw = load_all_domains()
    train_df, val_df, _ = preprocess(raw)
    load_time = time.perf_counter()

    print(f"Data loaded and preprocessed in {load_time - start_time:.2f} seconds")
    train_bert(train_df, val_df)

    if AutoTokenizer is not None:
        try:
            train_pretrained_bert(train_df, val_df)
        except Exception as exc:  
            print(f"Skipping pretrained DistilBERT training: {exc}")

    end_time = time.perf_counter()
    print(f"Training time: {end_time - load_time:.2f} seconds")

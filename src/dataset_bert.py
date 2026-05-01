"""DistilBERT tokenizer and dataset helpers for ReviewPulse.

Provides device resolution, tokenizer loading, BertReviewDataset, and
DataLoader factories used by both training (train_bert.py) and evaluation
(evaluate.py).
"""

from typing import Any, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset import MAX_LEN
from src.model_bert import DISTILBERT_MODEL_NAME, PRETRAINED_DISTILBERT_MODEL_NAME

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover - optional dependency
    AutoTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

BATCH_SIZE       = 16
SEED             = 42
LOCAL_FILES_ONLY = False
MODEL_NAME       = DISTILBERT_MODEL_NAME


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def resolve_device(device: Optional[torch.device] = None) -> torch.device:
    """Resolve the best available device for training or evaluation."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(
    model_name: str = MODEL_NAME,
    local_files_only: bool = LOCAL_FILES_ONLY,
):
    """Load the Hugging Face tokenizer used by DistilBERT."""
    if AutoTokenizer is None:
        raise ImportError(
            "transformers is required for DistilBERT tokenization."
        ) from _TRANSFORMERS_IMPORT_ERROR
    return AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )


# ---------------------------------------------------------------------------
# Dataset + DataLoaders
# ---------------------------------------------------------------------------

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
    val_loader   = _make(val_df,   shuffle=False)
    test_loader  = _make(test_df,  shuffle=False)

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
) -> DataLoader:
    """Create the DistilBERT test DataLoader with one tokenization pass."""
    encodings = encode_texts(test_df["text"].tolist(), tokenizer, max_len=max_len)
    return DataLoader(
        BertReviewDataset(encodings, test_df["label"].tolist()),
        batch_size=batch_size,
        shuffle=False,
    )

"""Sequence tokenization and DataLoader helpers for ReviewPulse."""

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.tokenization.vocab import PAD_TOKEN, UNK_TOKEN

MAX_LEN = 256
BATCH_SIZE = 64


def tokenize_and_pad(
    texts: list[str],
    vocab: dict[str, int],
    max_len: int = MAX_LEN,
) -> torch.Tensor:
    """Convert cleaned text strings into a padded integer tensor."""
    unk_idx = vocab[UNK_TOKEN]
    pad_idx = vocab[PAD_TOKEN]
    result = []
    for text in texts:
        ids = [vocab.get(tok, unk_idx) for tok in text.split()][:max_len]
        ids += [pad_idx] * (max_len - len(ids))
        result.append(ids)
    return torch.tensor(result, dtype=torch.long)


class ReviewDataset(Dataset):
    """PyTorch Dataset wrapping tokenised review tensors and binary labels."""

    def __init__(self, tokens: torch.Tensor, labels: list[int]) -> None:
        self.tokens = tokens
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[idx], self.labels[idx]


def make_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    vocab: dict[str, int],
    batch_size: int = BATCH_SIZE,
    max_len: int = MAX_LEN,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap train / val / test DataFrames in PyTorch DataLoaders."""

    def _make(df: pd.DataFrame, shuffle: bool) -> DataLoader:
        tokens = tokenize_and_pad(df["text"].tolist(), vocab, max_len)
        generator = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(
            ReviewDataset(tokens, df["label"].tolist()),
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    train_loader = _make(train_df, shuffle=True)
    val_loader = _make(val_df, shuffle=False)
    test_loader = _make(test_df, shuffle=False)

    print(
        f"make_dataloaders: "
        f"train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, "
        f"test={len(test_loader.dataset)}  "
        f"(batch_size={batch_size}, max_len={max_len})"
    )
    return train_loader, val_loader, test_loader

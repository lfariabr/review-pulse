"""PyTorch input pipeline for ReviewPulse.

Issue #5  — build_vocab, save_vocab, load_vocab
Issue #6  — tokenize_and_pad, ReviewDataset, make_dataloaders
Issue #7  — load_glove (optional)
"""

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
EMBEDDINGS_DIR = Path(__file__).parent.parent / "embeddings"

PAD_TOKEN = "<pad>"   # index 0 — zero vector, masks padding in the LSTM
UNK_TOKEN = "<unk>"  # index 1 — fallback for out-of-vocabulary words

EMBEDDING_DIM = 100   # matches glove.6B.100d
MAX_LEN = 256
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Issue #5 — Vocabulary builder
# ---------------------------------------------------------------------------

def build_vocab(
    texts: list[str],
    max_vocab: int = 30_000,
    min_freq: int = 2,
) -> dict[str, int]:
    """Build a word → index vocabulary from a list of pre-cleaned texts.

    Must be called on **training texts only** to avoid data leakage into
    the validation and test splits.

    Special tokens:
        <pad>  → 0   (padding, zero-initialised embedding)
        <unk>  → 1   (any word not in the vocabulary)

    Args:
        texts:     List of cleaned review strings (from preprocess.clean_text).
        max_vocab: Maximum vocabulary size including special tokens.
        min_freq:  Minimum word frequency required for inclusion.

    Returns:
        dict mapping word → integer index.
    """
    counter: Counter = Counter()
    for text in texts:
        counter.update(text.split())

    vocab: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, freq in counter.most_common():
        if freq < min_freq:
            break
        if len(vocab) >= max_vocab:
            break
        vocab[word] = len(vocab)

    print(
        f"build_vocab: {len(vocab):,} tokens  "
        f"(max_vocab={max_vocab}, min_freq={min_freq})"
    )
    return vocab


def save_vocab(vocab: dict[str, int], path: Optional[Path] = None) -> Path:
    """Persist vocabulary to a JSON file for reuse at inference time."""
    if path is None:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        path = OUTPUTS_DIR / "vocab.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"save_vocab: saved {len(vocab):,} tokens → {path}")
    return path


def load_vocab(path: Optional[Path] = None) -> dict[str, int]:
    """Load a vocabulary previously saved by save_vocab()."""
    if path is None:
        path = OUTPUTS_DIR / "vocab.json"
    with open(path, encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"load_vocab: loaded {len(vocab):,} tokens ← {path}")
    return vocab


# ---------------------------------------------------------------------------
# Issue #7 — Optional GloVe loader
# ---------------------------------------------------------------------------

def load_glove(
    vocab: dict[str, int],
    glove_path: Optional[Path] = None,
) -> np.ndarray:
    """Initialise an embedding matrix from pre-trained GloVe vectors.

    If the GloVe file does not exist the function returns a randomly
    initialised matrix so training can proceed without the download.

    Args:
        vocab:      Word → index mapping from build_vocab().
        glove_path: Path to glove.6B.100d.txt. Defaults to embeddings/.

    Returns:
        Float32 numpy array of shape (vocab_size, EMBEDDING_DIM).
    """
    if glove_path is None:
        glove_path = EMBEDDINGS_DIR / "glove.6B.100d.txt"

    embeddings = np.random.uniform(
        -0.1, 0.1, (len(vocab), EMBEDDING_DIM)
    ).astype(np.float32)
    embeddings[0] = 0.0  # <pad> is always a zero vector

    if not Path(glove_path).exists():
        print(
            f"load_glove: {glove_path} not found — "
            "using random initialisation (train from scratch)"
        )
        return embeddings

    found = 0
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in vocab:
                embeddings[vocab[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1

    print(
        f"load_glove: {found:,} / {len(vocab):,} vocab words "
        f"initialised from GloVe"
    )
    return embeddings


# ---------------------------------------------------------------------------
# Issue #6 — PyTorch Dataset & DataLoader
# ---------------------------------------------------------------------------

def tokenize_and_pad(
    texts: list[str],
    vocab: dict[str, int],
    max_len: int = MAX_LEN,
) -> torch.Tensor:
    """Convert cleaned text strings into a padded integer tensor.

    Args:
        texts:   List of cleaned review strings.
        vocab:   Word → index mapping from build_vocab().
        max_len: Fixed sequence length; longer reviews are truncated,
                 shorter ones are right-padded with the <pad> index (0).

    Returns:
        LongTensor of shape (len(texts), max_len).
    """
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
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap train / val / test DataFrames in PyTorch DataLoaders.

    The vocab is built externally (from training texts only) and passed in
    so that val/test tokenisation uses the same mapping without leakage.

    Args:
        train_df:   Training split from preprocess.split_data().
        val_df:     Validation split.
        test_df:    Test split.
        vocab:      Word → index mapping built from train_df texts only.
        batch_size: Mini-batch size for training and evaluation.
        max_len:    Sequence length passed to tokenize_and_pad().

    Returns:
        (train_loader, val_loader, test_loader)
    """
    def _make(df: pd.DataFrame, shuffle: bool) -> DataLoader:
        tokens = tokenize_and_pad(df["text"].tolist(), vocab, max_len)
        return DataLoader(
            ReviewDataset(tokens, df["label"].tolist()),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    train_loader = _make(train_df, shuffle=True)
    val_loader   = _make(val_df,   shuffle=False)
    test_loader  = _make(test_df,  shuffle=False)

    print(
        f"make_dataloaders: "
        f"train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, "
        f"test={len(test_loader.dataset)}  "
        f"(batch_size={batch_size}, max_len={max_len})"
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from src.parser import load_all_domains
    from src.preprocess import preprocess

    raw = load_all_domains()
    train, val, test = preprocess(raw)

    vocab = build_vocab(train["text"].tolist())
    save_vocab(vocab)

    embeddings = load_glove(vocab)
    print(f"Embedding matrix: {embeddings.shape}")

    train_loader, val_loader, test_loader = make_dataloaders(train, val, test, vocab)
    tokens, labels = next(iter(train_loader))
    print(f"Batch — tokens: {tokens.shape}, labels: {labels.shape}")

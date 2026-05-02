"""Vocabulary and embedding helpers for ReviewPulse."""

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import OUTPUTS_DIR  # noqa: F401 - re-exported for callers

EMBEDDINGS_DIR = Path(__file__).parents[2] / "embeddings"

PAD_TOKEN = "<pad>"   # index 0 - zero vector, masks padding in the LSTM
UNK_TOKEN = "<unk>"  # index 1 - fallback for out-of-vocabulary words

EMBEDDING_DIM = 100   # matches glove.6B.100d


def build_vocab(
    texts: list[str],
    max_vocab: int = 30_000,
    min_freq: int = 2,
) -> dict[str, int]:
    """Build a word -> index vocabulary from a list of pre-cleaned texts."""
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
    print(f"save_vocab: saved {len(vocab):,} tokens -> {path}")
    return path


def load_vocab(path: Optional[Path] = None) -> dict[str, int]:
    """Load a vocabulary previously saved by save_vocab()."""
    if path is None:
        path = OUTPUTS_DIR / "vocab.json"
    with open(path, encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"load_vocab: loaded {len(vocab):,} tokens <- {path}")
    return vocab


def load_glove(
    vocab: dict[str, int],
    glove_path: Optional[Path] = None,
) -> np.ndarray:
    """Initialise an embedding matrix from pre-trained GloVe vectors.

    If the GloVe file does not exist the function returns a randomly
    initialised matrix so training can proceed without the download.
    """
    if glove_path is None:
        glove_path = EMBEDDINGS_DIR / "glove.6B.100d.txt"

    embeddings = np.random.uniform(
        -0.1, 0.1, (len(vocab), EMBEDDING_DIM)
    ).astype(np.float32)
    embeddings[0] = 0.0  # <pad> is always a zero vector

    if not Path(glove_path).exists():
        print(
            f"load_glove: {glove_path} not found - "
            "using random initialisation (train from scratch)"
        )
        return embeddings

    found = 0
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = parts[1:]
            if len(vector) != EMBEDDING_DIM:
                raise ValueError(
                    f"load_glove: expected {EMBEDDING_DIM}-dim vectors, "
                    f"got {len(vector)} in {glove_path}. "
                    f"Make sure you are using glove.6B.100d.txt."
                )
            if word in vocab:
                embeddings[vocab[word]] = np.array(vector, dtype=np.float32)
                found += 1

    n_special = sum(token in vocab for token in (PAD_TOKEN, UNK_TOKEN))
    coverage = found / max(len(vocab) - n_special, 1) * 100
    print(
        f"load_glove: {found:,} / {len(vocab):,} vocab words initialised "
        f"from GloVe  ({coverage:.1f}% coverage)"
    )
    return embeddings

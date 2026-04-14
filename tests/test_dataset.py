import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.dataset import (
    PAD_TOKEN,
    UNK_TOKEN,
    ReviewDataset,
    build_vocab,
    load_vocab,
    make_dataloaders,
    save_vocab,
    tokenize_and_pad,
)

import pandas as pd


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

TRAIN_TEXTS = [
    "the movie was great and i loved it",
    "terrible film would not recommend",
    "absolutely fantastic performance by the cast",
    "boring and slow not worth watching",
    "one of the best films i have seen",
]

TEST_TEXTS = [
    "unseen word zygote should become unk",
]


# ---------------------------------------------------------------------------
# build_vocab
# ---------------------------------------------------------------------------

def test_vocab_contains_pad_and_unk():
    vocab = build_vocab(TRAIN_TEXTS)
    assert PAD_TOKEN in vocab
    assert UNK_TOKEN in vocab


def test_pad_is_index_zero():
    vocab = build_vocab(TRAIN_TEXTS)
    assert vocab[PAD_TOKEN] == 0


def test_unk_is_index_one():
    vocab = build_vocab(TRAIN_TEXTS)
    assert vocab[UNK_TOKEN] == 1


def test_vocab_respects_max_vocab():
    vocab = build_vocab(TRAIN_TEXTS, max_vocab=5)
    assert len(vocab) <= 5


def test_vocab_respects_min_freq():
    # "zygote" appears 0 times in TRAIN_TEXTS, so it must not appear
    vocab = build_vocab(TRAIN_TEXTS, min_freq=2)
    # most words appear only once — only those with freq >= 2 survive
    for word in vocab:
        if word in (PAD_TOKEN, UNK_TOKEN):
            continue
        count = sum(text.split().count(word) for text in TRAIN_TEXTS)
        assert count >= 2


def test_vocab_built_from_train_only():
    # words that appear only in TEST_TEXTS must not be in the vocab
    vocab = build_vocab(TRAIN_TEXTS, min_freq=1)
    assert "zygote" not in vocab


def test_vocab_indices_are_unique():
    vocab = build_vocab(TRAIN_TEXTS, min_freq=1)
    indices = list(vocab.values())
    assert len(indices) == len(set(indices))


# ---------------------------------------------------------------------------
# save_vocab / load_vocab
# ---------------------------------------------------------------------------

def test_save_and_load_vocab_roundtrip(tmp_path: Path):
    vocab = build_vocab(TRAIN_TEXTS, min_freq=1)
    path = tmp_path / "vocab.json"
    save_vocab(vocab, path)
    loaded = load_vocab(path)
    assert vocab == loaded


def test_save_vocab_creates_valid_json(tmp_path: Path):
    vocab = build_vocab(TRAIN_TEXTS, min_freq=1)
    path = tmp_path / "vocab.json"
    save_vocab(vocab, path)
    with open(path) as f:
        data = json.load(f)
    assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# tokenize_and_pad
# ---------------------------------------------------------------------------

def test_tokenize_output_shape():
    vocab = build_vocab(TRAIN_TEXTS, min_freq=1)
    tokens = tokenize_and_pad(TRAIN_TEXTS, vocab, max_len=16)
    assert tokens.shape == (len(TRAIN_TEXTS), 16)


def test_tokenize_padding_is_zero():
    vocab = build_vocab(TRAIN_TEXTS, min_freq=1)
    # use a very long max_len so all sequences are padded
    tokens = tokenize_and_pad(["hi"], vocab, max_len=50)
    # positions beyond the first word should be 0 (PAD index)
    assert (tokens[0, 1:] == 0).all()


def test_tokenize_truncates_long_sequences():
    vocab = build_vocab(TRAIN_TEXTS, min_freq=1)
    tokens = tokenize_and_pad(TRAIN_TEXTS, vocab, max_len=3)
    assert tokens.shape[1] == 3


def test_tokenize_unk_for_oov_words():
    vocab = build_vocab(TRAIN_TEXTS, min_freq=1)
    tokens = tokenize_and_pad(["zygote"], vocab, max_len=5)
    # "zygote" is OOV → mapped to UNK index (1)
    assert tokens[0, 0].item() == vocab[UNK_TOKEN]


def test_tokenize_returns_long_tensor():
    vocab = build_vocab(TRAIN_TEXTS, min_freq=1)
    tokens = tokenize_and_pad(TRAIN_TEXTS, vocab)
    assert tokens.dtype == torch.long


# ---------------------------------------------------------------------------
# ReviewDataset
# ---------------------------------------------------------------------------

def test_dataset_length():
    vocab = build_vocab(TRAIN_TEXTS, min_freq=1)
    tokens = tokenize_and_pad(TRAIN_TEXTS, vocab)
    labels = [1, 0, 1, 0, 1]
    ds = ReviewDataset(tokens, labels)
    assert len(ds) == 5


def test_dataset_item_shapes():
    vocab = build_vocab(TRAIN_TEXTS, min_freq=1)
    tokens = tokenize_and_pad(TRAIN_TEXTS, vocab, max_len=16)
    ds = ReviewDataset(tokens, [1] * len(TRAIN_TEXTS))
    t, l = ds[0]
    assert t.shape == (16,)
    assert l.shape == ()


def test_dataset_label_dtype():
    vocab = build_vocab(TRAIN_TEXTS, min_freq=1)
    tokens = tokenize_and_pad(TRAIN_TEXTS, vocab)
    ds = ReviewDataset(tokens, [1, 0, 1, 0, 1])
    _, label = ds[0]
    assert label.dtype == torch.float32


# ---------------------------------------------------------------------------
# make_dataloaders
# ---------------------------------------------------------------------------

def _make_split_df(texts, labels):
    return pd.DataFrame({"text": texts, "label": labels})


def test_make_dataloaders_returns_three_loaders():
    vocab = build_vocab(TRAIN_TEXTS, min_freq=1)
    train_df = _make_split_df(TRAIN_TEXTS[:3], [1, 0, 1])
    val_df   = _make_split_df(TRAIN_TEXTS[3:4], [0])
    test_df  = _make_split_df(TRAIN_TEXTS[4:], [1])
    loaders = make_dataloaders(train_df, val_df, test_df, vocab, batch_size=2)
    assert len(loaders) == 3


def test_make_dataloaders_batch_shape():
    vocab = build_vocab(TRAIN_TEXTS, min_freq=1)
    train_df = _make_split_df(TRAIN_TEXTS, [1, 0, 1, 0, 1])
    val_df   = _make_split_df(TRAIN_TEXTS[:2], [1, 0])
    test_df  = _make_split_df(TRAIN_TEXTS[:2], [1, 0])
    train_loader, _, _ = make_dataloaders(
        train_df, val_df, test_df, vocab, batch_size=3, max_len=16
    )
    tokens, labels = next(iter(train_loader))
    assert tokens.shape[1] == 16
    assert labels.shape[0] == tokens.shape[0]

"""Compatibility wrapper for src.data.preprocess."""

from src.data.preprocess import (
    MAX_WORDS,
    MIN_WORDS,
    SEED,
    audit_labels,
    clean_text,
    drop_ambiguous,
    preprocess,
    remove_outliers,
    split_data,
)

__all__ = [
    "SEED",
    "MIN_WORDS",
    "MAX_WORDS",
    "audit_labels",
    "drop_ambiguous",
    "clean_text",
    "remove_outliers",
    "split_data",
    "preprocess",
]

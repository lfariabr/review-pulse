"""Compatibility wrapper for src.data.parser."""

from src.data.parser import (
    DATA_DIR,
    DOMAINS,
    LABEL_MAP,
    UNLABELED_FILENAME,
    load_all_domains,
    load_unlabeled_domains,
    parse_review_file,
)

__all__ = [
    "DATA_DIR",
    "DOMAINS",
    "LABEL_MAP",
    "UNLABELED_FILENAME",
    "load_all_domains",
    "load_unlabeled_domains",
    "parse_review_file",
]

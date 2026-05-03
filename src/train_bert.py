"""Compatibility wrapper for DistilBERT training."""

from src.training.bert import *  # noqa: F401,F403
from src.training.bert import main


if __name__ == "__main__":
    main()

"""Compatibility wrapper for BiLSTM training."""

from src.training.bilstm import *  # noqa: F401,F403
from src.training.bilstm import main


if __name__ == "__main__":
    main()

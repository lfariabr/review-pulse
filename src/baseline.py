"""Compatibility wrapper for baseline training."""

from src.training.baseline import *  # noqa: F401,F403
from src.training.baseline import main


if __name__ == "__main__":
    main()

"""Compatibility wrapper for the baseline model pipeline."""

from src.models.baseline import *  # noqa: F401,F403
from src.models.baseline import main


if __name__ == "__main__":
    main()

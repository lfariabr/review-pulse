"""CLI-compatible wrapper for the ``src.evaluation`` package.

Usage:
    python -m src.evaluate
"""

from src.evaluation import (
    CHECKPOINT_PATH,
    CONFUSION_PNG,
    ERROR_CSV,
    _classification_metrics,
    _load_distilbert_test_df,
    _run_distilbert_evaluation,
    check_distilbert_and_evaluate,
    collect_bert_predictions,
    collect_predictions,
    compute_metrics,
    error_analysis,
    main,
    plot_confusion_matrix,
    run_evaluation,
    run_evaluation_distilbert,
    run_evaluation_distilbert_deploy,
)
from src.inference import load_checkpoint

__all__ = [
    "CHECKPOINT_PATH",
    "CONFUSION_PNG",
    "ERROR_CSV",
    "_classification_metrics",
    "_load_distilbert_test_df",
    "_run_distilbert_evaluation",
    "check_distilbert_and_evaluate",
    "collect_bert_predictions",
    "collect_predictions",
    "compute_metrics",
    "error_analysis",
    "load_checkpoint",
    "main",
    "plot_confusion_matrix",
    "run_evaluation",
    "run_evaluation_distilbert",
    "run_evaluation_distilbert_deploy",
]


if __name__ == "__main__":
    main()

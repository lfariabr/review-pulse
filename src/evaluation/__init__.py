"""Evaluation package public API."""

from .bert import (
    _load_distilbert_test_df,
    _run_distilbert_evaluation,
    collect_bert_predictions,
    run_evaluation_distilbert,
    run_evaluation_distilbert_deploy,
)
from .bilstm import CHECKPOINT_PATH, collect_predictions, run_evaluation
from .errors import ERROR_CSV, error_analysis
from .metrics import _classification_metrics, compute_metrics
from .plots import CONFUSION_PNG, plot_confusion_matrix
from .runner import check_distilbert_and_evaluate, main

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
    "main",
    "plot_confusion_matrix",
    "run_evaluation",
    "run_evaluation_distilbert",
    "run_evaluation_distilbert_deploy",
]

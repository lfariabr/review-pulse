"""CLI runner for evaluation workflows."""

from .bert import run_evaluation_distilbert_deploy
from .bilstm import run_evaluation


def check_distilbert_and_evaluate():
    """Run DistilBERT evaluation if the deploy checkpoint exists, otherwise skip."""
    from src.config import DISTILBERT_PATH

    if not DISTILBERT_PATH.exists():
        print(
            "Skipping DistilBERT deployment evaluation: "
            f"checkpoint not found at {DISTILBERT_PATH}"
        )
        return

    try:
        run_evaluation_distilbert_deploy(checkpoint_path=DISTILBERT_PATH)
    except ImportError as exc:
        print(f"Skipping DistilBERT evaluation: missing dependency — {exc}")


def main() -> None:
    """Run the CLI-compatible evaluation flow."""
    run_evaluation()
    check_distilbert_and_evaluate()

"""Error analysis helpers for evaluation artifacts."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR

ERROR_CSV = OUTPUTS_DIR / "error_analysis.csv"


def error_analysis(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    n_examples: int = 50,
) -> pd.DataFrame:
    """Collect misclassified examples and optionally save to CSV."""
    errors = test_df.copy().reset_index(drop=True)
    errors["predicted"] = y_pred
    errors["true"] = y_true
    errors = errors[errors["predicted"] != errors["true"]].copy()

    errors["error_type"] = errors.apply(
        lambda r: "false_positive" if r["predicted"] == 1 else "false_negative",
        axis=1,
    )

    fp = errors[errors["error_type"] == "false_positive"].head(n_examples // 2)
    fn = errors[errors["error_type"] == "false_negative"].head(n_examples // 2)
    sample = pd.concat([fp, fn]).sort_index()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        sample[["text", "true", "predicted", "error_type"]].to_csv(
            save_path, index=False
        )
        print(
            f"error_analysis: {len(errors)} misclassified → {save_path} "
            f"({len(sample)} examples saved)"
        )
    return errors

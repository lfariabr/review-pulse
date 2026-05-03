"""TF-IDF + Logistic Regression baseline for ReviewPulse.

Provides a classical benchmark to compare against the BiLSTM model.
Expected val/test accuracy: ~85–88%.
"""

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.pipeline import Pipeline

from src.config import BASELINE_PATH

MODEL_PATH = BASELINE_PATH


def build_pipeline() -> Pipeline:
    """Return an untrained TF-IDF + LogisticRegression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30_000,
            ngram_range=(1, 2),    # unigrams + bigrams
            sublinear_tf=True,     # log(tf) smoothing — helps with long reviews
            min_df=2,              # ignore terms appearing in fewer than 2 docs
        )),
        ("clf", LogisticRegression(
            max_iter=1_000,
            C=1.0,
            solver="lbfgs",
        )),
    ])


def train_baseline(
    train_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> Pipeline:
    """Train the baseline pipeline on the training split and save it.

    Args:
        train_df:  DataFrame with 'text' and 'label' columns.
        save_path: Where to persist the fitted pipeline. Defaults to
                   outputs/baseline.joblib.

    Returns:
        Fitted sklearn Pipeline.
    """
    pipeline = build_pipeline()
    pipeline.fit(train_df["text"], train_df["label"])

    path = save_path or MODEL_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"train_baseline: model saved → {path}")
    return pipeline


def evaluate_baseline(
    pipeline: Pipeline,
    df: pd.DataFrame,
    split_name: str = "val",
) -> dict:
    """Evaluate a fitted pipeline and print a classification report.

    Args:
        pipeline:   Fitted pipeline from train_baseline().
        df:         DataFrame with 'text' and 'label' columns.
        split_name: Label for the printed report (e.g. 'val', 'test').

    Returns:
        Dict with 'accuracy' and 'f1' keys.
    """
    preds = pipeline.predict(df["text"])
    acc = accuracy_score(df["label"], preds)
    f1 = f1_score(df["label"], preds)

    print(f"\n=== Baseline — {split_name} ===")
    print(classification_report(
        df["label"], preds, target_names=["negative", "positive"]
    ))
    return {"accuracy": round(acc, 4), "f1": round(f1, 4)}


def load_baseline(path: Optional[Path] = None) -> Pipeline:
    """Load a previously saved baseline pipeline."""
    path = path or MODEL_PATH
    pipeline = joblib.load(path)
    print(f"load_baseline: loaded ← {path}")
    return pipeline


def main() -> None:
    """Train and evaluate the baseline model from the CLI."""
    from src.data.parser import load_all_domains
    from src.data.preprocess import preprocess

    raw = load_all_domains()
    train, val, test = preprocess(raw)

    pipeline = train_baseline(train)

    val_metrics = evaluate_baseline(pipeline, val, split_name="val")
    test_metrics = evaluate_baseline(pipeline, test, split_name="test")

    print(f"\nSummary  val → acc={val_metrics['accuracy']}  f1={val_metrics['f1']}")
    print(f"Summary test → acc={test_metrics['accuracy']}  f1={test_metrics['f1']}")


if __name__ == "__main__":
    main()

# .venv/bin/python3 -m src.training.baseline -v

from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.training.baseline import (
    build_pipeline,
    evaluate_baseline,
    load_baseline,
    train_baseline,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

TEXTS_POS = [
    "this product is absolutely fantastic i love it so much",
    "excellent quality exceeded my expectations highly recommended",
    "best purchase i have ever made works perfectly every time",
    "outstanding performance really impressed with this item",
    "great value for the price would definitely buy again",
]

TEXTS_NEG = [
    "terrible product broke after one day complete waste of money",
    "do not buy this absolute garbage worst purchase ever made",
    "very disappointed with the quality not worth the price at all",
    "stopped working after a week would not recommend to anyone",
    "awful experience the product did not work as described",
]

ALL_TEXTS  = TEXTS_POS + TEXTS_NEG
ALL_LABELS = [1] * 5 + [0] * 5


def _small_df(texts=ALL_TEXTS, labels=ALL_LABELS) -> pd.DataFrame:
    return pd.DataFrame({"text": texts, "label": labels})


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------

def test_build_pipeline_returns_pipeline():
    assert isinstance(build_pipeline(), Pipeline)


def test_models_baseline_wrapper_exports_public_api():
    from src.models.baseline import (
        build_pipeline as model_build_pipeline,
        evaluate_baseline as model_evaluate_baseline,
        load_baseline as model_load_baseline,
        train_baseline as model_train_baseline,
    )

    assert model_build_pipeline is build_pipeline
    assert model_evaluate_baseline is evaluate_baseline
    assert model_load_baseline is load_baseline
    assert model_train_baseline is train_baseline


def test_build_pipeline_has_tfidf_and_clf():
    p = build_pipeline()
    step_names = [name for name, _ in p.steps]
    assert "tfidf" in step_names
    assert "clf" in step_names


# ---------------------------------------------------------------------------
# train_baseline
# ---------------------------------------------------------------------------

def test_train_baseline_returns_fitted_pipeline(tmp_path):
    df = _small_df()
    pipeline = train_baseline(df, save_path=tmp_path / "baseline.joblib")
    assert isinstance(pipeline, Pipeline)


def test_train_baseline_saves_file(tmp_path):
    df = _small_df()
    path = tmp_path / "baseline.joblib"
    train_baseline(df, save_path=path)
    assert path.exists()


def test_train_baseline_predictions_are_binary(tmp_path):
    df = _small_df()
    pipeline = train_baseline(df, save_path=tmp_path / "baseline.joblib")
    preds = pipeline.predict(df["text"])
    assert set(preds).issubset({0, 1})


def test_train_baseline_prediction_shape(tmp_path):
    df = _small_df()
    pipeline = train_baseline(df, save_path=tmp_path / "baseline.joblib")
    preds = pipeline.predict(df["text"])
    assert len(preds) == len(df)


# ---------------------------------------------------------------------------
# evaluate_baseline
# ---------------------------------------------------------------------------

def test_evaluate_baseline_returns_accuracy_and_f1(tmp_path):
    df = _small_df()
    pipeline = train_baseline(df, save_path=tmp_path / "baseline.joblib")
    metrics = evaluate_baseline(pipeline, df, split_name="test")
    assert "accuracy" in metrics
    assert "f1" in metrics


def test_evaluate_baseline_accuracy_is_float(tmp_path):
    df = _small_df()
    pipeline = train_baseline(df, save_path=tmp_path / "baseline.joblib")
    metrics = evaluate_baseline(pipeline, df)
    assert isinstance(metrics["accuracy"], float)
    assert 0.0 <= metrics["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# load_baseline
# ---------------------------------------------------------------------------

def test_load_baseline_roundtrip(tmp_path):
    df = _small_df()
    path = tmp_path / "baseline.joblib"
    original = train_baseline(df, save_path=path)
    loaded = load_baseline(path)
    # predictions from original and loaded must be identical
    assert list(original.predict(df["text"])) == list(loaded.predict(df["text"]))


# ---------------------------------------------------------------------------
# integration — real data accuracy floor
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_baseline_real_data_accuracy():
    """Sanity check: baseline must exceed 80% accuracy on the real val split."""
    from src.data.parser import load_all_domains
    from src.data.preprocess import preprocess
    import tempfile

    raw = load_all_domains()
    train, val, _ = preprocess(raw)

    with tempfile.TemporaryDirectory() as tmp:
        pipeline = train_baseline(train, save_path=Path(tmp) / "baseline.joblib")
        metrics = evaluate_baseline(pipeline, val, split_name="val")

    assert metrics["accuracy"] >= 0.80, (
        f"Expected ≥ 80% val accuracy, got {metrics['accuracy']:.2%}"
    )

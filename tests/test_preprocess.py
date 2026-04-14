import pandas as pd
import pytest

from src.preprocess import (
    audit_labels,
    clean_text,
    drop_ambiguous,
    remove_outliers,
    split_data,
)


# --- helpers ---

def _make_df(ratings, labels, texts=None) -> pd.DataFrame:
    n = len(ratings)
    return pd.DataFrame({
        "text": texts if texts else ["good product review here"] * n,
        "rating": ratings,
        "label": labels,
        "domain": ["books"] * n,
        "source_file": ["positive.review" if l == 1 else "negative.review" for l in labels],
    })


# --- audit_labels ---

def test_audit_flags_three_star_as_ambiguous():
    df = _make_df([5.0, 3.0, 1.0], [1, 0, 0])
    result = audit_labels(df)
    assert result["is_ambiguous"].tolist() == [False, True, False]


def test_audit_flags_positive_file_with_low_rating():
    df = _make_df([1.0], [1])  # positive label but 1-star rating
    result = audit_labels(df)
    assert result["rating_conflict"].iloc[0]


def test_audit_flags_negative_file_with_high_rating():
    df = _make_df([5.0], [0])  # negative label but 5-star rating
    result = audit_labels(df)
    assert result["rating_conflict"].iloc[0]


def test_audit_no_conflict_on_clean_rows():
    df = _make_df([5.0, 1.0], [1, 0])
    result = audit_labels(df)
    assert not result["rating_conflict"].any()
    assert not result["is_ambiguous"].any()


def test_audit_adds_two_columns():
    df = _make_df([4.0], [1])
    result = audit_labels(df)
    assert "is_ambiguous" in result.columns
    assert "rating_conflict" in result.columns


# --- drop_ambiguous ---

def test_drop_ambiguous_removes_three_star():
    df = _make_df([5.0, 3.0, 1.0], [1, 1, 0])
    result = drop_ambiguous(df)
    assert 3.0 not in result["rating"].values


def test_drop_ambiguous_removes_conflicts():
    df = _make_df([1.0, 5.0], [1, 0])  # both are conflicts
    result = drop_ambiguous(df)
    assert len(result) == 0


def test_drop_ambiguous_does_not_add_audit_columns():
    df = _make_df([5.0, 1.0], [1, 0])
    result = drop_ambiguous(df)
    assert "is_ambiguous" not in result.columns
    assert "rating_conflict" not in result.columns


# --- clean_text ---

def test_clean_text_lowercases():
    assert clean_text("GREAT PRODUCT") == "great product"


def test_clean_text_strips_html():
    assert clean_text("<b>Good</b>") == "good"


def test_clean_text_expands_negation_dont():
    assert "not" in clean_text("I don't like it")


def test_clean_text_expands_negation_wont():
    assert "will not" in clean_text("I won't buy again") or "not" in clean_text("I won't buy again")


def test_clean_text_removes_punctuation():
    result = clean_text("Wow!!! Amazing... product.")
    assert "!" not in result
    assert "." not in result


def test_clean_text_collapses_whitespace():
    result = clean_text("too   many    spaces")
    assert "  " not in result


def test_clean_text_empty_string():
    assert clean_text("") == ""


# --- remove_outliers ---

def test_remove_outliers_drops_short_reviews():
    df = pd.DataFrame({
        "text": ["hi", "this is a much longer review with plenty of words here"],
        "label": [1, 0],
    })
    result = remove_outliers(df, min_words=5, max_words=500)
    assert len(result) == 1
    assert "longer" in result["text"].iloc[0]


def test_remove_outliers_drops_long_reviews():
    df = pd.DataFrame({
        "text": ["word " * 600, "this short review is well within the limit"],
        "label": [1, 0],
    })
    result = remove_outliers(df, min_words=5, max_words=500)
    assert len(result) == 1
    assert "short" in result["text"].iloc[0]


def test_remove_outliers_keeps_boundary_values():
    df = pd.DataFrame({
        "text": ["word " * 10, "word " * 500],
        "label": [1, 0],
    })
    result = remove_outliers(df, min_words=10, max_words=500)
    assert len(result) == 2


# --- split_data ---

def test_split_data_total_size():
    df = pd.DataFrame({"text": ["x"] * 100, "label": [i % 2 for i in range(100)]})
    train, val, test = split_data(df, seed=42)
    assert len(train) + len(val) + len(test) == 100


def test_split_data_reproducible():
    df = pd.DataFrame({"text": ["x"] * 100, "label": [i % 2 for i in range(100)]})
    train_a, _, _ = split_data(df, seed=42)
    train_b, _, _ = split_data(df, seed=42)
    assert list(train_a.index) == list(train_b.index)


def test_split_data_stratified_label_balance():
    df = pd.DataFrame({"text": ["x"] * 200, "label": [1] * 100 + [0] * 100})
    train, val, test = split_data(df, seed=42)
    for split in [train, val, test]:
        ratio = split["label"].mean()
        assert 0.4 <= ratio <= 0.6

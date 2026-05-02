import re

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
MIN_WORDS = 10
MAX_WORDS = 500

# A "conflict" means the star rating contradicts the filename label.
# e.g., a review in positive.review rated 1 or 2 stars is suspicious.
_CONFLICT_POS_THRESHOLD = 3.0  # positive-file reviews rated below this are flagged
_CONFLICT_NEG_THRESHOLD = 3.0  # negative-file reviews rated above this are flagged


def audit_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add audit columns to flag ambiguous or conflicting label/rating pairs.

    Adds two boolean columns:
        is_ambiguous    — True for 3-star reviews (no clear sentiment signal)
        rating_conflict — True where the filename label contradicts the star rating
    """
    df = df.copy()
    df["is_ambiguous"] = df["rating"] == 3.0
    df["rating_conflict"] = (
        ((df["label"] == 1) & (df["rating"] < _CONFLICT_POS_THRESHOLD)) |
        ((df["label"] == 0) & (df["rating"] > _CONFLICT_NEG_THRESHOLD))
    )
    return df


def drop_ambiguous(df: pd.DataFrame) -> pd.DataFrame:
    """Drop 3-star and rating-conflicting reviews.

    Ethical justification: 3-star reviews carry no clear positive or negative
    signal — forcing them into either class introduces label noise. Rating
    conflicts (e.g. a 1-star review in positive.review) suggest the filename
    label does not reflect the reviewer's actual opinion. Dropping both
    categories improves label quality and is documented as a data decision.
    """
    audited = audit_labels(df)
    n_before = len(audited)
    clean = audited[~audited["is_ambiguous"] & ~audited["rating_conflict"]].copy()
    clean = clean.drop(columns=["is_ambiguous", "rating_conflict"])
    print(f"drop_ambiguous: removed {n_before - len(clean)} rows  ({n_before} → {len(clean)})")
    return clean.reset_index(drop=True)


def clean_text(text: str) -> str:
    """Normalise a review string for model input.

    Steps:
      1. Lowercase
      2. Strip residual HTML tags
      3. Expand negation contractions to preserve sentiment signal
         (e.g. "don't" → "do not", so "not" survives punctuation removal)
      4. Remove non-alphabetic characters
      5. Collapse whitespace
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    # Expand contractions before stripping punctuation
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_outliers(
    df: pd.DataFrame,
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
) -> pd.DataFrame:
    """Drop reviews outside the word-count range.

    Default thresholds are heuristics — validate with EDA (features.py)
    before treating them as ground truth.
    """
    counts = df["text"].str.split().str.len()
    mask = (counts >= min_words) & (counts <= max_words)
    print(f"remove_outliers: removed {(~mask).sum()} rows  (< {min_words} or > {max_words} words)")
    return df[mask].reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70 / 15 / 15 train / validation / test split.

    Stratification preserves the positive/negative ratio in each split.
    The seed is fixed across all runs for reproducibility.
    """
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    adjusted_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=adjusted_val, random_state=seed, stratify=train_val["label"]
    )
    print(f"split_data: train={len(train)}, val={len(val)}, test={len(test)}  (seed={seed})")
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full preprocessing pipeline.

    1. Drop ambiguous / conflicting labels
    2. Clean text
    3. Remove length outliers
    4. Split into train / val / test
    """
    df = drop_ambiguous(df)
    df = df.copy()
    df["text"] = df["text"].map(clean_text)
    df = remove_outliers(df)
    return split_data(df)


if __name__ == "__main__":
    from src.data.parser import load_all_domains

    raw = load_all_domains()
    print(f"Raw: {len(raw)} reviews\n")

    audited = audit_labels(raw)
    print(f"Ambiguous (3-star):  {audited['is_ambiguous'].sum()}")
    print(f"Rating conflicts:    {audited['rating_conflict'].sum()}\n")

    train, val, test = preprocess(raw)
    print(f"\nLabel balance — train:\n{train['label'].value_counts()}")

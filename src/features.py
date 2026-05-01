"""EDA helper functions.

All functions accept the raw DataFrame from load_all_domains() and return
summary DataFrames or print reports. Plot functions optionally save to outputs/.
"""

import matplotlib.pyplot as plt
import pandas as pd

from src.config import OUTPUTS_DIR


# --- summary tables ---

def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of positive / negative reviews."""
    counts = df["label"].value_counts().rename({1: "positive", 0: "negative"})
    pct = (counts / len(df) * 100).round(1)
    return pd.DataFrame({"count": counts, "pct": pct})


def domain_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Review counts per domain, split by label."""
    return (
        df.groupby(["domain", "label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "negative", 1: "positive"})
    )


def rating_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Rating value counts across the full dataset."""
    return df["rating"].value_counts().sort_index().rename("count").to_frame()


def length_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for review word counts."""
    wc = df["text"].str.split().str.len()
    stats = wc.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(1)
    return stats.to_frame(name="word_count")


def label_audit_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Run audit_labels and return a summary of flagged rows."""
    from src.preprocess import audit_labels

    audited = audit_labels(df)
    summary = {
        "total_reviews": len(df),
        "ambiguous_3star": int(audited["is_ambiguous"].sum()),
        "rating_conflicts": int(audited["rating_conflict"].sum()),
        "clean_reviews": int((~audited["is_ambiguous"] & ~audited["rating_conflict"]).sum()),
    }
    return pd.DataFrame.from_dict(summary, orient="index", columns=["count"])


# --- plots ---

def plot_length_distribution(
    df: pd.DataFrame,
    min_words: int = 10,
    max_words: int = 500,
    save: bool = True,
) -> None:
    """Histogram of review word counts with outlier threshold lines."""
    wc = df["text"].str.split().str.len()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(wc, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.axvline(min_words, color="#DD3333", linestyle="--", label=f"min={min_words}")
    ax.axvline(max_words, color="#FF8800", linestyle="--", label=f"max={max_words}")
    ax.set_xlabel("Word count")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Review length distribution")
    ax.legend()
    fig.tight_layout()

    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        path = OUTPUTS_DIR / "length_distribution.png"
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")

    plt.show()


def plot_domain_balance(df: pd.DataFrame, save: bool = True) -> None:
    """Stacked bar chart of positive / negative counts per domain."""
    balance = domain_balance(df)

    fig, ax = plt.subplots(figsize=(8, 4))
    balance.plot(
        kind="bar",
        ax=ax,
        color=["#DD3333", "#4C72B0"],
        edgecolor="white",
        linewidth=0.4,
        width=0.6,
    )
    ax.set_xlabel("Domain")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Review counts per domain")
    ax.set_xticklabels(balance.index, rotation=20, ha="right")
    ax.legend(title="Label")
    fig.tight_layout()

    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        path = OUTPUTS_DIR / "domain_balance.png"
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")

    plt.show()


if __name__ == "__main__":
    from src.parser import load_all_domains

    df = load_all_domains()

    print("=== Class balance ===")
    print(class_balance(df), "\n")

    print("=== Domain balance ===")
    print(domain_balance(df), "\n")

    print("=== Rating distribution ===")
    print(rating_distribution(df), "\n")

    print("=== Length stats ===")
    print(length_stats(df), "\n")

    print("=== Label audit summary ===")
    print(label_audit_summary(df), "\n")

    plot_length_distribution(df)
    plot_domain_balance(df)

from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent.parent / "data"

DOMAINS = {
    "books": DATA_DIR / "books",
    "dvd": DATA_DIR / "dvd",
    "electronics": DATA_DIR / "electronics",
    "kitchen": DATA_DIR / "kitchen_&_housewares",
}

# Primary label comes from the filename, not the star rating.
# The rating is retained for auditing and ethics analysis in preprocess.py.
LABEL_MAP = {
    "positive.review": 1,
    "negative.review": 0,
}
UNLABELED_FILENAME = "unlabeled.review"


def parse_review_file(filepath: Path, label: int) -> list[dict]:
    """Parse a single .review pseudo-XML file into a list of record dicts.

    Skips any <review> block that has no <review_text> tag.
    Missing <rating> is stored as None — handled downstream in audit_labels().
    """
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(text, "html.parser")
    records = []
    for review in soup.find_all("review"):
        text_tag = review.find("review_text")
        if text_tag is None:
            continue
        rating_tag = review.find("rating")
        records.append({
            "text": text_tag.get_text(strip=True),
            "rating": float(rating_tag.get_text(strip=True)) if rating_tag else None,
            "label": label,
            "domain": None,       # filled by load_all_domains()
            "source_file": filepath.name,
        })
    return records


def _resolve_domain_path(data_dir: Path, domain_path: Path) -> Path:
    """Resolve a domain path against a custom data directory when provided."""
    return data_dir / domain_path.name if data_dir != DATA_DIR else domain_path


def load_all_domains(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load positive and negative reviews from all 4 domains.

    Returns a DataFrame with columns: text, rating, label, domain, source_file.
    """
    records = []
    for domain, domain_path in DOMAINS.items():
        resolved = _resolve_domain_path(data_dir, domain_path)
        for filename, label in LABEL_MAP.items():
            filepath = resolved / filename
            if not filepath.exists():
                continue
            domain_records = parse_review_file(filepath, label)
            for r in domain_records:
                r["domain"] = domain
            records.extend(domain_records)

    df = pd.DataFrame(records)
    return df.reset_index(drop=True)


def load_unlabeled_domains(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load unlabeled reviews from all domains where the file exists.

    Returns a DataFrame with the same schema as ``load_all_domains()``, but with
    ``label`` fixed to ``-1`` to indicate that the text should only be used for
    unsupervised workflows such as local pretraining.
    """
    records = []
    for domain, domain_path in DOMAINS.items():
        resolved = _resolve_domain_path(data_dir, domain_path)
        filepath = resolved / UNLABELED_FILENAME
        if not filepath.exists():
            continue
        domain_records = parse_review_file(filepath, label=-1)
        for record in domain_records:
            record["domain"] = domain
        records.extend(domain_records)

    df = pd.DataFrame(records)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_all_domains()
    print(f"Loaded {len(df):,} reviews\n")
    print(df.groupby(["domain", "label"]).size().unstack(fill_value=0).rename(columns={0: "negative", 1: "positive"}))
    print(f"\nNull ratings: {df['rating'].isna().sum()}")
    print(df.dtypes)

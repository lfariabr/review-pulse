# Issue Breakdown — ReviewPulse

## Issue #1 — Project setup & structure ✅

What landed in `lfariabr/review-pulse`:
- `src/__init__.py` — package entry point
- `tests/`, `notebooks/`, `outputs/`, `embeddings/` — directories with `.gitkeep`
- `.gitignore` — excludes model artifacts, GloVe files, pycache, `.env`, `.DS_Store`
- `requirements.txt` — all dependencies pinned
- `README.md` — install, data setup, train, evaluate, app, and test commands

---

## Issue #2 — Implement review parser (`parser.py`) ✅

What landed:
- `src/parser.py` — `parse_review_file()` + `load_all_domains()`, 8,000 reviews loaded cleanly
- `tests/test_parser.py` — 10 tests: unit fixtures + integration against real data, all green
- `conftest.py` — pytest path fix so `src` is importable from tests
- `.venv/` added to `.gitignore`

---

## Issue #3 — Label audit & preprocessing (`preprocess.py`) ✅

> Notable finding: 0 ambiguous/conflicting rows — the dataset owners pre-split by filename with no 3-star reviews included, so `drop_ambiguous` is a no-op on this data. Good to mention in the ethics section: label quality is already high, but we audited it explicitly and documented the decision.

What landed:
- `src/preprocess.py` — `audit_labels`, `drop_ambiguous`, `clean_text` (with negation expansion), `remove_outliers`, `split_data`, `preprocess` pipeline
- `tests/test_preprocess.py` — 21 tests, all green

Final splits on real data: **train=5,404 / val=1,158 / test=1,159**, near-perfect label balance.

---

## Issue #4 — EDA & feature analysis (`features.py` + `EDA.ipynb`) ✅

Key findings from the real data — useful for the presentation and ethics section:

| Finding | Implication |
|---|---|
| Perfectly balanced (50/50) | Accuracy is a reliable metric; no class weighting needed |
| 1,000 reviews per domain per class | No domain dominates training |
| 0 ambiguous or conflicting labels | Label quality is high — good news for the model |
| Median review is 90 words, 95th pct is 417 | `max=500` threshold is safe; only ~3.5% dropped |
| Rating split: 1★(2421), 2★(1579) vs 4★(1121), 5★(2879) | Positive reviews skew toward 5★ — worth mentioning in ethics |

What landed:
- `src/features.py` — `class_balance`, `domain_balance`, `rating_distribution`, `length_stats`, `label_audit_summary`, `plot_length_distribution`, `plot_domain_balance`
- `notebooks/EDA.ipynb` — full walkthrough with conclusions table
- `outputs/length_distribution.png` and `outputs/domain_balance.png` (gitignored)

---

## Issue #5 — Vocabulary builder (`dataset.py`) ✅

What landed:
- `build_vocab()` — 15,924 tokens from training texts only, `min_freq=2` eliminates noise
- `save_vocab()` / `load_vocab()` — JSON round-trip for reuse at inference time

---

## Issue #6 — PyTorch Dataset & DataLoader (`dataset.py`) ✅

What landed (separate commit from #5):
- `tokenize_and_pad()` — fixed-length `(batch, 256)` LongTensor
- `ReviewDataset` — standard `torch.utils.data.Dataset`
- `make_dataloaders()` — seeded `torch.Generator` for reproducible training-set shuffling; val/test loaders never shuffled
- `tests/test_dataset.py` — 21 tests covering vocab, tokenisation, Dataset, DataLoader, and seed reproducibility, all green

Batch shape confirmed on real data: `tokens=(64, 256), labels=(64)`.

---

## Issue #7 — Optional GloVe loader (`dataset.py`) ✅

What landed (separate commit from #5/#6):
- `load_glove()` — maps vocab to 100d GloVe vectors; validates embedding dimension; reports coverage %; gracefully falls back to random init if file not present
- `README.md` — dedicated GloVe section with comparison table and step-by-step setup instructions

---

## Issues #8–21 — Pending

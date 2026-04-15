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

## Issue #8 — TF-IDF + Logistic Regression baseline (`baseline.py`) ✅

Key results on real data, that's the benchmark the BiLSTM needs to beat. 
┌───────┬──────────┬───────┐            
│ Split │ Accuracy │  F1   │            
├───────┼──────────┼───────┤
│ Val   │ 83.5%    │ 83.2% │            
├───────┼──────────┼───────┤          
│ Test  │ 82.7%    │ 81.9% │            
└───────┴──────────┴───────┘          
    
What also landed:
- `src/baseline.py` — build_pipeline, train_baseline, evaluate_baseline, load_baseline                   
- `tests/test_baseline.py` — 9 unit tests + 1 slow integration test (runs full pipeline, marked @pytest.mark.slow)   
- `pytest.ini` — registers the slow marker so -m "not slow" works cleanly  

## Issue #9 — BiLSTM model definition (`model.py`) ✅

What landed:
- `src/model.py` — `BiLSTMSentiment` nn.Module: `Embedding(padding_idx=0) → Dropout → BiLSTM(hidden=256, layers=2, bidirectional) → concat final fwd+bwd hidden → Dropout → Linear(512, 1)`. Raw logit output; optional `pretrained_embeddings` from `load_glove()`.
- `tests/test_model.py` — 21 tests: architecture checks, forward pass shape/dtype, batch size variants, GloVe init, bad shape rejection, determinism in eval mode. All green.

## Issue #10 — Training loop with validation & early stopping (`train.py`) ✅

Real data results (MPS, 10 epochs, GloVe 97.4% coverage):

| Epoch | Train Loss | Val Loss | Val Acc | Val F1 |
|-------|-----------|---------|--------|-------|
| 1 | 0.6570 | 0.5701 | 72.6% | 71.5% |
| 2 | 0.5831 | 0.4932 | 75.3% | 75.0% |
| 3 | 0.5080 | 0.4360 | 80.6% | 79.8% |
| 4 | 0.4756 | 0.4591 | 78.7% | 75.2% |
| 5 | 0.4375 | 0.4500 | 78.8% | 80.8% |
| 6 | 0.3784 | 0.4208 | 80.4% | 82.1% |
| 7 | 0.3128 | 0.3865 | 83.9% | 83.6% |
| 8 | 0.2542 | 0.4241 | 83.6% | 83.9% |
| **9** | **0.2145** | **0.3851** | **84.3%** | **84.0% ← best** |
| 10 | 0.1937 | 0.4777 | 80.6% | 82.6% |

BiLSTM+GloVe (84.0% val F1) beats TF-IDF baseline (83.2% val F1). Overfitting visible at epoch 10 — checkpoint correctly saved at epoch 9.

What landed:
- `src/train.py` — `train_one_epoch`, `evaluate_epoch`, `train`: Adam + BCEWithLogitsLoss + grad clipping (max_norm=5), best checkpoint by val F1 to `outputs/bilstm.pt`
- Checkpoint contains model state, model config, vocab path, and full metrics history
- Auto-detects CUDA → MPS → CPU
- `tests/test_train.py` — 12 unit tests + 1 slow integration test, all green

---

## Issue #11 — Evaluation & error analysis (`evaluate.py`) ✅

What landed:
- `src/evaluate.py` — `load_checkpoint`, `collect_predictions`, `plot_confusion_matrix`, `error_analysis`, `run_evaluation`
- Outputs: `outputs/confusion_matrix.png`, `outputs/error_analysis.csv` (50 misclassified examples, evenly split FP/FN)
- Side-by-side comparison table printed at runtime: TF-IDF baseline vs BiLSTM+GloVe
- `tests/test_evaluate.py` — 11 unit tests + 1 slow integration test, all green

---

## Issues #12–21 — Pending

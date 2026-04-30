# ReviewPulse — Architecture

> Written for a new contributor. Describes where each concern lives today and names the intended boundaries for the v2.x refactor series.

---

## 1. Repository Layout

```
review-pulse/
├── app.py                  ← Streamlit UI (orchestration only)
├── src/
│   ├── parser.py           ← raw data → DataFrame
│   ├── preprocess.py       ← cleaning, auditing, splitting
│   ├── dataset.py          ← vocab, GloVe loader, PyTorch Dataset + DataLoaders
│   ├── features.py         ← EDA helpers (balance, length stats, plots)
│   ├── baseline.py         ← TF-IDF + LogReg: train, evaluate, load
│   ├── model.py            ← BiLSTMSentiment nn.Module
│   ├── train.py            ← BiLSTM training loop + checkpointing
│   ├── model_bert.py       ← DistilBERTSentiment nn.Module (HF wrapper)
│   ├── train_bert.py       ← DistilBERT training loop + checkpointing
│   ├── evaluate.py         ← batch evaluation: metrics, confusion matrix, error analysis
│   └── inference.py        ← single-text prediction; module-level model caching
├── outputs/                ← model artifacts (gitignored except distilbert.pt)
│   ├── baseline.joblib     ← trained TF-IDF + LogReg pipeline
│   ├── vocab.json          ← BiLSTM vocabulary
│   ├── bilstm.pt           ← BiLSTM checkpoint (epoch 9, val F1 84.0%)
│   └── distilbert.pt       ← DistilBERT checkpoint (epoch 12, val F1 87.8%)
├── data/                   ← raw Blitzer et al. 2007 review files (not in git)
│   ├── books/
│   ├── dvd/
│   ├── electronics/
│   └── kitchen_&_housewares/
├── embeddings/             ← GloVe vectors (not in git, ~800 MB)
├── tests/                  ← 143 unit tests + 5 slow integration tests
└── docs/                   ← project documentation
```

---

## 2. Four Distinct Concerns

The codebase has four roles that must stay separate. Conflating them is the primary source of complexity today.

| Role | What it does | Entrypoint |
|---|---|---|
| **Training** | reads raw data, fits a model, writes an artifact to `outputs/` | `python -m src.baseline`, `python -m src.train`, `python -m src.train_bert` |
| **Inference** | loads a cached artifact, cleans one text string, returns a prediction dict | `src.inference.predict_sentiment(text, model_name)` |
| **Evaluation** | loads artifacts + held-out data, computes batch metrics, writes reports | `python -m src.evaluate` |
| **App** | UI only — calls inference, renders results | `streamlit run app.py` |

The app **never** trains. Evaluation **never** writes model artifacts. Inference **never** reads raw data.

---

## 3. Data Flow

### 3.1 Shared preprocessing (all three models)

```
data/{domain}/positive.review
data/{domain}/negative.review
        │
        ▼
src/parser.py
  parse_review_file()      — BeautifulSoup pseudo-XML → list[dict]
  load_all_domains()       — iterates 4 domains → pd.DataFrame (8,000 rows)
        │
        ▼
src/preprocess.py
  audit_labels()           — flag is_ambiguous, rating_conflict
  drop_ambiguous()         — remove 3-star rows (0 dropped on this dataset)
  clean_text()             — lowercase, strip HTML, expand negations, remove punctuation
  remove_outliers()        — drop < 10 or > 500 words (279 dropped)
  split_data()             — stratified 70/15/15, seed=42
        │
        ▼
  train_df (5,404) · val_df (1,158) · test_df (1,159)
```

### 3.2 Baseline — TF-IDF + Logistic Regression

```
Training:
  train_df.text
        │
        ▼
  src/baseline.py
    build_pipeline()       — TfidfVectorizer(ngram_range=(1,2)) + LogisticRegression
    train_baseline()       — fits pipeline, saves → outputs/baseline.joblib
        │
        ▼
  outputs/baseline.joblib

Inference (one text):
  src/inference.py
    load_baseline_model()  — joblib.load, cached in _baseline_cache
    predict_baseline()     — clean_text → pipeline.predict_proba → dict
```

### 3.3 BiLSTM + GloVe

```
Training:
  train_df.text
        │
        ▼
  src/dataset.py
    build_vocab()          — token counts from train_df only (no leakage), min_freq=2
    save_vocab()           — outputs/vocab.json
    load_glove()           — maps vocab → 100d vectors (97.4% coverage)
    make_dataloaders()     — tokenize_and_pad (max_len=256), ReviewDataset, DataLoader
        │
        ▼
  src/model.py
    BiLSTMSentiment        — Embedding(15,924 × 100d) → Dropout → BiLSTM(256, 2, bidir)
                             → concat fwd+bwd hidden (512d) → Dropout → Linear(512→1)
        │
        ▼
  src/train.py
    train()                — Adam(lr=1e-3), BCEWithLogitsLoss, grad clip(5.0)
                             10 epochs, checkpoint by val F1
                             saves → outputs/bilstm.pt  (epoch 9, val F1 84.0%)

Inference (one text):
  src/inference.py
    load_bilstm_model()    — load_checkpoint + load_vocab, cached in _bilstm_cache
    predict_bilstm()       — clean_text → tokenize_and_pad → model → sigmoid → dict
```

### 3.4 DistilBERT (Hugging Face)

```
Training:
  train_df, val_df
        │
        ▼
  src/train_bert.py
    load_tokenizer()       — AutoTokenizer.from_pretrained("distilbert-base-uncased")
    make_bert_dataloaders()— encode_texts → BertReviewDataset → DataLoader

  src/model_bert.py
    DistilBERTSentiment    — DistilBertForSequenceClassification(num_labels=1)
                             + optional encoder freeze

  src/train_bert.py
    train_bert()           — Stage 1: freeze encoder, train head (head_epochs=10)
                             Stage 2: unfreeze last 2 layers, fine-tune (epochs=12)
                             Adam(head_lr=1e-4 / encoder_lr=2e-5), BCEWithLogitsLoss
                             ReduceLROnPlateau, checkpoint by val F1
                             saves → outputs/distilbert.pt  (epoch 12, val F1 87.8%)

Inference (one text):
  src/inference.py
    load_distilbert_model()— load_pretrained_bert_bundle (model + tokenizer + checkpoint)
                             cached in _distilbert_cache
    predict_distilbert()   — clean_text → tokenizer → model → sigmoid → dict
```

---

## 4. Inference API

All three models are accessible through a single entry point:

```python
from src.inference import predict_sentiment

result = predict_sentiment("This blender is great!", model_name="baseline")
# → {"label": "Positive review", "confidence": 0.923, "model": "baseline"}
```

`model_name` accepts `"baseline"` (default), `"bilstm"`, or `"distilbert"`.

Each model is loaded once per process and cached at module level:
- `_baseline_cache` — sklearn Pipeline
- `_bilstm_cache` — (BiLSTMSentiment, vocab dict, device)
- `_distilbert_cache` — (DistilBERTSentiment, tokenizer, checkpoint dict, device)

`clean_text()` from `src/preprocess.py` is called by every prediction path before the model sees the text.

---

## 5. Evaluation

```
python -m src.evaluate
```

Runs sequentially:

1. `run_evaluation()` — loads `bilstm.pt` + `baseline.joblib`, runs both on `test_df`, prints comparison table, writes `outputs/confusion_matrix.png` and `outputs/error_analysis.csv`.
2. `check_distilbert_and_evaluate()` — if `outputs/distilbert.pt` exists and `transformers` is importable, runs `run_evaluation_distilbert_deploy()`, writes `outputs/confusion_matrix_distilbert_deploy.png` and `outputs/error_analysis_distilbert_deploy.csv`.

Verified results (held-out test split, 1,159 reviews, seed=42):

| Model | Accuracy | F1 | Misclassified |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 82.7% | 81.9% | 201 |
| BiLSTM + GloVe | 81.0% | 80.3% | 220 |
| DistilBERT | 88.2% | 88.6% | 137 |

---

## 6. Streamlit App

```
streamlit run app.py
```

`app.py` is UI orchestration only. It:
- calls `_load_baseline()`, `_load_bilstm()`, `_load_distilbert()` via `@st.cache_resource` — each wraps the corresponding `src.inference` loader
- calls `predict_sentiment(text, model_name)` on button click
- catches `ImportError / FileNotFoundError / RuntimeError` from DistilBERT loading and shows a warning instead of crashing

The app does not import `src.train`, `src.baseline` (training path), or `src.parser` directly.

---

## 7. Model Artifacts

| Artifact | Produced by | Consumed by | Size |
|---|---|---|---|
| `outputs/vocab.json` | `src/dataset.save_vocab()` | `src/inference.load_bilstm_model()` | ~350 KB |
| `outputs/baseline.joblib` | `src/baseline.train_baseline()` | `src/inference.load_baseline_model()` | ~5 MB |
| `outputs/bilstm.pt` | `src/train.train()` | `src/inference.load_bilstm_model()`, `src/evaluate.run_evaluation()` | ~25 MB |
| `outputs/distilbert.pt` | `src/train_bert.train_bert()` | `src/inference.load_distilbert_model()`, `src/evaluate.check_distilbert_and_evaluate()` | ~29 MB |

`bilstm.pt` and `baseline.joblib` are gitignored. `distilbert.pt` is currently committed (tracked in Issue #28 for migration to external hosting).

---

## 8. Test Coverage

```
pytest tests/ -q -m "not slow"   # 143 unit tests
pytest tests/                     # + 5 slow integration tests
```

BERT-specific test files (`test_model_bert.py`, `test_train_bert.py`) use `pytest.importorskip("transformers")` and skip cleanly when the optional dependency is absent.

---

## 9. Target Boundaries (Refactor Series)

The current codebase works but has accumulated coupling across the training/inference/evaluation boundary. Intended targets for v2.x:

| Concern | Current state | Target state |
|---|---|---|
| **Artifact paths** | `OUTPUTS_DIR`, `CHECKPOINT_PATH`, `DEPLOY_CHECKPOINT_PATH` duplicated across `dataset.py`, `inference.py`, `train_bert.py` | Single `src/config.py` that owns all path constants |
| **DistilBERT checkpoint** | Committed to git (~29 MB) | Hosted on luisfaria webserver (Issue #28); optional download fallback |
| **evaluate.py** | BiLSTM and DistilBERT evaluation mixed in one module | Separate `evaluate_bilstm()` and `evaluate_distilbert()` entry points with a common metrics helper |
| **train_bert.py** | 700+ lines; tokenizer loading, dataset, training loop, and checkpoint serialisation all in one file | Split into `tokenizer.py`, `dataset_bert.py`, `train_bert.py` |
| **inference.py** | `load_checkpoint()` lives here for historical reasons but is only used by `evaluate.py` | Move to `evaluate.py` or a shared `checkpoint.py` |

These are documentation targets only — no code changes in this issue.

# Tests

```bash
.venv/bin/pytest tests/ -q
.venv/bin/pytest tests/ -q -m "not slow"   # skip real-data integration tests
```

| File | Covers | Tests |
|---|---|---|
| `test_parser.py` | `src/parser.py` — pseudo-XML parsing, domain loading, missing-field handling | 10 |
| `test_preprocess.py` | `src/preprocess.py` — label audit, text cleaning, outlier removal, split reproducibility | 21 |
| `test_dataset.py` | `src/dataset.py` — vocab no-leakage, tokenisation, padding length, Dataset, DataLoader | 21 |
| `test_baseline.py` | `src/baseline.py` — TF-IDF + LogReg pipeline | 9 unit + 1 slow |
| `test_model.py` | `src/model.py` — BiLSTMSentiment architecture, forward-pass shape, packed-sequence correctness | 22 |
| `test_train.py` | `src/train.py` — training loop, checkpoint keys, history length | 12 unit + 1 slow |
| `test_evaluate.py` | `src/evaluate.py` — checkpoint loading, predictions, confusion matrix, error analysis | 11 unit + 1 slow |
| `test_inference.py` | `src/inference.py` — inference response shape, model routing, invalid model rejection | 11 unit + 2 slow |

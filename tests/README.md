# Tests

```bash
.venv/bin/pytest tests/ -q
.venv/bin/pytest tests/ -q -m "not slow"   # skip real-data integration tests
```

| File | Covers |
|---|---|
| `test_parser.py` | `src.data.parser` — pseudo-XML parsing, domain loading, missing-field handling |
| `test_preprocess.py` | `src.data.preprocess` — label audit, text cleaning, outlier removal, split reproducibility |
| `test_dataset.py` | `src.tokenization.*` — vocab no-leakage, tokenisation, padding length, Dataset, DataLoader |
| `test_baseline.py` | `src.training.baseline` — TF-IDF + LogReg pipeline |
| `test_model.py` | `src.models.bilstm` — BiLSTMSentiment architecture, forward-pass shape, packed-sequence correctness |
| `test_train.py` | `src.training.bilstm` — training loop, checkpoint keys, history length |
| `test_evaluate.py` | `src.evaluation` — checkpoint loading, predictions, confusion matrix, error analysis |
| `test_inference.py` | `src.inference` — inference response shape, model routing, invalid model rejection |
| `test_wrapper_removal.py` | Issue #59 wrapper deletion and canonical import paths |

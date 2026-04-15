# Tests

```bash
.venv/bin/pytest tests/ -q
.venv/bin/pytest tests/ -q -m "not slow"   # skip real-data integration tests
```

| File | Covers | Tests |
|---|---|---|
| `test_parser.py` | `src/parser.py` — pseudo-XML parsing, domain loading | 10 |
| `test_preprocess.py` | `src/preprocess.py` — label audit, cleaning, splits | 21 |
| `test_dataset.py` | `src/dataset.py` — vocab, GloVe loader, Dataset, DataLoader | 21 |
| `test_baseline.py` | `src/baseline.py` — TF-IDF + LogReg pipeline | 9 unit + 1 slow |
| `test_model.py` | `src/model.py` — BiLSTMSentiment architecture, forward pass | 21 |

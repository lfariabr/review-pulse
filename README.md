# ReviewPulse

Multi-domain Amazon product review sentiment classifier. Trained on 8,000 labelled reviews across Books, DVDs, Electronics, and Kitchen & Housewares.

Built for ISY503 Intelligent Systems, Assessment 3.

[Main repo with +800 commits in ~1 year of work](https://github.com/lfariabr/masters-swe-ai)

## Project structure

```
review-pulse/
  app.py               # Streamlit inference UI
  requirements.txt
  src/
    parser.py          # pseudo-XML parser → DataFrame
    preprocess.py      # label audit, cleaning, outlier removal, splits
    features.py        # EDA helpers
    dataset.py         # vocab, GloVe loader, PyTorch Dataset/DataLoader
    baseline.py        # TF-IDF + LogisticRegression pipeline
    model.py           # BiLSTMSentiment nn.Module
    train.py           # training loop + checkpointing
    inference.py       # predict_sentiment() shared by app and evaluate
    evaluate.py        # metrics, confusion matrix, error analysis
  tests/               # pytest test suite
  notebooks/           # EDA.ipynb, optional distilbert.ipynb
  data/                # local .review files (not committed)
  outputs/             # generated artifacts (not committed)
  embeddings/          # optional GloVe files (not committed)
  docs/                # planning docs, diagrams, submission checklists
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Windows:** use `.venv\Scripts\activate` instead.

Place `.review` data files under `data/` in the following structure:

```
data/
  books/positive.review
  books/negative.review
  dvd/positive.review
  dvd/negative.review
  electronics/positive.review
  electronics/negative.review
  kitchen_&_housewares/positive.review
  kitchen_&_housewares/negative.review
```

### GloVe embeddings (optional)

GloVe pre-trained vectors can improve accuracy but are not required to run the project.

| | Without GloVe | With GloVe |
|---|---|---|
| Embeddings | Learned from scratch | Initialised from GloVe 100d |
| Setup | Nothing extra needed | Download ~800 MB file |
| Expected accuracy | ~88–91% | ~92–95% |

To enable GloVe:
1. Download `glove.6B.zip` from [Stanford NLP](https://nlp.stanford.edu/projects/glove/)
2. Extract and place `glove.6B.100d.txt` in the `embeddings/` directory
3. Re-run training — `load_glove()` detects the file automatically

If `embeddings/glove.6B.100d.txt` is absent, training proceeds with randomly initialised embeddings and a warning is printed. The `embeddings/` directory is gitignored.

## Train

Train the TF-IDF baseline:

```bash
python -m src.baseline
```

Train the BiLSTM:

```bash
python -m src.train
```

## Evaluate

```bash
python -m src.evaluate
```

Outputs saved to `outputs/`: confusion matrix, error examples CSV, and per-model metrics.

## Run the app

```bash
streamlit run app.py
```

Enter a review in the text area, select a model (BiLSTM or Baseline), and click **Classify**.

## Run tests

```bash
pytest tests/
```

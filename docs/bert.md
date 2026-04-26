# ReviewPulse Issue #21: Clean DistilBERT Transformers Branch

## Summary

Create a clean branch from `main`, named `feature/issue-21-distilbert-transformers`, to implement an assessment-safe pretrained DistilBERT path using Hugging Face `transformers`.

The branch should replace the current PR direction with a focused implementation: no custom `TransformerEncoder`, no MLM pretraining, no generated vocab artifact, and no RoBERTa implementation in this branch. RoBERTa should be documented as the next natural extension.

## Key Changes

- Add a pretrained DistilBERT model path using `AutoTokenizer` and `AutoModel` or `AutoModelForSequenceClassification`.
- Use a custom PyTorch training loop consistent with the existing `src/train.py` style:
  - train/validation split from the existing preprocessing flow
  - `BCEWithLogitsLoss` or Hugging Face sequence-classification loss
  - validation F1 checkpointing
  - default `freeze_encoder=True`
  - optional full fine-tuning controlled by an explicit argument
- Save only the trained checkpoint needed for local inference, but do not commit generated artifacts.
- Update `.gitignore` for transformer checkpoints and generated transformer outputs.
- Add `model_name="distilbert"` support to the existing inference API.
- Add a third Streamlit option for DistilBERT, disabled with a clear note if the checkpoint is missing.
- Keep `transformers` either optional or clearly documented as required only for Issue #21 transformer support.

## Public Interface

`predict_sentiment(text, model_name="distilbert")` should return the same result shape as baseline/BiLSTM:

```python
{
    "label": "Positive review" | "Negative review",
    "confidence": float,
    "model": "distilbert",
}
```

Add a DistilBERT training entrypoint:

```bash
python -m src.train_distilbert
```

Add DistilBERT evaluation support without breaking the existing default:

```bash
python -m src.evaluate
```

`python -m src.evaluate` should still evaluate baseline/BiLSTM safely. DistilBERT evaluation should run only when its checkpoint exists or when explicitly requested.

## Test Plan

- Unit test the DistilBERT model wrapper with mocked or small transformer outputs where possible.
- Unit test inference routing for `model_name="distilbert"`.
- Unit test missing-checkpoint behavior so baseline/BiLSTM still work.
- Unit test app model-option logic at the function/helper level if UI logic is extractable.
- Run:

```bash
pytest tests/ -q -m "not slow"
python -m src.evaluate
```

- Optional manual acceptance:
  - Train DistilBERT on the real split.
  - Record validation/test F1.
  - Compare against TF-IDF and BiLSTM in `docs/issueBreakdown.md`.

## Assumptions

- Branch starts from `main`, not Victor's PR branch.
- DistilBERT is the only implemented transformer model in this branch.
- RoBERTa is documented as future work, not built now.
- Generated transformer checkpoints are not committed to git.
- The app remains usable without DistilBERT artifacts present.

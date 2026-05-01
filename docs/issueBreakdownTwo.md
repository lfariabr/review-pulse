# Issue Breakdown II - ReviewPulse Refactor Track

This document tracks the second wave of ReviewPulse work: moving the project from a working assessment build into a cleaner, easier-to-maintain codebase.

The v2.0.0 application already works with three model families:

- baseline: TF-IDF + Logistic Regression
- neural: BiLSTM + GloVe
- transformer: Hugging Face DistilBERT

The refactor track is not about chasing higher metrics first. It is about making the codebase easier to understand, safer to extend, and cleaner to explain in a portfolio or technical interview.

## Mental Model

Keep these four concerns separate:

| Concern | Question it answers | Current entrypoint |
|---|---|---|
| Training | How do we create model artifacts? | `python -m src.baseline`, `python -m src.train`, `python -m src.train_bert` |
| Inference | How do we predict one user-provided review? | `src.inference.predict_sentiment()` |
| Evaluation | How do we score models on the held-out test split? | `python -m src.evaluate` |
| App | How does the user interact with the models? | `streamlit run app.py` |

Training produces artifacts. Inference consumes artifacts for one review. Evaluation consumes artifacts and test data for metrics. The app should only orchestrate UI and call inference.

---

## Status Snapshot

| Issue | Status | Purpose |
|---|---|---|
| #30 | Closed | Architecture documentation and target boundaries |
| #31 | Open | Centralize paths, model names, and shared constants |
| #32 | Open | Introduce a predictor interface for all models |
| #33 | Open | Replace inference branching with a model registry |
| #34 | Open | Separate evaluation metrics from artifact/report writing |
| #35 | Open | Reuse shared prediction logic in evaluation where practical |
| #36 | Open | Split DistilBERT training and checkpoint concerns |
| #37 | Open | Move Streamlit model availability/loading into an app service |
| #38 | Open | Document model artifact policy and model-card notes |
| #39 | Open | Reorganize tests by concern and add refactor safety checks |

---

## Issue #30 - Architecture: document current pipeline and target boundaries

Status: Closed

What landed:

- `docs/architecture.md` explains the current repository layout.
- Training, inference, evaluation, and app responsibilities are documented as separate concerns.
- Baseline, BiLSTM, and DistilBERT data flows are documented.
- Mermaid diagrams were added for the model pipelines.
- Target refactor boundaries were named for the v2.x cleanup series.

Why it matters:

- This creates a shared map before changing code.
- It helps avoid refactoring blindly.
- It gives new contributors a way to understand why the next issues exist.

---

## Issue #31 - Refactor: centralize paths, model names, and shared constants

Status: Open

Goal:

- Create one source of truth for artifact paths, model identifiers, labels, display names, and shared thresholds.

Expected direction:

- Add a small config/constants module, likely `src/config.py`.
- Move duplicated constants such as `OUTPUTS_DIR`, checkpoint paths, baseline path, vocab path, and valid model names into that module.
- Update modules to import shared constants instead of redefining them.

Acceptance shape:

- No behavior change.
- Existing commands still work.
- Fast tests remain green.

Why it comes first:

- Later issues need stable names and paths before introducing predictors, registries, and cleaner evaluation helpers.

---

## Issue #32 - Refactor: introduce a predictor interface for all models

Status: Open

Goal:

- Give baseline, BiLSTM, and DistilBERT the same single-text prediction contract.

Expected direction:

- Add a simple predictor shape or protocol.
- Wrap each model behind a common method, for example `predict(text: str) -> dict`.
- Preserve the current output contract:

```python
{
    "label": "Positive review" | "Negative review",
    "confidence": float,
    "model": "baseline" | "bilstm" | "distilbert",
}
```

Acceptance shape:

- `predict_sentiment()` keeps working.
- Model loading remains cached.
- Predictors do not import evaluation plotting code.

Why it matters:

- This makes inference easier to reason about because every model follows the same interface.

---

## Issue #33 - Refactor: simplify inference with a model registry

Status: Open

Goal:

- Replace growing `if/elif` model routing with a small registry.

Expected direction:

- Add a mapping from model name to predictor factory or predictor loader.
- Keep `"baseline"` as the default model.
- Keep `"bilstm"` and `"distilbert"` selectable.
- Preserve clear `ValueError` behavior for invalid model names.

Acceptance shape:

- `predict_sentiment(text, model_name="baseline")` remains the public API.
- Adding a future model should require registering it, not editing multiple conditionals.

Why it matters:

- This is the bridge from "three hardcoded models" to "a maintainable model comparison app".

---

## Issue #34 - Refactor: separate evaluation metrics from artifact writing

Status: Open

Goal:

- Make evaluation easier to test by separating pure metric computation from side effects.

Expected direction:

- Keep metric helpers separate from CSV/PNG/report writing.
- Preserve `python -m src.evaluate`.
- Keep confusion matrix and error analysis outputs.

Acceptance shape:

- Metrics can be tested without writing files.
- Report writing remains available for the assessment evidence.

Why it matters:

- Evaluation is currently one of the confusing areas because it mixes scoring, plotting, error analysis, and model loading.

---

## Issue #35 - Refactor: reuse shared prediction logic in evaluation where practical

Status: Open

Goal:

- Reduce duplicated prediction behavior between inference and evaluation without making evaluation inefficient.

Expected direction:

- Reuse labels, thresholds, constants, and model contracts from the inference side.
- Keep batch evaluation efficient for neural models.
- Do not force the Streamlit single-text path to depend on plotting or evaluation files.

Acceptance shape:

- Existing metrics stay materially the same.
- Evaluation and inference agree on labels, thresholds, and model names.

Why it matters:

- Inference and evaluation should be consistent, but not tightly coupled.

---

## Issue #36 - Refactor: split DistilBERT training and checkpoint concerns

Status: Open

Goal:

- Break down `src/train_bert.py`, currently the largest and densest module.

Expected direction:

- Separate tokenizer/dataset helpers from the training loop.
- Separate checkpoint save/load helpers from CLI orchestration.
- Preserve compatibility with `outputs/distilbert.pt`.
- Preserve `python -m src.train_bert`.

Acceptance shape:

- DistilBERT tests pass.
- Existing checkpoint can still be loaded.
- No retraining is required for the refactor.

Why it matters:

- The transformer implementation is valuable, but it is also the easiest part of the codebase to let become hard to maintain.

---

## Issue #37 - Refactor: move Streamlit model availability and loading into an app service

Status: Open

Goal:

- Keep `app.py` focused on UI and move app-specific loading/availability checks elsewhere.

Expected direction:

- Add a small app service/helper module.
- Move model availability checks into that layer.
- Preserve graceful DistilBERT unavailable behavior.
- Keep core inference in `src.inference`, not inside the UI.

Acceptance shape:

- `streamlit run app.py` behaves the same.
- The app still uses the inference API or predictor layer.
- App code becomes easier to scan.

Why it matters:

- UI files become fragile when they also own loading policy, error handling, and model orchestration.

---

## Issue #38 - Docs: add model artifact policy and model card notes

Status: Open

Goal:

- Document what each artifact is, how it is produced, and how it should be handled.

Expected direction:

- Document `baseline.joblib`, `vocab.json`, `bilstm.pt`, and `distilbert.pt`.
- Explain which artifacts are committed and which generated files are ignored.
- Explain that the compact DistilBERT checkpoint still depends on the Hugging Face base model.
- Document security and trust considerations around loading model artifacts.
- Reference #28 for future external hosting of the DistilBERT checkpoint.

Acceptance shape:

- A maintainer can understand artifact ownership without reading every training file.
- Deployment limitations are explicit.

Why it matters:

- Model artifacts are part of the architecture, not just random files in `outputs/`.

---

## Issue #39 - Tests: reorganize tests by concern and add refactor safety checks

Status: Open

Goal:

- Make the tests describe the intended architecture and protect the refactor work.

Expected direction:

- Add or reorganize tests around:
  - config/constants;
  - predictor interface behavior;
  - model registry routing;
  - invalid model handling;
  - metric helpers without file writes;
  - app model availability helpers.
- Keep slow tests marked.
- Keep BERT tests guarded when `transformers` is unavailable.

Acceptance shape:

- Fast tests pass:

```bash
pytest tests/ -q -m "not slow"
```

- Full tests pass when local data/artifacts are available:

```bash
pytest tests/
```

Why it matters:

- Refactors are only senior-level work when tests protect behavior while the internals move.

---

## Recommended Execution Order

1. #31 - centralize config/constants.
2. #32 - introduce predictor interface.
3. #33 - add model registry.
4. #34 - split evaluation metrics from file/report writing.
5. #35 - align evaluation with shared prediction contracts where practical.
6. #36 - split DistilBERT training/checkpoint code.
7. #37 - move app loading/availability policy into an app service.
8. #38 - document artifact policy and model-card notes.
9. #39 - tighten/refactor tests around the new boundaries.

This order keeps each PR small and reviewable. It also avoids touching Streamlit or DistilBERT internals before the shared constants and prediction contracts are stable.

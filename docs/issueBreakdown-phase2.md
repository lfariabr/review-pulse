# Issue Breakdown II - ReviewPulse Refactor Track

This document records the second wave of ReviewPulse work: moving the project from a working assessment build into a cleaner, safer, more maintainable codebase.

The v2.0.0 application already worked with three model families:

- baseline: TF-IDF + Logistic Regression
- neural: BiLSTM + GloVe
- transformer: Hugging Face DistilBERT

The #30-#39 track did not chase higher metrics. It clarified architecture, reduced coupling, hardened model loading, and added tests that protect the current boundaries.

## Mental Model

Keep these four concerns separate:

| Concern | Question it answers | Current entrypoint |
|---|---|---|
| Training | How do we create model artifacts? | `python -m src.baseline`, `python -m src.train`, `python -m src.train_bert` |
| Inference | How do we predict one user-provided review? | `src.inference.predict_sentiment()` |
| Evaluation | How do we score models on the held-out test split? | `python -m src.evaluate` |
| App | How does the user interact with the models? | `streamlit run app.py` |

Training produces artifacts. Inference consumes artifacts for one review. Evaluation consumes artifacts and test data for metrics. The app orchestrates UI and calls inference through the app service layer.

---

## Status Snapshot

All issues in this refactor track are closed.

| Issue | Status | Delivered |
|---|---|---|
| #30 | Closed | Architecture documentation and target boundaries |
| #31 | Closed | Centralized paths, model names, and shared constants |
| #32 | Closed | Predictor interface and per-model predictor classes |
| #33 | Closed | Model registry and runtime registration API |
| #34 | Closed | Evaluation helpers can compute without file writes |
| #35 | Closed | Shared metrics helper and boundary tests |
| #36 | Closed | DistilBERT training/checkpoint split |
| #37 | Closed | Streamlit app service layer and sample utility |
| #38 | Closed | Model artifact policy and DistilBERT model-card notes |
| #39 | Closed | Config contract tests and module boundary tests |

---

## Issue #30 - Architecture: document current pipeline and target boundaries

Status: Closed

What landed:

- `docs/architecture.md` explains repository layout and current boundaries.
- Training, inference, evaluation, and app responsibilities are documented separately.
- Baseline, BiLSTM, and DistilBERT data flows are documented.
- Mermaid diagrams were added for the model pipelines.
- The document now includes model artifact policy and DistilBERT model-card notes.

Why it matters:

- New contributors can understand the system before editing code.
- It creates a defensible portfolio artifact: not just code, but architecture reasoning.

---

## Issue #31 - Refactor: centralize paths, model names, and shared constants

Status: Closed

What landed:

- `src/config.py` became the single source of truth for:
  - `OUTPUTS_DIR`
  - `BASELINE_PATH`
  - `BILSTM_CHECKPOINT_PATH`
  - `VOCAB_PATH`
  - `DISTILBERT_PATH`
  - `MODEL_BASELINE`
  - `MODEL_BILSTM`
  - `MODEL_DISTILBERT`
  - `ALL_MODELS`
  - `PRED_THRESHOLD`
- Downstream modules now import shared constants instead of redefining paths and model names.
- `tests/test_config.py` documents the config contract.

Why it matters:

- Model names and artifact paths no longer drift across modules.
- Later refactors can move code without changing the meaning of core constants.

---

## Issue #32 - Refactor: introduce a predictor interface for all models

Status: Closed

What landed:

- `src/inference.py` now defines a `Predictor` protocol.
- The three model paths are wrapped by:
  - `BaselinePredictor`
  - `BiLSTMPredictor`
  - `DistilBERTPredictor`
- The old flat functions remain as compatibility delegates:
  - `predict_baseline()`
  - `predict_bilstm()`
  - `predict_distilbert()`
- `predict_sentiment()` still returns the same response shape.

Why it matters:

- All single-text prediction paths now follow one interface.
- Future models can plug into the same contract instead of adding new ad hoc branches.

---

## Issue #33 - Refactor: simplify inference with a model registry

Status: Closed

What landed:

- `_PREDICTORS` maps model name to predictor instance.
- `predict_sentiment()` dispatches through registry lookup.
- `register_predictor(name, predictor, overwrite=False)` allows future models such as RoBERTa to be registered without editing the dispatch function.
- `get_available_models()` returns the live registry keys.
- Invalid predictor objects and accidental duplicate registrations are rejected.

Why it matters:

- The app is now a real model-comparison shell, not a hardcoded three-branch script.
- Adding a future model becomes a registration problem rather than an inference rewrite.

---

## Issue #34 - Refactor: separate evaluation metrics from artifact writing

Status: Closed

What landed:

- `plot_confusion_matrix(..., save_path=None)` computes and returns the confusion matrix without writing a file.
- `error_analysis(..., save_path=None)` computes misclassified examples without writing CSV.
- `run_evaluation(..., save_outputs=False)` can run evaluation without PNG/CSV side effects.
- Tests verify no-write behavior with monkeypatched default output paths.

Why it matters:

- Evaluation logic is easier to test.
- Report generation remains available for assessment evidence, but it is no longer inseparable from metric computation.

---

## Issue #35 - Refactor: reuse shared prediction logic in evaluation where practical

Status: Closed

What landed:

- `compute_metrics(y_true, y_pred)` is a public pure helper for accuracy and F1.
- Evaluation and inference share `PRED_THRESHOLD` from `src.config`.
- `collect_predictions()` documents why batch evaluation remains separate from single-text inference.
- A subprocess-based boundary test ensures importing `src.inference` does not pull in matplotlib.

Why it matters:

- Inference and evaluation agree on labels, threshold, and metrics without making the Streamlit path depend on plotting or batch evaluation code.

---

## Issue #36 - Refactor: split DistilBERT training and checkpoint concerns

Status: Closed

What landed:

- `src/dataset_bert.py` owns:
  - device resolution
  - tokenizer loading
  - `BertReviewDataset`
  - DistilBERT DataLoader factories
- `src/checkpoint_bert.py` owns:
  - tokenizer serialization
  - compact checkpoint saving
  - checkpoint bundle loading
  - save strategy validation for `head_only`, `partial_encoder`, and `full`
- `src/train_bert.py` now focuses on:
  - training epochs
  - stage orchestration
  - optimizer setup
  - CLI flow
- Existing imports continue to work through re-exports in `src/train_bert.py`.

Why it matters:

- The largest and riskiest file was split by responsibility without changing behavior.
- The compact DistilBERT checkpoint behavior is now explicit and test-covered.

---

## Issue #37 - Refactor: move Streamlit model availability and loading into an app service

Status: Closed

What landed:

- `src/app_service.py` owns:
  - `MODEL_OPTIONS`
  - cached model loaders
  - `warm_up_model()`
  - `is_distilbert_available()`
  - DistilBERT unavailable message
- `src/utils/samples.py` owns demo review samples and sample selection.
- `app.py` now focuses on layout, input, button state, and result display.
- DistilBERT unavailable behavior catches `ImportError`, `FileNotFoundError`, `RuntimeError`, and `OSError`.
- `.streamlit/config.toml` uses `fileWatcherType = "poll"` to avoid noisy transformers watcher behavior.

Why it matters:

- The UI file is smaller and easier to reason about.
- Model loading policy is testable outside Streamlit page rendering.

---

## Issue #38 - Docs: add model artifact policy and model card notes

Status: Closed

What landed:

- `docs/architecture.md` now documents all committed model artifacts:
  - `outputs/baseline.joblib`
  - `outputs/vocab.json`
  - `outputs/bilstm.pt`
  - `outputs/distilbert.pt`
- DistilBERT checkpoint format is documented:
  - `model_config`
  - `model_state`
  - `weights_dtype`
  - `save_strategy`
  - `trainable_encoder_layers`
  - tokenizer files
  - validation history
- DistilBERT model-card notes document:
  - training setup
  - metrics
  - compact checkpoint behavior
  - known failure modes
  - security and deployment caveats

Why it matters:

- Model artifacts are now treated as part of the architecture, not random files.
- The limitations around `torch.load`, Hugging Face cache/network dependency, and committed binary artifacts are explicit.

---

## Issue #39 - Tests: reorganize tests by concern and add refactor safety checks

Status: Closed

What landed:

- `tests/test_config.py` protects the `src.config` contract.
- `tests/test_boundaries.py` protects module boundaries:
  - config must not import the rest of `src`
  - inference must not import training/evaluation/parser
  - app service must not import training/evaluation/parser
- App service, registry, no-write evaluation, and BERT checkpoint paths have focused tests.
- Current test status:

```bash
pytest tests/ -q -m "not slow"   # 189 passed, 5 deselected
pytest tests/                    # 194 passed
```

Why it matters:

- The refactor series is now protected by tests that encode architecture decisions, not only function outputs.

---

## Final Outcome

The #30-#39 refactor series moved ReviewPulse from "working assessment project" toward "maintainable ML application":

- constants centralized;
- inference normalized behind predictors and registry;
- evaluation made more testable;
- DistilBERT training split by concern;
- Streamlit model loading moved out of UI;
- model artifacts documented;
- architecture boundary tests added.

The next improvement is a deeper modular package refactor. That plan is tracked in `docs/issueBreakdown-phase3.md`.

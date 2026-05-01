# Issue Breakdown III - Modular Package Refactor Plan

This document proposes the next ReviewPulse refactor wave: moving from a flat `src/` folder into clearer packages by concern.

This is intentionally a future plan, not a completed implementation log. The #30-#39 series cleaned the highest-risk coupling first. The next wave should improve package organization while preserving all current behavior.

## Current Problem

`src/` now contains many useful modules, but the folder still reads like an implementation timeline:

```text
src/
  app_service.py
  baseline.py
  checkpoint_bert.py
  config.py
  dataset.py
  dataset_bert.py
  evaluate.py
  features.py
  inference.py
  model.py
  model_bert.py
  parser.py
  preprocess.py
  train.py
  train_bert.py
  utils/
```

This works, but it makes ownership harder to see:

- model definitions live next to training scripts;
- tokenization/data loading lives beside generic dataset helpers;
- inference contains predictors, loaders, registry, and checkpoint loading;
- evaluation contains metrics, plotting, error analysis, and runners;
- app helpers live at the top level.

The next refactor should group modules by architectural concern.

## Target Shape

Proposed package layout:

```text
src/
  config.py

  data/
    parser.py
    preprocess.py
    features.py

  tokenization/
    vocab.py
    sequence.py
    bert.py

  models/
    baseline.py
    bilstm.py
    bert.py

  training/
    baseline.py
    bilstm.py
    bert.py

  inference/
    loaders.py
    predictors.py
    registry.py
    api.py

  evaluation/
    metrics.py
    plots.py
    errors.py
    bilstm.py
    bert.py
    runner.py

  app/
    service.py
    samples.py
```

## Compatibility Strategy

Do not do a big-bang move.

Every move should keep the old public module as a wrapper until all imports are migrated:

```text
src/inference.py      -> re-export from src/inference/api.py
src/evaluate.py       -> re-export from src/evaluation/runner.py
src/train.py          -> re-export from src/training/bilstm.py
src/train_bert.py     -> re-export from src/training/bert.py
src/model.py          -> re-export from src/models/bilstm.py
src/model_bert.py     -> re-export from src/models/bert.py
src/dataset.py        -> re-export from tokenization/training helpers during migration
```

This keeps current commands working:

```bash
python -m src.baseline
python -m src.train
python -m src.train_bert
python -m src.evaluate
streamlit run app.py
pytest tests/
```

## Proposed Issue Track

### Issue #50 - Modular refactor: create package skeleton and import policy

Goal:

- Add package folders with `__init__.py` files only.
- Do not move behavior yet.
- Document import rules in this file and/or `docs/architecture.md`.

Files expected:

- `src/data/__init__.py`
- `src/tokenization/__init__.py`
- `src/models/__init__.py`
- `src/training/__init__.py`
- `src/inference/__init__.py`
- `src/evaluation/__init__.py`
- `src/app/__init__.py`

Acceptance:

- No behavior changes.
- Full test suite passes.

---

### Issue #51 - Modular refactor: move app service and samples

Goal:

- Move `src/app_service.py` to `src/app/service.py`.
- Move `src/utils/samples.py` to `src/app/samples.py`.
- Keep compatibility wrappers:
  - `src/app_service.py`
  - `src/utils/samples.py`

Acceptance:

- `streamlit run app.py` still works.
- `tests/test_app_service.py` passes.
- Existing imports still work.

Why first:

- App service is already isolated and low-risk.

---

### Issue #52 - Modular refactor: move model definitions

Goal:

- Move model definitions into `src/models/`.

Proposed moves:

- `src/model.py` -> `src/models/bilstm.py`
- `src/model_bert.py` -> `src/models/bert.py`
- baseline pipeline helpers may move later; avoid mixing too much in this issue.

Compatibility:

- Keep `src/model.py` and `src/model_bert.py` as re-export wrappers.

Acceptance:

- `tests/test_model.py` passes.
- `tests/test_model_bert.py` passes.
- Inference and training imports still work.

---

### Issue #53 - Modular refactor: move tokenization and dataset helpers

Goal:

- Separate text/vocab/tokenization concerns from training concerns.

Proposed moves:

- vocabulary helpers from `src/dataset.py` -> `src/tokenization/vocab.py`
- `tokenize_and_pad()` / sequence constants -> `src/tokenization/sequence.py`
- `src/dataset_bert.py` -> `src/tokenization/bert.py`

Compatibility:

- Keep `src/dataset.py` and `src/dataset_bert.py` wrappers until all imports migrate.

Acceptance:

- `tests/test_dataset.py` passes.
- `tests/test_train.py` passes.
- `tests/test_train_bert.py` passes.
- No retraining required.

Risk:

- This is one of the highest-coupling moves because BiLSTM training, inference, and evaluation all touch `dataset.py`.

---

### Issue #54 - Modular refactor: move training modules

Goal:

- Move training orchestration into `src/training/`.

Proposed moves:

- `src/train.py` -> `src/training/bilstm.py`
- `src/train_bert.py` -> `src/training/bert.py`
- baseline training logic from `src/baseline.py` -> `src/training/baseline.py` if it can be done without mixing too much scope.

Compatibility:

- Keep `src/train.py`, `src/train_bert.py`, and `src/baseline.py` as CLI-compatible wrappers.

Acceptance:

```bash
python -m src.train
python -m src.train_bert
python -m src.baseline
pytest tests/test_train.py tests/test_train_bert.py tests/test_baseline.py -q
```

---

### Issue #55 - Modular refactor: split inference package

Goal:

- Split `src/inference.py` by responsibility.

Proposed modules:

- `src/inference/loaders.py` - artifact/model loading
- `src/inference/predictors.py` - `Predictor`, `BaselinePredictor`, `BiLSTMPredictor`, `DistilBERTPredictor`
- `src/inference/registry.py` - `_PREDICTORS`, `register_predictor()`, `get_available_models()`
- `src/inference/api.py` - `predict_sentiment()` and compatibility-facing API

Compatibility:

- Keep `src/inference.py` as a re-export wrapper.

Acceptance:

- `tests/test_inference.py` passes.
- `tests/test_boundaries.py` passes.
- `app.py` still calls `predict_sentiment()`.

Risk:

- Keep inference free of `matplotlib`, `src.evaluate`, training modules, and raw-data parser imports.

---

### Issue #56 - Modular refactor: split evaluation package

Goal:

- Split `src/evaluate.py` by responsibility.

Proposed modules:

- `src/evaluation/metrics.py` - `compute_metrics()`
- `src/evaluation/plots.py` - `plot_confusion_matrix()`
- `src/evaluation/errors.py` - `error_analysis()`
- `src/evaluation/bilstm.py` - BiLSTM batch evaluation helpers
- `src/evaluation/bert.py` - DistilBERT batch evaluation helpers
- `src/evaluation/runner.py` - CLI orchestration / `run_evaluation()`

Compatibility:

- Keep `src/evaluate.py` as a CLI-compatible wrapper.

Acceptance:

```bash
python -m src.evaluate
pytest tests/test_evaluate.py -q
pytest tests/test_boundaries.py -q
```

Risk:

- Evaluation may import plotting; inference and app service must not.

---

### Issue #57 - Modular refactor: move data parsing and EDA helpers

Goal:

- Move raw-data and EDA concerns into `src/data/`.

Proposed moves:

- `src/parser.py` -> `src/data/parser.py`
- `src/preprocess.py` -> `src/data/preprocess.py`
- `src/features.py` -> `src/data/features.py`

Compatibility:

- Keep original modules as re-export wrappers.

Acceptance:

- `tests/test_parser.py` passes.
- `tests/test_preprocess.py` passes.
- EDA notebook import paths still work or are updated.

---

### Issue #58 - Modular refactor: migrate tests to new package paths

Goal:

- Update tests to assert the new module paths directly.
- Keep a small compatibility test that old imports still work.

Acceptance:

- Full test suite passes.
- Compatibility wrappers are explicitly tested.
- No old import path remains in app/training/evaluation implementation unless intentionally kept.

---

### Issue #59 - Modular refactor: remove compatibility wrappers

Goal:

- Remove old wrapper modules only after all implementation and tests use new package paths.

Candidates:

- `src/model.py`
- `src/model_bert.py`
- `src/train.py`
- `src/train_bert.py`
- `src/inference.py`
- `src/evaluate.py`
- `src/app_service.py`
- `src/dataset_bert.py`

This issue should happen last and only if the assessment/deployment no longer depends on old entrypoints.

Acceptance:

- New CLI entrypoints are documented.
- README and architecture docs are updated.
- Full suite passes.

Risk:

- Removing wrappers too early will break existing commands and external references. Prefer keeping wrappers unless there is a strong reason to delete them.

## Agent Prompt Template

Use this prompt for small-agent execution:

```text
You are working on ReviewPulse. Do exactly one incremental modular refactor.

Issue:
<paste one issue from docs/issueBreakdown-phase3.md>

Rules:
- Do not change model behavior.
- Do not retrain models.
- Do not delete existing public imports yet.
- If moving code, leave the old module as a compatibility wrapper.
- Keep CLI entrypoints working unless the issue explicitly says otherwise.
- Do not touch unrelated files.
- Run the relevant tests and report exact results.

Acceptance:
- Existing tests for the touched concern pass.
- Import compatibility is preserved.
- The final diff is small enough to review.
```

## Recommended Execution Order

1. #50 package skeleton.
2. #51 app service and samples.
3. #52 model definitions.
4. #53 tokenization/dataset helpers.
5. #54 training modules.
6. #55 inference package.
7. #56 evaluation package.
8. #57 data parsing and EDA helpers.
9. #58 migrate tests to new package paths.
10. #59 remove wrappers only if still worth it.

## Senior-Engineer Guardrail

The goal is not to make the tree look "enterprise". The goal is to make responsibility obvious:

- models define architectures;
- training creates artifacts;
- inference predicts one review;
- evaluation scores batches;
- app code renders UI;
- data code parses and cleans raw data;
- tokenization code turns text into model-ready tensors.

If a move does not make one of those boundaries clearer, do not move it yet.

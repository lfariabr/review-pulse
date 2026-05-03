# DistilBERT Package Plan

This is a planning note for the next incremental DistilBERT refactor. It maps
the current lifecycle and defines a model-only package target. This document is
intentionally docs-only: no Python code, tests, checkpoints, or model behavior
should change in this PR.

## Current DistilBERT Lifecycle

DistilBERT is currently spread across model definition, tokenization, training,
checkpointing, evaluation, inference, and app-facing availability logic.

| Lifecycle step | Current owner | Current responsibility |
|---|---|---|
| Model identity and defaults | `src/models/bert.py` | Defines `PRETRAINED_DISTILBERT_MODEL_NAME`, `DISTILBERT_MODEL_NAME`, and `BERT_DROPOUT`. |
| Model class | `src/models/bert.py` | Defines `DistilBERTSentiment`, the `torch.nn.Module` wrapper around Hugging Face `DistilBertForSequenceClassification`. |
| Encoder freezing | `src/models/bert.py` | Owns `freeze_distilbert_encoder()`, `unfreeze_distilbert_encoder()`, and `unfreeze_last_encoder_layers()`. |
| Forward pass | `src/models/bert.py` | Accepts `input_ids` and `attention_mask`, then returns one binary sentiment logit per review. |
| Tokenizer loading | `src/tokenization/bert.py` | Loads the Hugging Face tokenizer with `AutoTokenizer.from_pretrained()`. |
| Dataset and dataloaders | `src/tokenization/bert.py` | Owns `BertReviewDataset`, `encode_texts()`, `make_bert_dataloaders()`, and `make_bert_test_loader()`. |
| Training loop | `src/training/bert.py` | Owns per-epoch train/eval helpers, optimizer setup, two-stage training, checkpoint-on-best-validation-F1, and CLI entrypoint. |
| Checkpoint save/load | `src/checkpoint_bert.py` | Serializes tokenizer files, saves compact FP16 checkpoint payloads, validates checkpoint load keys, and loads `(model, tokenizer, checkpoint, device)` bundles. |
| Evaluation | `src/evaluation/bert.py` | Loads the held-out test split, builds the test loader, collects batch predictions, computes metrics, writes confusion matrix and error analysis outputs. |
| Inference loading | `src/inference/loaders.py` | Caches and returns the deployed DistilBERT bundle via `load_distilbert_model()`. |
| Single-text prediction | `src/inference/predictors.py` and `src/inference/api.py` | Cleans text, tokenizes one review, runs the model, applies sigmoid plus `PRED_THRESHOLD`, and returns the prediction dict. |
| App availability and use | `src/app/service.py` and `app.py` | Caches app loading with Streamlit, checks DistilBERT availability, shows fallback warning, exposes DistilBERT in the model selector, and calls `predict_sentiment()`. |

The app uses the deployed checkpoint at `outputs/distilbert.pt`. That checkpoint
is compact rather than fully self-contained: it stores the classification head
and fine-tuned encoder layers, while frozen base encoder weights are supplied by
Hugging Face `distilbert-base-uncased` at load time or from the local HF cache.

## Target Model-Only Package

Recommended target:

```text
src/models/bert/
  __init__.py
  config.py
  model.py
  freezing.py
```

`src/models/bert/` should own only the model definition and model-local helpers.
It should not own tokenization, dataloaders, training loops, checkpoint
serialization, evaluation reports, inference caches, app availability, or CLI
orchestration.

Proposed file responsibilities:

| Target file | Belongs there |
|---|---|
| `src/models/bert/__init__.py` | Public re-export surface for `DistilBERTSentiment`, `PretrainedDistilBERTSentiment`, `BERT_DROPOUT`, `DISTILBERT_MODEL_NAME`, and `PRETRAINED_DISTILBERT_MODEL_NAME`. This keeps `from src.models.bert import ...` working. |
| `src/models/bert/config.py` | DistilBERT model constants only: pretrained model name aliases and model dropout default. |
| `src/models/bert/model.py` | Optional `transformers` import, `DistilBERTSentiment`, classifier/pre-classifier properties, constructor, and forward pass. |
| `src/models/bert/freezing.py` | Useful only if it removes real coupling. Candidate helpers: freeze all encoder params, unfreeze all encoder params, unfreeze last N transformer layers, and report trainable encoder layer indexes. |

Keep `freezing.py` only if the implementation remains clearer than methods plus
a small checkpoint helper. If creating it forces awkward object mutation or
more indirection, keep freeze/unfreeze methods in `model.py` for now and move
only trainable-layer inspection later.

## What Stays in Existing Packages

### `src/training/bert.py`

Keep training orchestration here:

- training hyperparameters and CLI defaults;
- `train_one_epoch_bert()` and `evaluate_epoch_bert()`;
- optimizer helpers and parameter-group construction;
- staged training policy: head stage first, then partial or full encoder fine-tuning;
- validation F1 checkpoint selection;
- calls into tokenization helpers for tokenizer and dataloaders;
- calls into checkpoint helpers for tokenizer serialization and checkpoint saving;
- `main()` for `python -m src.training.bert`.

This module may continue to re-export compatibility helpers in the short term,
but the desired direction is for evaluation and inference to import from the
owning modules directly.

### `src/tokenization/bert.py`

Keep tokenization and dataloader construction here:

- `AutoTokenizer` optional dependency handling;
- tokenizer loading and `local_files_only` behavior;
- `BertReviewDataset`;
- `encode_texts()`;
- train/validation/test dataloader factories;
- DistilBERT test dataloader creation;
- BERT tokenization defaults such as batch size, seed, and local-files mode.

Do not move model construction, optimizer logic, checkpoint payloads, or app
availability checks into tokenization.

### `src/evaluation/bert.py`

Keep evaluation/reporting behavior here:

- batch prediction collection for transformer models;
- loading the DistilBERT held-out test split;
- creating the DistilBERT test loader for evaluation;
- loading the deploy checkpoint for evaluation;
- computing classification metrics;
- printing classification reports;
- writing confusion matrix and error-analysis files.

The next cleanup should import `make_bert_test_loader` from
`src.tokenization.bert` and `load_pretrained_bert_bundle` from
`src.checkpoint_bert` directly, rather than through `src.training.bert`.

### `src/inference/`

Keep runtime prediction behavior here:

- model artifact caches;
- `load_distilbert_model()` as the deployed bundle loader;
- `DistilBERTPredictor` single-text prediction;
- prediction registry and public `predict_sentiment()` API;
- compatibility-facing package exports;
- no training or evaluation imports.

`src/inference/` should continue to use `src.checkpoint_bert` for checkpoint
loading and `src.tokenization` only where runtime tokenization is needed.

## Proposed PR Breakdown

### PR 1: Docs-only package plan

Goal: land this file as the planning artifact.

Scope:

- add `docs/bert-package-plan.md`;
- no Python changes;
- no tests changed;
- no checkpoint or app behavior changes.

Acceptance:

- the document maps where DistilBERT is defined, trained, evaluated, loaded, and used;
- the document defines what belongs in `src/models/bert/`;
- the document defines what remains in training, tokenization, evaluation, and inference.

### PR 2: Split `src.models.bert` into a package

Goal: create the model-only package without changing public imports.

Scope:

- replace `src/models/bert.py` with `src/models/bert/`;
- add `config.py` for model constants;
- add `model.py` for `DistilBERTSentiment` and the alias;
- add `__init__.py` that re-exports the same public API currently exported by `src.models.bert`;
- add `freezing.py` only if the helper split is clearer than keeping methods in `model.py`.

Non-goals:

- no training behavior changes;
- no checkpoint format changes;
- no app or inference behavior changes;
- no compatibility wrapper removal as part of this PR.

Acceptance:

- `from src.models.bert import DistilBERTSentiment` still works;
- existing model tests still pass;
- checkpoint loading still constructs the same model class with the same defaults.

Suggested verification:

```bash
pytest tests/test_model_bert.py -q
pytest tests/test_train_bert.py -q
```

### PR 3: Align checkpoint imports and freezing helpers

Goal: make checkpointing depend on the model package through explicit public
imports.

Scope:

- update `src/checkpoint_bert.py` imports to use the new package exports;
- if `freezing.py` exists, move trainable encoder layer inspection there and
  call it from checkpoint save logic;
- keep `_save_checkpoint()` payload keys, `save_strategy`, and load validation
  behavior unchanged.

Non-goals:

- no `torch.load()` safety migration;
- no checkpoint schema migration;
- no retraining.

Acceptance:

- checkpoint payload keys are unchanged;
- `head_only`, `partial_encoder`, and `full` save strategies behave the same;
- deploy checkpoint evaluation and inference still load existing
  `outputs/distilbert.pt`.

Suggested verification:

```bash
pytest tests/test_train_bert.py -q
pytest tests/test_evaluate.py -q
pytest tests/test_inference.py -q
```

### PR 4: Tighten evaluation imports

Goal: remove unnecessary evaluation coupling to training orchestration.

Scope:

- update `src/evaluation/bert.py` to import `make_bert_test_loader` directly
  from `src.tokenization.bert`;
- update `src/evaluation/bert.py` to import `load_pretrained_bert_bundle`
  directly from `src.checkpoint_bert`;
- leave result files, metric calculations, labels, and printed reports unchanged.

Non-goals:

- no test split changes;
- no metric or threshold changes;
- no plot or error-analysis output changes.

Acceptance:

- DistilBERT evaluation no longer needs to import `src.training.bert`;
- evaluation outputs remain the same for the same checkpoint and test data;
- module boundary tests continue to protect inference and app layers from
  training/evaluation imports.

Suggested verification:

```bash
pytest tests/test_evaluate.py tests/test_boundaries.py -q
```

## Guardrails

- Preserve the public import path `src.models.bert`.
- Preserve the deployed checkpoint path `outputs/distilbert.pt`.
- Preserve checkpoint payload keys and `save_strategy` semantics.
- Preserve `PRED_THRESHOLD` behavior and prediction dict shape.
- Preserve Streamlit availability fallback behavior for missing checkpoint,
  missing `transformers`, corrupt checkpoint, or unavailable Hugging Face cache.
- Keep compatibility-removal work separate from the model package split unless
  the branch is explicitly scoped to Issue 59.

# RQ1/RQ2 validation record

**Validated:** 10 August 2026
**Verdict:** RQ1 and RQ2 are supported by both the stored evidence and a fresh
end-to-end run. The pipeline reproduces, but DistilBERT does not reproduce its
earlier metrics exactly; that result must be versioned rather than silently merged.

## Assessor summary

| Check | Status | Evidence |
|---|---|---|
| Evaluation implementation | Pass | 44 focused tests passed |
| Stored full-test metrics | Pass | Accuracy and macro-F1 reproduce from all six confusion matrices |
| RQ1 conclusion | Supported | Aspect-conditioned models lead both review-only controls on the mixed-polarity subset |
| RQ2 conclusion | Supported | DistilBERT leads predictive metrics; ATAE-LSTM provides a much smaller aspect-aware alternative |
| Dataset provenance | Pass | Official SemEval filenames, SHA-256 hashes and audit counts verified locally |
| Reference audit | Pass with one correction | Seven in-text citations map to seven references; DistilBERT venue wording needs correction |
| End-to-end rerun | Pass with one anomaly | All four models trained and evaluated; three reproduced exactly, while fresh DistilBERT metrics diverged |

## Validated results

All metric manifests use seed `42`, label order `negative`, `neutral`, `positive`,
and the same dataset hashes. Stored full-test metrics reproduce exactly from their
confusion matrices. The table below reports the fresh end-to-end run.

| Model | Test accuracy | Test macro-F1 | Mixed accuracy | Mixed macro-F1 |
|---|---:|---:|---:|---:|
| TF-IDF | 0.7018 | 0.4605 | 0.4430 | 0.3319 |
| LSTM | 0.6687 | 0.4326 | 0.4167 | 0.3264 |
| ATAE-LSTM | 0.6438 | 0.4799 | 0.4737 | 0.4491 |
| DistilBERT | **0.8366** | **0.7490** | **0.7061** | **0.6956** |

The fresh TF-IDF, LSTM and ATAE-LSTM values exactly match the previous validated
record. DistilBERT improved over that record (`0.8250/0.7199`, mixed
`0.6667/0.6473`) and the frozen report (`0.8259/0.7231`, mixed
`0.6623/0.6427`). This is a reproducibility anomaly, not evidence that the older
values are invalid.

### RQ1 — value of aspect input

**Supported.** ATAE-LSTM and DistilBERT both outperform TF-IDF and the
target-agnostic LSTM on mixed-polarity accuracy and macro-F1. This supports
explicit aspect conditioning when one review contains aspects with opposing labels.

### RQ2 — performance and efficiency trade-off

**Supported.** DistilBERT leads the predictive metrics but uses a 256.11 MB
artifact. ATAE-LSTM uses 2.64 MB and improves target-sensitive behaviour without
improving aggregate accuracy. Timing remains observational because the recorded
models were not evaluated on identical hardware.

## Reproducibility record

Recorded validation environment: Python 3.12.10, PyTorch 2.13.0,
scikit-learn 1.8.0, Transformers 5.14.1, NVIDIA driver 580, CUDA 13.0,
Linux 7.0-generic x86_64.

### Commands executed

```bash
.venv/bin/python -m pytest -q \
  tests/absa/test_evaluation.py \
  tests/absa/test_evaluation_runner.py \
  tests/absa/test_comparison.py \
  tests/absa/test_training_protocol.py \
  tests/absa/test_training_runner.py \
  tests/absa/test_parser.py \
  tests/absa/test_prepare_semeval_restaurants.py

.venv/bin/python scripts/prepare_semeval_restaurants.py --verify
.venv/bin/python -m src.absa.data.audit

# Fresh end-to-end reproduction.
.venv/bin/python -m src.absa.training.runner --device auto
.venv/bin/python -m src.absa.evaluation.runner
```

Results:

- Combined validation: **48 passed in 6.22 seconds** (44 evaluation/training and 4 dataset preparation/parser tests).
- Dataset audit: 3,693 train and 1,134 test aspect examples; zero invalid offsets.
- Training and evaluation completed successfully for TF-IDF, LSTM, ATAE-LSTM
  and DistilBERT; evaluation artifacts were regenerated under
  `outputs/absa/evaluation/`.

### Fresh artifact identity

| Artifact | SHA-256 |
|---|---|
| DistilBERT model | `e1109eb06d0f0bf7dd42bb94dae4d6bbdf49faf07d0eaacc9e4045447bb02d1f` |
| Predictions | `b11c1b44804927df748d4b778457504c81f9d3536d75092758d07e70d18d88bd` |
| Evaluation results | `8c7f77c5b81becd902d7506bdff224939656dc59fa01514d929e62a058260b72` |

The fresh DistilBERT run used seed `42`, CUDA, two epochs, batch size `8`,
learning rate `2e-5`, and `distilbert-base-uncased`. It selected epoch 2 with
development macro-F1 `0.6904`. The configuration records the model name but not
an immutable Hugging Face revision.

### Dataset identity

| Split | Local file | Bytes | SHA-256 |
|---|---|---:|---|
| Train | `restaurants_train.xml` | 1,235,614 | `223601da1bded6caa4ef9cf91a7007578141ca6d8ed50d5a5c217565f89d2fc5` |
| Test | `restaurants_test.xml` | 359,021 | `f21509cfa37e16534cd5b2da043be487355b64ef48fe8d6aaacaeca6b49cc0fb` |

The files are installed under `data/semeval2014/restaurants/` and match the
hashes recorded by the verified experiment.

### Full reproduction commands

```bash
# Regenerate the four canonical model artifacts.
.venv/bin/python -m src.absa.training.runner --device auto

# Regenerate predictions, metrics, tables and error analysis.
.venv/bin/python -m src.absa.evaluation.runner --device auto
```

Do not use `--allow-unverified-artifacts` for report evidence.

## Report and reference audit

- Frozen report inspected at `lfariabr/masters-swe-ai@5b5d671`.
- Seven bibliography entries are cited in the report body.
- The six ACL Anthology entries match their authors, titles, years, venues and
  page ranges.
- The DistilBERT arXiv record matches its authors, title and year, but does not
  confirm the report's “5th Workshop” venue. Cite it as the arXiv preprint unless
  a primary workshop source is supplied.

## Remaining observations and actions

1. **Version the result sets.** Keep the frozen report, previous validated record
   and fresh run separate, each with its artifact commit, evaluation commit and
   prediction hash.
2. **Resolve the DistilBERT divergence.** The full pipeline and SemEval XML
   reproduce end to end, but exact transformer metrics do not. Causes are yet to be confirmed.
3. **Preserve the scope of the claim.** Current evidence uses one seed and a
   228-instance mixed-polarity subset. It supports the observed run, not a
   multi-seed uncertainty claim or a controlled cross-device speed comparison.

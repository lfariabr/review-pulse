# RQ1/RQ2 validation record

Date: 2026-08-10

## Commands

```bash
.venv/bin/python -m pytest -q tests/absa/test_evaluation.py \
  tests/absa/test_evaluation_runner.py tests/absa/test_comparison.py \
  tests/absa/test_training_protocol.py tests/absa/test_training_runner.py

# Recomputed accuracy and macro-F1 from every stored test confusion matrix.
.venv/bin/python -c "# sklearn metric cross-check over outputs/absa/*_metrics.json"

# Inspected the frozen report and its references.
git show 5b5d671:2026-T2/DLE/assignments/Assessment3/report/DLE602_A3_Report_v3.md
```
 - Environment: Python 3.12.10, PyTorch 2.13.0, scikit-learn 1.8.0 and Transformers 5.14.1, NVidia driver 580, CUDA version 13.0 on Linux 7.0-generic x86_64
 
## Results

- Tests: **44 passed** in 7.56 s.
- All six stored confusion matrices reproduce their recorded accuracy and macro-F1.
- All metric manifests share commit `cef08fa`, seed 42, label order, and train/test hashes.
- Current canonical results:

| Model | Test acc. | Test macro-F1 | Mixed acc. | Mixed macro-F1 |
|---|---:|---:|---:|---:|
| TF-IDF | 0.7018 | 0.4605 | 0.4430 | 0.3319 |
| LSTM | 0.6687 | 0.4326 | 0.4167 | 0.3264 |
| ATAE-LSTM | 0.6438 | 0.4799 | 0.4737 | 0.4491 |
| DistilBERT | 0.8250 | 0.7199 | 0.6667 | 0.6473 |

- References: **7/7 match** the cited authors, titles, years, where applicable; **7/7 are cited in the report body**. No pages/venues were verified.

## Observations

- **RQ1 supported:** both aspect-conditioned models beat both review-only controls on mixed-polarity accuracy and macro-F1. This supports aspect input on reviews containing opposing aspect labels.
- **RQ2 supported:** DistilBERT leads predictive metrics but uses a 256.11 MB artifact; ATAE-LSTM is 2.64 MB and improves mixed-polarity behaviour without improving aggregate accuracy.
- **Value mismatch:** frozen report commit `5b5d671` contains older DistilBERT values (`0.8259/0.7231`, mixed `0.6623/0.6427`) than the current verified record above.

- **Reference correction:** specify DistilBERT source.

- **Reproduction limitation:** Predictions and mixed-subset metrics were not regenerated end to end. Validation covers stored metrics, confusion matrices, provenance, tests, and report consistency. 


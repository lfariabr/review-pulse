# ReviewPulse v3.0.0 release verification (#89)

> **Final release note — 15 August 2026.** The measurements below preserve the pre-release
> verification trail. The release is now frozen at `c2ee52ab4c4415eb2ddc4223500040147b2a92b9`
> (`v3.0.0`). The final archives are 54,048,531 bytes (lightweight) and 301,254,100 bytes
> (all-models); their authoritative SHA-256 digests are published in the GitHub release and in
> the reproducibility report. Any earlier candidate values or open gates in this historical record
> are superseded, not current blockers.

Evidence recorded while preparing the release branch. Every measurement below was
produced on the branch `release/v3.0.0`; the final tag must be created only after
the A3 report and contribution evidence are complete, so the source commit and
archive digest in `docs/submission-checklist.md` stay blank until then.

Environment: macOS 26.5, Apple Silicon.

## 1. Git LFS

Six artifacts are tracked through LFS by `.gitattributes`:

| Artifact | Size |
|---|---:|
| `outputs/absa/distilbert/model.safetensors` | 268 MB |
| `outputs/absa/atae_lstm.pt` | 2.8 MB |
| `outputs/absa/target_lstm.pt` | 2.4 MB |
| `outputs/absa/target_gru.pt` | 2.1 MB |
| `outputs/absa/text_cnn.pt` | 1.9 MB |
| `outputs/absa/tfidf_baseline.joblib` | 812 KB |

All six materialise after `git lfs pull` in a fresh clone; none remained an
unresolved pointer. The package builder independently rejects pointer files, so a
clone without `git lfs pull` fails the build rather than shipping stubs.

## 2. Clean-room installation

A fresh clone of `release/v3.0.0` with a new virtual environment installed from
`requirements.txt -c constraints-a3.txt` without manual intervention. Resolved
versions match the recorded baseline:

| Package | Resolved |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.13.0 |
| scikit-learn | 1.8.0 |
| Transformers | 5.14.1 |
| Streamlit | 1.59.2 |
| pandas | 3.0.3 |

## 3. Test suite: the clean-room count differs, by design

| Environment | Passed | Skipped |
|---|---:|---:|
| Development machine | 363 | 3 |
| Clean room | 357 | 9 |

The six-test gap is **expected and must not be treated as a regression**. The
sample-provenance tests in `tests/absa/test_samples.py` check each demo sample
against `outputs/absa/evaluation/predictions.csv`, which is intentionally not
tracked: it carries the review text and gold polarity of 1,120 annotated
instances and publishing it would redistribute a substantial part of the licensed
corpus.

Those tests therefore **skip silently in any clean clone**, and a green suite
there is not evidence that the samples still match the dataset. The check must be
run where the frozen evaluation outputs exist. Confirmed on the development
machine for this release: all six executed and passed rather than skipping.

The three development-machine skips are end-to-end accuracy, evaluation and
training checks against the legacy Amazon `.review` corpus, which is also not
redistributed. Five parser aggregation checks use a synthetic four-domain
fixture and therefore run in every environment.

## 4. Offline behaviour

`scripts/smoke_absa.py` passes in the clean room with **no SemEval data present**:
all four core models clean-load from the LFS artifacts and return one prediction
per aspect. The application is therefore usable by a reader who never obtains the
dataset.

### Inference runs on CPU by construction, not by accident

Every neural predictor loaded wholly on CPU in the clean room. Parameter devices
were inspected directly after loading:

| Model | Parameter devices |
|---|---|
| TF-IDF review-only | not applicable, scikit-learn |
| LSTM review-only | `cpu` |
| GRU review-only | `cpu` |
| Text CNN review-only | `cpu` |
| ATAE-LSTM | `cpu` |
| DistilBERT sentence-pair | `cpu` |

This held **while MPS was available on the host** (`torch.backends.mps.is_available()`
returned `True`, CUDA `False`), so the result is not an artefact of a
CPU-only machine. Two distinct mechanisms produce it. The four `.pt` adapters in
`src/absa/inference/predictors.py` pin `map_location="cpu"` when loading a
checkpoint. The v3 DistilBERT does not use `map_location` at all: it loads
through `from_pretrained(..., local_files_only=True)` and stays on CPU because
the resulting model is never moved to an accelerator. Both paths are therefore
device-independent, but not for the same reason.

This evidences the A2 risk-register contingency for artifact loading failure,
"fall back to CPU inference", and the checklist item requiring CPU-only import
and application startup. A marker on any machine, with or without an accelerator,
runs the same path. Training is unaffected: `src/absa/training/distilbert.py`
still selects CUDA, then MPS, then CPU, and the recorded DistilBERT run used MPS.

Commands that genuinely require the corpus — `src.absa.data.audit` and
`src.absa.evaluation.runner` — previously exited with a bare `FileNotFoundError`
traceback naming an absolute path. Both are documented first-run commands, so a
reader following the README reached a raw crash before learning the data must be
acquired separately. `parse_aspect_examples` now raises a `FileNotFoundError`
carrying the official source URL, the `prepare_semeval_restaurants.py` invocation
and a pointer to `semeval-restaurants.md`.

## 5. Package size

Built with `scripts/build_a3_package.py`. Two consecutive builds of the same mode
produced an identical archive SHA-256, confirming the deterministic contract.

| Mode | Size | Entries | Contents |
|---|---:|---:|---|
| `none` | 2.5 MB | 176 | Source, tests and documentation only |
| `lightweight` | 52 MB | 191 | Adds legacy artifacts and the five small v3 models |
| `all` | 288 MB | 197 | Adds the 268 MB DistilBERT directory |

**288 MB is the decision point.** It exceeds the upload limit of many learning
management systems, and the A2 risk register already recorded the contingency:
ship lightweight artifacts and document reproducible DistilBERT retrieval. The
LMS limit must be confirmed before choosing `all`; if `lightweight` is submitted,
the report must state that the DistilBERT path shows the controlled
missing-artifact state until the checkpoint is installed separately.

## 6. Content scan

No tracked file matches SemEval XML, `.review` data, virtual environments,
byte-code caches, editor state, `.env` files, private keys or common credential
patterns. The generated `sha256.json` data manifest is untracked as intended.

The built archive contains no `.xml`, no `predictions.csv`, no `results.json`, no
`error_analysis.json`, no `__pycache__` and nothing under `data/semeval2014/`
beyond `.gitkeep`.

Structural filtering cannot detect a credential embedded inside an otherwise
approved source file, so the manual secret review in
`docs/submission-checklist.md` remains required before sign-off.

## 7. Academic report linkage

The report v3 Markdown source is committed in
`lfariabr/masters-swe-ai@5b5d671`; the current PDF was last regenerated at
`0bec946`, before the final test-count synchronisation. The submission checklist
therefore retains the contribution and final-PDF gates. Regenerate the PDF and
record the resulting academic commit before building the submission archive.

> **Superseded, 15 August 2026.** This section records the state at the time of
> verification and is kept as written. The report has since moved to
> `report/DLE602_A3_Report_v4.md`, frozen at `lfariabr/masters-swe-ai@6eaad14`,
> with Appendices A-H complete. Only the final PDF and DOCX export remains.

## 8. Historical pre-release gates (superseded)

- Final A3 report PDF, and its commit recorded in the checklist.
- Contribution evidence from all group members.
- Confirmed LMS upload limit. The 51.53 MiB lightweight candidate is the planned
  mode for the anticipated 100 MB ceiling; the exact limit still needs confirmation.

> **Historical status, 15 August 2026.** Contribution evidence is complete: Juan's Streamlit
> QA landed in PR #120 and PR #121, and Victor's reproduction and shipped-artifact
> verification in PR #123 at `8787a73`, closing all seven Appendix F checks. The
> final PDF and the LMS limit are the two items still genuinely open. A 15 August
> rebuild measured the lightweight archive at 54,042,836 bytes, confirming the
> 51.53 MiB figure above and its fit under a 100 MB ceiling; the complete archive
> at 301,248,404 bytes would need a shared link under that ceiling.
- Final archive built from the post-merge commit, with its SHA-256 recorded.
- `v3.0.0` tag, subsequently created at `c2ee52a` after the items above were resolved.

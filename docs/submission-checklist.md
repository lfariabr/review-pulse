# DLE602 A3 Submission Checklist — ReviewPulse v3.0.0

Use this checklist against one frozen source commit. Do not create the final tag or upload the ZIP until every required item is evidenced.

**What a tick means here.** A ticked item was verified and recorded on the pre-release
baseline, `main` at merge commit `6588d95` (PR #118), with the evidence held in
`dle602-a3/release-verification.md`. Items that are properties of an artefact which does
not exist yet, namely the final PDF, the final archive and the `v3.0.0` tag, stay blank
even where a preflight measurement exists, because they must be re-established against the
frozen commit. Group-owned acceptance items stay blank until the named contributor
supplies evidence.

## Status — 8 August 2026

Three decisions taken in the academic repository change what several items below mean. They are
recorded here so this file and the A3 report agree.

- **Juan's QA findings are closed as accepted risk.** UI-06/07/08/10/12 will not be triaged before
  submission. They ship as known, documented findings and never as release blockers. The report
  states this in the RQ3 row of Table 1 and in the contribution log.
- **Anonymous public access is no longer a blocking gate.** The sentence holding it open was
  removed from Appendix E. One incognito check before upload is still worth doing, but nothing in
  the report depends on it.
- **Victor's independent reproduction is the only outstanding evidence in the report.** Appendix F
  is the sole remaining "Pending", and its rows are removed, never published empty, if no
  evidence arrives. **Closed on 15 August:** all seven Appendix F checks now record observed
  command output and a Pass, and Table F2 reports the shipped-artifact results beside the
  separately versioned CUDA retrain. No group-owned evidence is outstanding.

Unchanged: every gate below that measures an artefact still requires that artefact. The final
archive digests and the `v3.0.0` tag are not evidenced yet — the remote currently carries only
`v3.0.0-rc.1`.

## Dry run — 15 August 2026

Both archives were built from `main` at `8787a73` with a clean tree and no report bundled, to prove
the pipeline before the final export lands. The final build adds the report PDF and therefore
changes both digests; these values are evidence that the builder works, never the shipped numbers.

| Mode | Bytes | Size | SHA-256 |
|---|---:|---:|---|
| `lightweight` | 54,042,836 | 51.5 MB | `08ee82d7962352aca82f54ad54b82e6e17ac178d590b901d2b58a70f4fcc9181` |
| `all` | 301,248,404 | 287.3 MB | `60f99ae6a2ee28ae773ec3ffe8246a0c6ef0f57a30c55ccbd4c6c180a08f2e30` |

The measured sizes confirm the earlier `release/v3.0.0` preflight figures of 52 MB and 288 MB. The
July archives still sitting in `dist/` are superseded and must not be uploaded: the lightweight one
is 11 MB because it was built before the LFS artifacts were materialised, so it carries pointer
files where the models should be.

Content scan of the extracted lightweight archive passed every exclusion gate: no `.git`, `.venv`,
`__pycache__`, `.pytest_cache`, `.env`, `.DS_Store`, no SemEval XML, no `predictions.csv` and no
unresolved LFS pointer. `data/` contains only `.gitkeep`. Full suite on the source tree: **363
passed / 3 skipped**, matching the recorded baseline.

## Release identity

- [x] Exact post-merge #89 source commit recorded: `7adb3ca401913e2486038ddf592292baea0e9511`
- [x] Academic report commit recorded: `6eaad14` (source `report/DLE602_A3_Report_v4.md`; export tracked separately)
- [x] Submission ZIP SHA-256 and size recorded. This file ships inside both archives, so it cannot state their digests without invalidating them. The measured values live in the academic checklist, the `v3.0.0` GitHub release and the checksum note accompanying the upload; `PACKAGE_MANIFEST.json` covers every entry one level down
- [x] LMS upload limit confirmed: the 300 MB-class archive uploads directly, so no shared link is required
- [x] Artifact mode chosen against that limit: **both** modes are uploaded. Measured on `release/v3.0.0`: `none` 2.5 MB, `lightweight` 52 MB, `all` 288 MB
- [ ] `v3.0.0` tag points to the verified source commit
- [ ] GitHub release notes and submitted package describe the same contents

The implementation baseline before #89 is merge commit `0f02be3` (PR #100). The final archive must be built only after #89 is merged and must identify that exact post-merge commit.

## Report and group record

- [ ] Final report is 1,350–1,650 words under its declared counting rule. The v4 source declares 1,550 at the end of Section 6; confirm the figure survives the export
- [x] Canonical four-model results remain separate from exploratory GRU/TextCNN results. Table 2 is canonical; GRU and TextCNN appear only in Appendix A
- [x] Tables, figures and token-evidence examples trace to frozen outputs. Canonical evidence cites commit `bf36c3b3`; the supplemental track cites artifact commit `cef08fa`, evaluation commit `941148c` and its prediction SHA-256
- [x] Attention and attribution are described as indicative, not causal. Stated in the RQ3 answer and repeated in the Table 4 caption
- [ ] Contribution record and dated hand-offs are confirmed by all members
- [x] Academic Integrity Declaration and Statement of Acknowledgement are complete. Report sections 12 and 13, including the AI-tool acknowledgement
- [x] Report delivery decided: uploaded as its own file, deliberately not bundled inside either archive, so the archive digests depend only on the source tree and the model artifacts

Group members:

- Luis Faria — A00187785
- Victor Dorantes — A00179705
- Juan Martinez — A00167145

## Source and licensing

- [x] Package includes the required Python source, tests, README and DLE602 documentation. Built from the `build_a3_package.py` allowlist; 176 entries in `none` mode before any artifact is added
- [x] No `.env`, credentials, tokens, private keys, editor state, caches or temporary files. Content scan in `dle602-a3/release-verification.md` section 6; the manual secret review below is still required
- [x] No `.git/`, `.venv/`, `__pycache__/`, `.pytest_cache/` or Hugging Face cache
- [x] No restricted SemEval XML or derived row-level dataset is redistributed. The archive carries no `.xml`, no `predictions.csv`, no `results.json` and nothing under `data/semeval2014/` beyond `.gitkeep`
- [x] SemEval acquisition, placement and checksum instructions are included. `dle602-a3/semeval-restaurants.md`, quick-start Path C, and the `FileNotFoundError` raised by `parse_aspect_examples`
- [x] Third-party dependencies and cited model/data sources are documented. `requirements.txt`, `constraints-a3.txt` and the report reference list
- [ ] Git status is clean before the package is built

## Environment and installation

Run in a new environment using the reviewed A3 constraints:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -c constraints-a3.txt
```

- [x] Python and platform versions recorded. Python 3.12.10 on macOS 26.5, Apple Silicon
- [x] Installation succeeds without undocumented manual changes. Fresh clone and new virtual environment, `dle602-a3/release-verification.md` section 2
- [x] Resolved critical dependency versions match `constraints-a3.txt`. torch 2.13.0, scikit-learn 1.8.0, transformers 5.14.1, streamlit 1.59.2, pandas 3.0.3
- [x] CPU-only import and application startup succeed. Verified on `release/v3.0.0`: every neural predictor loaded wholly on CPU in the clean room **even though MPS was available on that host**. The four `.pt` adapters pin `map_location="cpu"`; the v3 DistilBERT stays on CPU because the local model is never moved to an accelerator

## Automated verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/smoke_absa.py
.venv/bin/python scripts/export_absa_evidence.py
```

- [x] Full suite passes; counts and expected skips are recorded. Development machine 363 passed / 3 skipped, clean clone 357 / 9, extracted lightweight package 355 / 11; the deltas are explained in `dle602-a3/release-verification.md` section 3
- [x] Sample-provenance tests **executed, not skipped**: run where `outputs/absa/evaluation/predictions.csv` exists and confirm the six `test_sample_matches_the_official_test_split` cases are not in the skip list. They skip silently in a clean clone, so a green suite there does not evidence this check (see `dle602-a3/release-verification.md`)
- [x] Legacy ISY503 regression path remains functional. The v2 suite under `tests/` passes alongside the ABSA suite in the same run
- [x] All available v3 artifacts clean-load. `scripts/smoke_absa.py` covers the four canonical models in the clean room with no SemEval data present; `scripts/smoke_target_gru.py` and `scripts/smoke_text_cnn.py` cover the two exploratory models
- [x] `food` and `service` smoke predictions return one result per aspect
- [x] TF-IDF/LSTM/GRU/TextCNN evidence state is explicitly unsupported. Covered at registry level for all four by the parametrised `exposes_token_evidence` assertion over `REVIEW_ONLY_MODELS` in `tests/test_absa_results.py`. Per-predictor payload assertions exist for TF-IDF, GRU and TextCNN; the target-agnostic LSTM has no dedicated payload test and relies on the registry contract alone
- [x] ATAE-LSTM attention aligns to visible review offsets. `tests/absa/test_attention.py` covers visible-token-only alignment, short sequences and preserved case, punctuation and offsets
- [x] DistilBERT attribution aligns to visible review offsets. `tests/absa/test_attribution.py` asserts subword scores aggregate to exact visible tokens and rejects inconsistent inputs

## Data audit and evaluation evidence

With legitimately acquired Restaurants XML files:

```bash
.venv/bin/python -m src.absa.data.audit
.venv/bin/python -m src.absa.evaluation.runner --device cpu
```

- [x] Audit reproduces the documented label and offset counts. Report Table 1; all annotated offsets valid and 105 `conflict` annotations counted before exclusion
- [x] Grouped split overlap assertions pass. `tests/absa/test_splits.py` covers deterministic grouping and a loud overlap failure
- [x] Official retained test count is 1,120. `outputs/absa/evaluation/results.json` records `official_test_examples: 1120`
- [x] Mixed-polarity subset is 228 instances across 80 sentences. Same file, `mixed_polarity_examples: 228` and `mixed_polarity_sentences: 80`
- [x] Canonical evaluation output and prediction digest are preserved. Frozen at commit `bf36c3b3` with prediction SHA-256 `b80dc72c…`
- [x] Evaluation devices and cross-device timing limitations are documented honestly. DistilBERT was evaluated on MPS and the other models on CPU, so no full CPU evaluation was performed; the report states timing is observational and refuses to read architectural speedups from it

The supplemental six-model command is:

```bash
.venv/bin/python -m src.absa.evaluation.runner \
  --models tfidf target_lstm target_gru text_cnn atae_lstm distilbert \
  --device cpu
```

- [x] Supplemental output remains separate from `outputs/absa/evaluation/`. It writes to `outputs/absa/evaluation-six-model/`; both directories exist and neither overwrites the other
- [x] GRU and TextCNN remain labelled exploratory. Confined to report Appendix A and `dle602-a3/six-model-results.md`

## Streamlit acceptance

```bash
.venv/bin/streamlit run app.py
```

Juan Martinez delivered an initial 12-case authenticated Streamlit validation in PR #120,
recorded in `dle602-a3/validation-juan.md`. It confirmed several workflows, recorded three
acceptance failures requiring technical triage, and left two stale-state checks blocked because
the exact leaked fields and transitions were not captured. Those findings are now **accepted as documented risk** for this
submission and are not being triaged; see the status block above. The boxes below record what a
marker actually sees, so they stay blank until each behaviour is confirmed against the deployed
application. Automated tests cover parts of the underlying behaviour but are not a substitute for
that confirmation.

- [ ] Landing page clearly separates ISY503 v2.3.0 and DLE602 v3.0.0
- [ ] Intro page does not duplicate the sidebar logo
- [ ] Sidebar logo, menu order and favicon render correctly
- [ ] Sample generator fills a mixed-polarity review and aspects
- [ ] Manual comma-separated aspects preserve input order
- [ ] Each of the six v3 models can be selected when its artifact exists
- [ ] Supported token evidence renders safely with its limitation
- [ ] Missing artifacts and invalid input show controlled errors without silent fallback
- [ ] No stack trace or debug output appears in the user workflow
- [ ] Public deployment opens without authentication in an incognito/unauthenticated session. No longer a blocking gate, since no report claim depends on it, but worth one check before upload

Capture at least:

- [ ] v3 input/result view
- [ ] ATAE-LSTM heatmap
- [ ] DistilBERT attribution view
- [ ] one controlled missing-artifact or validation message

## Artifact strategy

Record every included artifact. The v3 models answer the research questions; the four
legacy v2 files are shipped by every artifact-bearing mode because the preserved ISY503
page needs them. `Bytes` and `SHA-256` are properties of the final package and stay blank
until it is built; `Runtime/network dependency` is a stable property of the artifact and is
recorded now.

| Artifact | Track | Included? | Bytes | SHA-256 | Runtime/network dependency |
|---|---|:---:|---:|---|---|
| TF-IDF | v3 | [ ] | | | None; `joblib` load from disk |
| Target LSTM | v3 | [ ] | | | None; `torch.load` from disk |
| Target GRU | v3 | [ ] | | | None; `torch.load` from disk |
| TextCNN | v3 | [ ] | | | None; `torch.load` from disk |
| ATAE-LSTM | v3 | [ ] | | | None; `torch.load` from disk |
| DistilBERT | v3 | [ ] | | | None; `from_pretrained(local_files_only=True)` from the bundled directory |
| `outputs/baseline.joblib` | legacy v2 | [ ] | | | None; `joblib` load from disk |
| `outputs/bilstm.pt` | legacy v2 | [ ] | | | None; `torch.load` from disk |
| `outputs/distilbert.pt` | legacy v2 | [ ] | | | **Hugging Face**: fetches the frozen `distilbert-base-uncased` base encoder. Excluded from the A3 offline guarantee |
| `outputs/vocab.json` | legacy v2 | [ ] | | | None; local vocabulary file |

- [x] Included v3 artifacts load fully offline. All six read from local files and none contacts a remote host at inference time
- [x] The preserved v2 DistilBERT external dependency is documented and excluded from the A3 offline guarantee. Legacy `outputs/distilbert.pt` stores only the classification head and fine-tuned layers, so its frozen base encoder is fetched from `distilbert-base-uncased`. This is stated in the README and quick-start Path A, and the offline guarantee is scoped to the v3 models it applies to
- [x] Artifact-bearing modes include the four legacy v2 files required by the preserved ISY503 page. `LEGACY_ARTIFACTS` ships `baseline.joblib`, `bilstm.pt`, `distilbert.pt` and `vocab.json`
- [x] The lightweight CPU strategy includes at least the verified small-model path. All five small v3 artifacts are in `lightweight` mode and each loaded wholly on CPU in the clean room
- [x] DistilBERT packaging decision is consistent with the confirmed LMS limit. The limit accepts the complete archive, so DistilBERT ships in `all` and both modes are uploaded; the lightweight archive is retained as a faster download and not as a size workaround
- [x] No package claims offline support if a Hugging Face download is still required. The README and quick-start Path A both carry the legacy-DistilBERT caveat

## Package inspection

Every item here is a property of one built archive, so all stay blank until the final
archive exists. Preflight builds from `release/v3.0.0` already passed the equivalent
checks, including an identical SHA-256 across two consecutive builds of the same mode; see
`dle602-a3/release-verification.md` sections 5 and 6. Those measurements inform the
artifact-mode decision but do not discharge these gates.

- [ ] Archive is built with `scripts/build_a3_package.py` using the selected artifact mode
- [ ] Archive is built from a documented allowlist, not the entire working directory
- [ ] Archive expands into one clearly named root folder
- [ ] README quick-start is visible at the package root
- [ ] No broken symlinks or absolute local paths
- [ ] Largest files and total uncompressed/compressed sizes are reviewed
- [ ] Secret scan returns no findings
- [ ] Cache/temporary-file scan returns no findings
- [ ] Restricted-data scan returns no findings
- [ ] ZIP is extracted into a clean directory and the documented verification path is rerun

## Final sign-off

| Gate | Owner | Status | Evidence |
|---|---|:---:|---|
| Report and references | Group | [ ] | Content complete and frozen at `masters-swe-ai@6eaad14`, source `report/DLE602_A3_Report_v4.md`. The Markdown carries the word count at the end of Section 6, but the Word-produced export keeps dropping it: v6 still returns zero `pdftotext` hits. Export correctness is tracked in the academic repo |
| Contribution record | Group | [x] | Juan's 12-case QA delivered and mapped in Appendix E; Victor's independent reproduction and shipped-artifact verification delivered and recorded across all seven Appendix F rows |
| Clean installation | Luis Faria | [x] | `dle602-a3/release-verification.md` section 2 |
| Tests and CPU smoke | Luis Faria | [x] | `dle602-a3/release-verification.md` sections 3 and 4 |
| Artifact checksums/sizes | | [ ] | Sizes recorded in sections 1 and 5; per-artifact SHA-256 still to be captured in the table above |
| Streamlit acceptance | Juan Martinez | [x] | Authenticated 12-case QA delivered in PR #120, hardened in PR #121. Three acceptance failures and two unreproduced observations are accepted as documented risk, not blockers |
| Package content/security scan | | [ ] | Preflight scan clean in section 6; rerun against the final archive |
| ZIP extraction retest | | [ ] | Extracted lightweight package recorded 355 passed / 11 skipped in preflight; rerun against the final archive |
| Final tag and GitHub release | | [ ] | Blocked by all outstanding gates above |

Final sequence:

1. Freeze the accepted report and source commits.
2. Merge #89.
3. Build and inspect the deterministic ZIP from the exact post-merge commit.
4. Extract and retest the ZIP.
5. Record sizes and SHA-256 digests.
6. Obtain group sign-off.
7. Create and publish `v3.0.0` from the verified release commit.

# DLE602 Assessment 3 - Submission Contents

**ReviewPulse v3.0.0 - Aspect-Based Sentiment Analysis on SemEval-2014 Task 4 Restaurants**

| Member | Student ID |
|---|---|
| Luis Faria | A00187785 |
| Victor Dorantes | A00179705 |
| Juan Martinez | A00167145 |

## What was submitted

| Item | File | Notes |
|---|---|---|
| Report | `DLE602_Faria_L_Assessment_3.pdf` | Marked deliverable |
| Report, editable | `DLE602_Faria_L_Assessment_3.docx` | Supplied for annotation |
| Source and models, lightweight | `ReviewPulse-v3.0.0-lightweight.zip` | Five of six v3 models, approx. 52 MB |
| Source and models, complete | `ReviewPulse-v3.0.0-all-models.zip` | All six v3 models, approx. 288 MB |

The report is uploaded as its own file and is deliberately not bundled inside either archive, so the
code package and the written submission can be opened independently.

Both archives are built from one frozen commit by `scripts/build_a3_package.py` and differ only in
whether the v3 DistilBERT directory is included. That directory is roughly 256 MB and accounts for
the entire size difference. Both archives are uploaded directly; the confirmed upload limit accepts
the complete one, so no external link is involved anywhere in this submission.

Start with the complete archive if you want to exercise all six models, including DistilBERT. The
lightweight archive exists for a faster download and runs five of the six.

## Integrity

Source commit: `c2ee52ab4c4415eb2ddc4223500040147b2a92b9` · Tag: `v3.0.0`

This file travels inside the archives, so it cannot state their SHA-256 digests: no file can carry
a checksum of the container holding it. The two archive digests are published outside the packages,
in the `v3.0.0` GitHub release and in the checksum note accompanying the upload. Verify a download
against those:

```bash
shasum -a 256 ReviewPulse-v3.0.0-lightweight.zip
```

What this file can confirm is everything one level down. Each archive carries
`PACKAGE_MANIFEST.json`, recording the source commit plus the byte size and SHA-256 of **every
entry**, so any individual file can be checked without trusting the archive digest at all. Builds
are deterministic: entry timestamps are fixed to the source commit time and paths are sorted, so
rebuilding from `c2ee52a` in the same mode reproduces identical bytes. Two consecutive builds were
confirmed byte-identical.

The report is uploaded alongside these archives and is not bundled inside them, so the digests
depend only on the source tree and the model artifacts.

## Running the code

Neither archive needs the SemEval corpus, an accelerator or a network connection: the trained
artifacts are shipped and inference runs on CPU.

```bash
unzip ReviewPulse-v3.0.0-lightweight.zip && cd ReviewPulse-v3.0.0
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt -c constraints-a3.txt
python -m pytest -q
streamlit run app.py
```

Expected suite results are environment-dependent and every skip is intentional. The extracted
lightweight archive gives **355 passed / 11 skipped**: six sample-provenance checks need the
non-redistributed `predictions.csv`, two package-builder checks need Git metadata that a ZIP does
not carry, and the rest cover the legacy Amazon corpus that the project cannot license. Report
Appendix H tabulates all three environments.

Selecting **DistilBERT sentence-pair** in the lightweight archive reports the model unavailable and
returns no prediction. A missing artifact is always reported and never silently substituted.
`scripts/smoke_absa.py` exercises that model, so it is deliberately unusable there;
`scripts/smoke_target_gru.py` and `scripts/smoke_text_cnn.py` are the equivalent clean-load checks.

## Live application

<https://review-pulse.streamlit.app/ReviewPulse_v3_0_0>

The deployment is convenience only. Every number in the report comes from the frozen artifacts in
the archives, so the submission is complete and assessable without visiting the link.

## Where to look first

| Question | Location |
|---|---|
| What was built and why | Report Sections 1–6 |
| Results tables | Report Section 5, Tables 2–5 |
| Code walkthrough, configurations, transcripts | Report Appendix G |
| How to run either package | Report Appendix H, `docs/dle602-a3/quickstart.md` |
| Reproduction commands | Report Appendix C |
| Dataset acquisition and checksums | `docs/dle602-a3/semeval-restaurants.md` |
| Independent validation | `docs/dle602-a3/validation-juan.md`, `validation-victor.md` |
| Contribution record | Report Appendix B |

## Data

The SemEval-2014 Task 4 Restaurants XML is licensed and is **not** redistributed. No archive
contains it, nor any row-level export derived from it. `data/semeval2014/` ships empty. Acquisition
instructions, expected filenames and SHA-256 checksums are in
`docs/dle602-a3/semeval-restaurants.md`; the parser raises a controlled `FileNotFoundError` pointing
there when the corpus is absent.

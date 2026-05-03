# Submission Checklist

Final checklist for group code submission. Use this document to verify all required components are present, tested, and documented before submission.

---

## Repository & Code

- [ ] **GitHub Repository Link**
  - URL: `https://github.com/lfariabr/review-pulse`
  - Status: Public, all commits visible
  - Git history: Clean with conventional commit messages

- [ ] **Main Branch Protected & Updated**
  - All PRs merged into main
  - All tests passing on main branch
  - Latest commit: verified with `git log --oneline -1`

---

## Deployment & Access

- [ ] **Streamlit App Deployed**
  - Deployed URL: to be confirmed by Luis
  - Status: Running locally; production deployment to be confirmed by Luis
  - Instructions: See README.md → "Quick Start"
  - Command: `streamlit run app.py` (from repository root)

- [ ] **README Setup Instructions Verified**
  - File: `README.md`
  - Sections confirmed:
    - [ ] Environment setup (Python, venv)
    - [ ] Dependency installation (`pip install -r requirements.txt`)
    - [ ] Data download / setup steps
    - [ ] How to run tests
    - [ ] How to run the app (Streamlit)
    - [ ] How to run inference / predictions
    - [ ] How to train models (if applicable)

---

## Model Artifacts

- [ ] **Model Artifacts Archived**
  - TF-IDF baseline: `outputs/baseline.joblib` (v2.1.0)
  - BiLSTM checkpoint: `outputs/bilstm.pt` (v2.1.0)
  - DistilBERT checkpoint: `outputs/distilbert.pt` (v2.1.0, integrated in app)
  - Vocabulary: `outputs/vocab.json` (JSON format)
  - All artifacts tested and verified to load
  - Open item: checkpoint hosting remains tracked in Issue #28

- [ ] **Model Reproducibility**
  - Training scripts: `src/training/bilstm.py`, `src/training/bert.py`
  - Inference scripts: `src/inference/` package (Closes #55)
  - Evaluation scripts: `src/evaluation/` package (Closes #56)
  - Seed fixed for reproducibility (tests verified)
  - Results can be replicated from training

---

## Issues & Pull Requests

- [ ] **GitHub Issues Tracked**
  - Phase 3 modular refactor issues: #50–#59
  - Core deliverables: #20–#30
  - All issues link to relevant PRs
  - Closed issues marked with PR/commit references

- [ ] **Pull Requests Documented**
  - All PRs tagged with conventional commit scope (e.g., `feat(training)`, `fix(inference)`)
  - All PRs link to issue(s) they close
  - All PRs have meaningful descriptions
  - All PRs passed CI tests before merge

---

## Releases & Tags

- [ ] **GitHub Release / Tag Created**
  - Latest: `v2.1.0` (current stable, includes DistilBERT)
  - Future release: `v2.2.0` post-refactor release
  - Release notes: Include key features, model performance, known limitations
  - Link: `https://github.com/lfariabr/review-pulse/releases`

---

## Presentation & Evidence

- [ ] **Presentation / Demo Video**
  - **Template:** See [docs/assessment-files/presentation-outline.md](assessment-files/presentation-outline.md) (slide-by-slide speaker notes)
  - **Demo test cases:** See [docs/assessment-files/demo-test-cases.md](assessment-files/demo-test-cases.md) (10 acceptance test cases with model outputs)
  - Owner: to be confirmed by Luis
  - Link: to be confirmed by Luis
  - Duration: ~13–14 minutes total (Luis ~6.25 min, Victor ~3.5 min, Samiran ~3.75 min)
  - Covers:
    - [ ] Problem framing: sentiment analysis in e-commerce (Slide 2)
    - [ ] Dataset: 8,000 Amazon reviews, 4 domains, 50/50 balance (Slide 3)
    - [ ] Baseline & BiLSTM: architecture & training (Slides 4–7)
    - [ ] DistilBERT: fine-tuning & performance gains (Slide 8)
    - [ ] Error analysis: 10 acceptance test cases, failure modes (Slide 9)
    - [ ] Live demo: Streamlit app, edge cases, confidence values (Slide 9 demo)
    - [ ] Ethics: label noise, domain bias, calibration risk (Slide 10)
    - [ ] Future work: RoBERTa, Platt scaling, LIME explainability (Slide 11)
  - Submitted to: to be confirmed by Luis

- [ ] **Individual Report**
  - **Template:** See [docs/assessment-files/individual-report-template.md](assessment-files/individual-report-template.md) (detailed template with examples)
  - **Target length:** 250 words ±10% (225–275 words per report)
  - **Slide assignments:**
    - [ ] Luis Faria (A00187785): Slides 1, 4, 5, 6, 7, 12 (Title · Preprocessing · Architecture · Training · Results · Summary)
    - [ ] Victor Meneses (A00179705): Slides 3, 8, 10 (Dataset · Transformers · Ethics & Limitations)
    - [ ] Samiran Shrestha (A00106473): Slides 2, 9, 11 (Problem Statement · Live Demo · Future Work)
  - **Requirements:**
    - [ ] Student name & ID confirmed
    - [ ] Contribution % agreed by team (must total 100%)
    - [ ] Distinct ethical angle per report
    - [ ] APA references included
    - [ ] Speaker notes align with presentation outline
  - Submitted to: to be confirmed by Luis

---

## Academic Integrity

- [ ] **Group Member Details**
  - [ ] Luis Faria — Student ID: A00187785 — 60% (primary technical implementation, app, tests, docs, PR hardening)
  - [ ] Victor Meneses — Student ID: A00179705 — 25% (DistilBERT implementation, dataset analysis, error analysis, ethics)
  - [ ] Samiran Shrestha — Student ID: A00106473 — 15% (problem framing, live demo, future work)
  - **Total:** 100% ✓ (per `docs/assessment-files/individual-report-template.md`)

- [ ] **Academic Integrity Declaration**
  - [ ] All group members have reviewed the submission
  - [ ] Code authored by group members (or properly attributed to open-source libraries)
  - [ ] No code submitted for other assessments in parallel
  - [ ] Torrens University academic integrity policy understood and acknowledged
  - [ ] Final academic integrity confirmation completed by Luis and team
  - [ ] Declaration statement signed by all members if required by institution

- [ ] **Citation & Attribution**
  - All external libraries listed in `requirements.txt`
  - Data sources documented in code comments
  - Academic papers / references cited in docstrings or README
  - No plagiarism: code is original or properly licensed

---

## Backup & Final Steps

- [ ] **Backup Copy Retained**
  - Local backup: to be confirmed by Luis
  - Cloud backup: to be confirmed by Luis
  - Retention period: Until final grade confirmed (minimum 1 month post-submission)
  - Proof: Screenshot or confirmation email

- [ ] **Final Tests Run**
  - Fast suite command: `pytest tests/ -q -m "not slow"`
  - Full suite command: `pytest tests/ -q`
  - Latest verified test output: rerun immediately before submission or cite the final CI run; avoid stale exact counts
  - Linting status: to be confirmed by Luis if used
  - No warnings or errors on startup

- [ ] **Code Review Checklist**
  - [ ] No hardcoded credentials in repository
  - [ ] No unnecessary large files (`.gitignore` is clean)
  - [ ] No debug print statements left in production code
  - [ ] Type hints and docstrings present where relevant
  - [ ] README is complete and accurate

---

## Submission Sign-Off

| Item | Status | Notes |
|------|--------|-------|
| Repository ready | ✓ | Public, accessible, clean history |
| Tests passing | ✓ | Fast suite command documented; use latest CI/local output before final submission |
| README complete | ✓ | Setup, usage, deployment documented |
| Model artifacts | ✓ | BiLSTM, TF-IDF, DistilBERT (v2.1.0) — all trained & packaged |
| Issues & PRs | ✓ | Phase 3 refactor tracked in #50–#59; merged PRs documented through #66 |
| Release created | ✓ | v2.1.0 tagged; v2.2.0 planned as post-refactor release |
| Presentation | 🕐 | Outline & test cases ready; video to be confirmed by Luis |
| Individual reports | 🕐 | Template ready; final reports to be confirmed by Luis |
| Academic integrity | ✓ | Group members & IDs confirmed |
| Backup retained | 🕐 | Location to be confirmed by Luis |

---

## Notes & Known Limitations

- **Model Performance:**
  - BiLSTM: Single-layer architecture, trained on 8,000 reviews
  - DistilBERT: Hugging Face fine-tuned (v2.1.0), achieves 88.2% test accuracy
  - TF-IDF baseline: Logistic regression (v2.1.0), achieves 81.9% F1
  - Held-out test F1: 88.6% (DistilBERT) vs. 80.3% (BiLSTM) vs. 81.9% (baseline)
  - Generalization: Trained on Amazon reviews (4 domains); may not transfer to other review types without fine-tuning

- **Deployment:**
  - Streamlit app: `streamlit run app.py` (runs locally)
  - Production deployment (Streamlit Cloud, AWS, etc.): Not yet configured
  - Model loading: Local checkpoints from `outputs/`
  - Open item: checkpoint hosting remains tracked in Issue #28

- **Future Work:**
  - Remove compatibility wrappers (Issue #59) once all consumers use new paths
  - Additional model architectures (RoBERTa, XLNet)
  - Cross-domain evaluation on different review datasets
  - Active learning pipeline for efficient labeling

---

**Last Updated:** 2026-05-04
**Document Owner:** Luis Faria
**Review Status:** Ready for team sign-off

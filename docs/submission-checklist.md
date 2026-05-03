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
  - Deployed URL: _(to be confirmed)_
  - Status: Running and accessible
  - Instructions: See README.md → "Quick Start"
  - Note: App requires `python src/app.py` or Streamlit cloud deployment

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
  - BiLSTM checkpoint: `outputs/bilstm.pt`
  - DistilBERT checkpoint: `outputs/distilbert.pt` _(or hosted URL)_
  - Vocabulary: `outputs/vocab.pkl`
  - All artifacts tested and verified to load
  - Checksums recorded (if applicable)

- [ ] **Model Reproducibility**
  - Training scripts: `src/training/bilstm.py`, `src/training/bert.py`
  - Inference scripts: `src/inference.py`
  - Evaluation scripts: `src/evaluate.py`
  - Seed fixed for reproducibility
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
  - Release/tag name: `v<X.Y.Z>` (e.g., `v1.0.0`)
  - Release notes: Include key features, model performance, known limitations
  - Artifacts attached (optional): Model checkpoints, test results
  - Link: `https://github.com/lfariabr/review-pulse/releases/tag/v<X.Y.Z>`

---

## Presentation & Evidence

- [ ] **Presentation / Demo Video**
  - Owner: _(group member name)_
  - Link: _(video URL or upload location)_
  - Duration: ~10–15 minutes
  - Covers: Problem, solution, model performance, demo
  - Submitted to: _(Torrens LMS or specified platform)_

- [ ] **Individual Report**
  - Owner: _(student name)_
  - Sections:
    - [ ] Your specific role(s) in the project
    - [ ] Key technical decisions you made
    - [ ] Challenges faced and how you overcame them
    - [ ] What you learned
    - [ ] Code/commit evidence of your work (with links)
  - Length: ~500–1000 words (per assignment brief)
  - Submitted to: _(Torrens LMS or specified platform)_

---

## Academic Integrity

- [ ] **Group Member Details**
  - Member 1: _(name)_ — Student ID: _(ID)_
  - Member 2: _(name)_ — Student ID: _(ID)_
  - Member 3: _(name)_ — Student ID: _(ID)_
  - _(Add/remove rows as needed)_

- [ ] **Academic Integrity Declaration**
  - [ ] All group members have reviewed the submission
  - [ ] Code authored by group members (or properly attributed to open-source libraries)
  - [ ] No code submitted for other assessments in parallel
  - [ ] Torrens University academic integrity policy understood and acknowledged
  - [ ] Declaration statement signed by all members (if required by institution)

- [ ] **Citation & Attribution**
  - All external libraries listed in `requirements.txt`
  - Data sources documented in code comments
  - Academic papers / references cited in docstrings or README
  - No plagiarism: code is original or properly licensed

---

## Backup & Final Steps

- [ ] **Backup Copy Retained**
  - Local backup: _(path or storage location)_
  - Cloud backup: _(e.g., Google Drive, Dropbox link)_
  - Retention period: Until final grade confirmed (minimum 1 month post-submission)
  - Proof: Screenshot or confirmation email

- [ ] **Final Tests Run**
  - All tests pass: `pytest tests/ -q`
  - Linting passed: _(if applicable)_
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
| Tests passing | ✓ | All 203 tests pass |
| README complete | ✓ | Setup, usage, deployment documented |
| Model artifacts | 🔌 | BiLSTM ready; DistilBERT status TBD |
| Issues & PRs | ✓ | Phase 3 modular refactor complete; core issues merged |
| Release created | 🕐 | Ready for post-implementation |
| Presentation | 🕐 | Video owner TBD |
| Individual reports | 🕐 | Owner(s) TBD |
| Academic integrity | 🕐 | Group details & declaration TBD |
| Backup retained | 🕐 | Location TBD |

---

## Notes & Known Limitations

- **Model Performance:**
  - BiLSTM: Baseline model, single-layer architecture
  - DistilBERT: Stretch goal; performance gains vs. BiLSTM to be quantified
  - Test set accuracy: _(to be confirmed from last full evaluation)_
  - Generalization: Trained on Amazon reviews; may not transfer to other domains without fine-tuning

- **Deployment:**
  - Streamlit app runs locally; production deployment (e.g., Streamlit Cloud, AWS) not yet configured
  - Model loading uses local checkpoints; no CDN fallback yet (Issue #28)

- **Future Work:**
  - Remove compatibility wrappers (Issue #59) once all consumers use new paths
  - Additional model architectures (RoBERTa, XLNet)
  - Cross-domain evaluation on different review datasets
  - Active learning pipeline for efficient labeling

---

**Last Updated:** 2026-05-03  
**Document Owner:** _(to be assigned)_  
**Review Status:** Ready for team sign-off

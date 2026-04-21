# Presentation Outline — ReviewPulse
## ISY503 Intelligent Systems · Assessment 3 · Group Presentation
**Duration:** 12–15 minutes + Q&A  
**Format:** Slides + live Streamlit demo

---

## Slide 1 — Title & Team (1 min)
**Speaker:** TBD

- **ReviewPulse** — Multi-domain Amazon Review Sentiment Classifier
- Team members + roles
- One-line pitch: *"We trained two sentiment classifiers on 8,000 Amazon reviews and built a live web app to compare them."*

---

## Slide 2 — Problem Statement (1 min)
**Speaker:** TBD

- Sentiment analysis is a foundational NLP task with real commercial value (product feedback, brand monitoring, recommendation systems)
- Challenge: reviews vary in length, domain, writing style, and use of negation/sarcasm
- Our goal: build, compare, and deploy two approaches — classical ML vs neural — and be honest about which one works better

---

## Slide 3 — Dataset (1.5 min)
**Speaker:** TBD

- **Source:** Blitzer et al. (2007) — Multi-Domain Sentiment Dataset
- 8,000 labelled Amazon reviews across 4 domains: Books, DVDs, Electronics, Kitchen
- Labels from filenames (`positive.review` → 1, `negative.review` → 0)
- Perfectly balanced: 50/50 positive/negative, 1,000 per domain per class
- **Splits:** train=5,404 / val=1,158 / test=1,159 (stratified 70/15/15, seed=42)
- **Label audit finding:** 0 ambiguous rows — no 3-star reviews in the dataset; label quality is high

---

## Slide 4 — Preprocessing Pipeline (1.5 min)
**Speaker:** TBD

```
Raw .review files
    → BeautifulSoup parser (pseudo-XML)
    → Label audit (flag 3-star / rating conflicts)
    → clean_text: lowercase, strip HTML, expand negations (n't → not), remove punctuation
    → remove_outliers: drop reviews < 10 or > 500 words (279 dropped, ~3.5%)
    → stratified train / val / test split
```

Key decision: negation expansion — *"wasn't good"* → *"was not good"* preserves sentiment signal for bag-of-words models.

---

## Slide 5 — Architecture Overview (2 min)
**Speaker:** TBD

### Model 1 — TF-IDF + Logistic Regression (Baseline)
- `TfidfVectorizer(max_features=30k, ngram_range=(1,2), sublinear_tf=True)`
- `LogisticRegression(C=1.0, max_iter=1000)`
- No training in the neural sense — fit in seconds

### Model 2 — BiLSTM + GloVe (Neural)
```
Token indices (batch × 256)
    → Embedding(15,924 × 100d, GloVe init, 97.4% coverage)
    → Dropout(0.5)
    → BiLSTM(hidden=256, layers=2, bidirectional, pack_padded_sequence)
    → concat final forward + backward hidden state (512d)
    → Dropout(0.5)
    → Linear(512 → 1) → raw logit → BCEWithLogitsLoss
```

Key design decision: `pack_padded_sequence` skips padding tokens — the LSTM sees only real words, not zero vectors.

---

## Slide 6 — Training (1 min)
**Speaker:** TBD

- Optimizer: Adam (lr=1e-3)
- Loss: BCEWithLogitsLoss
- Gradient clipping: max_norm=5.0
- 10 epochs, checkpoint saved by best val F1
- Device: Apple MPS (~2 min/epoch → ~20 min total vs ~160 min on CPU)

| Epoch | Val Acc | Val F1 |
|---|---|---|
| 1 | 72.6% | 71.5% |
| 5 | 78.8% | 80.8% |
| 7 | 83.9% | 83.6% |
| **9 ← best** | **84.3%** | **84.0%** |
| 10 | 80.6% | 82.6% ↓ overfitting |

---

## Slide 7 — Results (1.5 min)
**Speaker:** TBD

### Held-out test set

| Model | Accuracy | F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | **82.7%** | **81.9%** |
| BiLSTM + GloVe | 81.0% | 80.3% |

**Honest finding:** The simpler model generalises better on the test set. The BiLSTM beat the baseline on validation (84.0% vs 83.2%) but not on the held-out test. This is a mature ML result — overfitting to validation distribution is a real risk.

**Why this matters for the rubric:** The BiLSTM satisfies the requirement to *"define the network architecture and model class."* The comparison shows we understand when a neural model is and isn't the right tool.

---

## Slide 8 — Error Analysis (1 min)
**Speaker:** TBD

- 220 misclassified examples on the test set
- Shared failure modes across both models:
  - **Negation:** *"not bad at all"* → predicted Negative (both models)
  - **Sarcasm:** *"Oh great, stopped working after a week"* → near 50% confidence
  - **Out-of-distribution:** logistics/delivery text predicted Negative with no clear basis
- BiLSTM tends toward **overconfidence on short inputs** (e.g. "It is okay" → 88% negative)
- TF-IDF is better **calibrated** near the decision boundary

---

## Slide 9 — Live Demo (2 min)
**Speaker:** TBD

Run: `streamlit run app.py`

Demo script (in order):
1. Clear positive: *"This blender is incredible…"* → Positive 97.9% (BiLSTM)
2. Clear negative: *"Broke after two days…"* → Negative 99.6%
3. Negation trap: *"This is not bad at all"* → both predict Negative ← discuss
4. Sarcasm: *"Oh great, another product that stopped working"* → low confidence ← discuss
5. 💡 Generate button — random sample

---

## Slide 10 — Ethics & Limitations (1.5 min)
**Speaker:** TBD

| Concern | Detail |
|---|---|
| Label quality | Labels from filenames, not human raters — but 0 ambiguous rows found in audit |
| Domain bias | 4 categories only — generalisation to other product types is untested |
| Negation & sarcasm | Systematic failure mode in both models |
| Confidence calibration | BiLSTM logits are not calibrated — 99% confidence ≠ 99% accuracy |
| Dataset provenance | Blitzer et al. 2007 — reviews are ~20 years old; language patterns may have shifted |
| Deployment risk | App should not be used for consequential decisions without human review |

---

## Slide 11 — Future Work (30 sec)
**Speaker:** TBD

1. **DistilBERT / RoBERTa** — contextual embeddings for negation and sarcasm
2. **Confidence calibration** — Platt scaling or temperature scaling on BiLSTM logits
3. **More domains** — extend beyond the 4 Blitzer categories
4. **Explainability** — LIME or attention visualisation for why the model predicted each label

---

## Slide 12 — Summary & Questions (30 sec)
**Speaker:** TBD

- Built a full ML pipeline: raw data → two trained models → live web app
- Honest comparison: classical model wins on test; neural model demonstrates architecture requirement
- 117 unit tests, real error analysis, ethics documented
- **GitHub:** `github.com/lfariabr/review-pulse`

---

## Speaker Allocation Summary

| Slide | Content | Speaker |
|---|---|---|
| 1 | Title | TBD |
| 2 | Problem | TBD |
| 3 | Dataset | TBD |
| 4 | Preprocessing | TBD |
| 5 | Architecture | TBD |
| 6 | Training | TBD |
| 7 | Results | TBD |
| 8 | Error analysis | TBD |
| 9 | Live demo | TBD |
| 10 | Ethics | TBD |
| 11 | Future work | TBD |
| 12 | Summary | TBD |

> **Note for team:** Contributions need to be confirmed. See `docs/individual-report-template.md` for contribution breakdown.

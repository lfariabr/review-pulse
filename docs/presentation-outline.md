# Presentation Outline — ReviewPulse
## ISY503 Intelligent Systems · Assessment 3 · Group Presentation
**Duration:** 12–15 minutes + Q&A
**Format:** Slides + live Streamlit demo

> **Timing guide:** 12 slides × ~60–75s average = ~13–15 min total

---

## Slide 1 — Title & Team (45s)

**Content:**
- **ReviewPulse** — Multi-domain Amazon Review Sentiment Classifier
- Team members + student IDs + roles
- Subject: ISY503 Intelligent Systems · Assessment 3
- One-line pitch: *"We trained two sentiment classifiers on 8,000 Amazon reviews and built a live web app to compare them."*

**Speaker note:**
> "Today we're presenting ReviewPulse — a sentiment classifier trained on 8,000 Amazon product reviews across four domains. We built two models, compared them honestly, and deployed the result as a live web app. I'll walk you through the data, the architecture, the results, and what we learned."

**Visual:** Use ReviewPulse `logo.jpeg` wordmark as hero image on a dark background. No new image prompt needed.

---

## Slide 2 — Problem Statement (60s)

**Content:**
- Sentiment analysis: classifying text as positive or negative opinion
- Real-world value: product feedback loops, brand monitoring, recommendation systems
- Challenge: reviews vary in **length** (2–800 words), **domain**, **writing style**, **negation**, and **sarcasm**
- Our goal: build, compare, and deploy two approaches — classical ML vs neural — and be honest about which works better

**Speaker note:**
> "Sentiment analysis is one of the most commercially deployed NLP tasks — every product review platform, every brand monitoring tool runs some form of it. The challenge isn't just accuracy on clean text. Real reviews are messy: short and long, colloquial, full of negation — 'not bad at all' — and sarcasm — 'oh great, another broken product'. We set out to build two systems and let the data decide which one wins."

**Visual — 🍌 Nano Banana #1: Sentiment Spectrum**
> Create a professional horizontal spectrum diagram in Lucidchart flowchart style, white background. A wide horizontal gradient bar from deep red on the left to deep green on the right. Five review quote boxes arranged above the bar at different positions, connected by lines to the bar below them. From left to right: "Complete waste of money" (red box), "Somewhat disappointing" (orange box), "It's okay I guess" (yellow box, centre), "Pretty good value" (light green box), "Absolutely incredible!" (dark green box). Below the gradient bar, three labels: "Negative" (left, red), "Ambiguous" (centre, yellow), "Positive" (right, green). Title above: "The Sentiment Spectrum — what our model must learn to classify". Clean sans-serif font. Output as image.

*(~60s)*

---

## Slide 3 — Dataset (75s)

**Content:**
- **Source:** Blitzer et al. (2007) — Multi-Domain Sentiment Dataset
- 8,000 labelled Amazon reviews · 4 domains: Books, DVDs, Electronics, Kitchen
- Labels from filenames (`positive.review` → 1, `negative.review` → 0)
- Perfectly balanced: **50/50 positive/negative**, 1,000 per domain per class
- **Splits:** train=5,404 / val=1,158 / test=1,159 (stratified 70/15/15, `seed=42`)
- **Label audit:** 0 ambiguous rows — no 3-star reviews included; label quality is high

**Speaker note:**
> "The dataset comes from Blitzer et al. 2007 — a classic multi-domain benchmark. Eight thousand reviews, four product categories, perfectly balanced labels. Critically, labels come from the filename — positive.review or negative.review — not from star ratings. We audited this explicitly: zero ambiguous rows, zero conflicts between the rating and the label. That's important for ethics — we know our training signal is clean."

**Visual — 🍌 Nano Banana #2: Domain Balance Chart**
> Create a professional grouped bar chart in Lucidchart flowchart style, white background. X-axis: four groups labelled "Books", "DVDs", "Electronics", "Kitchen". Each group has two bars side by side: one dark green labelled "Positive (1,000)" and one dark red labelled "Negative (1,000)". All bars are exactly the same height. Y-axis: 0 to 1,200. Add a horizontal dashed line at y=1,000 labelled "Perfect balance". Title: "Dataset Balance — 1,000 reviews per domain per class". Legend top-right. Clean sans-serif font. Output as image.

*(~75s)*

---

## Slide 4 — Preprocessing Pipeline (75s)

**Content:**
```
Raw .review files (pseudo-XML)
    → BeautifulSoup parser → extract <review_text> + <rating>
    → Label audit: flag 3-star (ambiguous) + rating/label conflicts
    → clean_text: lowercase · strip HTML · expand negations (n't → not) · remove punctuation
    → remove_outliers: drop < 10 words or > 500 words  (279 dropped, ~3.5%)
    → stratified train / val / test split (70 / 15 / 15, seed=42)
```
Key decision: **negation expansion** — *"wasn't good"* → *"was not good"* preserves the negative signal for bag-of-words models.

**Speaker note:**
> "The preprocessing pipeline has five steps. The most interesting design decision is negation expansion — contractions like 'wasn't' become 'was not'. This matters especially for TF-IDF, which treats 'not' and 'good' as independent tokens. Without this step, 'not good' and 'good' look similar to the model. We also removed outliers — 279 reviews under 10 words or over 500 — because they add noise at the extremes. Final splits: 5,404 training, 1,158 validation, 1,159 test."

**Visual — 🍌 Nano Banana #3: Preprocessing Flowchart**
> Create a professional vertical flowchart in Lucidchart style, white background. Seven rounded rectangle boxes connected by downward arrows. Box 1 (dark navy): "Raw .review Files — pseudo-XML format". Box 2 (blue): "BeautifulSoup Parser — extract review_text + rating". Box 3 (blue): "Label Audit — flag 3-star and rating/label conflicts". Box 4 (blue): "clean_text — lowercase · HTML strip · negation expansion · remove punctuation". Box 5 (blue): "remove_outliers — drop < 10 or > 500 words (279 removed)". Box 6 (blue): "Stratified Split — 70% train / 15% val / 15% test (seed=42)". Box 7 (dark green): "train=5,404 · val=1,158 · test=1,159". Add a side callout box next to Box 4: "wasn't good → was not good". Clean sans-serif font. Title: "ReviewPulse Preprocessing Pipeline". Output as image.

*(~75s)*

---

## Slide 5 — Architecture Overview (90s)

**Content:**

### Model 1 — TF-IDF + Logistic Regression (Baseline)
- `TfidfVectorizer(max_features=30k, ngram_range=(1,2), sublinear_tf=True)`
- `LogisticRegression(C=1.0, max_iter=1000)`
- Fits in seconds — no GPU required

### Model 2 — BiLSTM + GloVe (Neural)
```
Token indices (batch × 256)
    → Embedding(15,924 tokens × 100d, GloVe init — 97.4% coverage)
    → Dropout(0.5)
    → BiLSTM(hidden=256, layers=2, bidirectional, pack_padded_sequence)
    → concat final forward + backward hidden state  (512d)
    → Dropout(0.5)
    → Linear(512 → 1) → raw logit
```
Key design decision: **`pack_padded_sequence`** — LSTM processes only real words, not zero-padded tokens.

**Speaker note:**
> "We built two models. The baseline is TF-IDF plus Logistic Regression — fast, interpretable, no GPU. The neural model is a bidirectional LSTM initialised with GloVe 100-dimensional embeddings — 97.4% of our vocabulary is covered. The architecture has two layers and reads the sequence in both directions, concatenating the final forward and backward hidden states into a 512-dimensional representation before classification. One key engineering fix: we use pack_padded_sequence, which tells the LSTM to stop at each review's last real word — without it, the model wastes computation on padding zeros."

**Visual — 🍌 Nano Banana #4: BiLSTM Architecture Diagram**
> Create a professional neural network architecture diagram in Lucidchart flowchart style, white background. Vertical flow from bottom to top. Box 1 (dark navy, bottom): "Input — Token Indices (batch × 256)". Arrow up to Box 2 (blue): "Embedding Layer — 15,924 × 100d (GloVe init, 97.4% coverage)". Arrow up to Box 3 (blue): "Dropout (p=0.5)". Arrow up to Box 4 (purple, wider): "BiLSTM — hidden=256, layers=2, bidirectional · pack_padded_sequence". Inside Box 4 show two parallel horizontal arrows labelled "→ Forward" and "← Backward". Arrow up to Box 5 (blue): "Concat final hidden states (512d)". Arrow up to Box 6 (blue): "Dropout (p=0.5)". Arrow up to Box 7 (dark green, top): "Linear(512 → 1) → Raw logit → BCEWithLogitsLoss". Title: "BiLSTM + GloVe Architecture". Clean sans-serif font. Output as image.

*(~90s)*

---

## Slide 6 — Training (60s)

**Content:**
- Optimizer: **Adam** (lr=1e-3)
- Loss: **BCEWithLogitsLoss** (numerically stable sigmoid + BCE in one step)
- Gradient clipping: max_norm=5.0
- 10 epochs · checkpoint saved at best **val F1**
- Device: Apple MPS (~20 min total vs ~160 min on CPU)

| Epoch | Val Acc | Val F1 |
|---|---|---|
| 1 | 72.6% | 71.5% |
| 5 | 78.8% | 80.8% |
| 7 | 83.9% | 83.6% |
| **9 ← best** | **84.3%** | **84.0%** |
| 10 | 80.6% | 82.6% ↓ overfitting |

**Speaker note:**
> "Training ran for 10 epochs using Adam with a learning rate of 1e-3 and gradient clipping at 5.0. We track val F1 — not accuracy — as the checkpoint criterion because F1 is more robust on binary classification tasks. The best checkpoint was saved at epoch 9. At epoch 10 you can see the val loss rising while training loss continues to fall — that's the textbook sign of overfitting, and exactly why we didn't just train to epoch 10."

**Visual — 🍌 Nano Banana #5: Training Curve**
> Create a professional line chart in Lucidchart flowchart style, white background. X-axis: "Epoch" from 1 to 10. Y-axis: "F1 Score" from 0.65 to 0.90. Two lines: Line 1 (dark blue, solid): "Val F1" with data points at (1, 0.715), (2, 0.750), (3, 0.798), (4, 0.752), (5, 0.808), (6, 0.821), (7, 0.836), (8, 0.840), (9, 0.840), (10, 0.826). Line 2 (orange, dashed): "Train F1" rising steadily from 0.70 to 0.95 across epochs 1–10. Add a vertical dashed line at epoch 9 labelled "Best checkpoint saved". Add annotation at epoch 10: "↓ Overfitting". Title: "BiLSTM Training — Val F1 over 10 Epochs". Legend top-left. Output as image.

*(~60s)*

---

## Slide 7 — Results (75s)

**Content:**

### Held-out test set

| Model | Accuracy | F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | **82.7%** | **81.9%** |
| BiLSTM + GloVe | 81.0% | 80.3% |

**Honest finding:** BiLSTM beat the baseline on **validation** (84.0% vs 83.2% F1) but the baseline generalised better on the **held-out test set**.

> The BiLSTM satisfies the rubric requirement to *"define the network architecture and model class."* The comparison proves we understand when a neural model is and isn't the right tool.

**Speaker note:**
> "Here's the honest result: on the held-out test set, the simpler model wins. TF-IDF at 81.9% F1, BiLSTM at 80.3%. The BiLSTM beat the baseline on validation — 84.0% vs 83.2% — but didn't generalise as well. This is a completely legitimate ML outcome. It means the BiLSTM overfit slightly to the validation distribution during the 10-epoch run. The takeaway is not that neural models are worse — it's that more training data, longer training, or better regularisation would likely flip this result. The BiLSTM fully satisfies the rubric requirement to define a neural architecture."

**Visual — 🍌 Nano Banana #6: Model Comparison Bar Chart**
> Create a professional grouped bar chart in Lucidchart flowchart style, white background. X-axis: two groups "Accuracy" and "F1 Score". Each group has two bars: one dark blue labelled "TF-IDF Baseline" and one purple labelled "BiLSTM + GloVe". Values: Accuracy — TF-IDF 82.7%, BiLSTM 81.0%. F1 — TF-IDF 81.9%, BiLSTM 80.3%. Y-axis: 75% to 90%. Add value labels on top of each bar. Add a horizontal dashed line at 80% labelled "80% threshold". Title: "Model Comparison — Held-out Test Set". Legend top-right. Output as image.

*(~75s)*

---

## Slide 8 — Error Analysis (60s)

**Content:**
- 220 misclassified examples on the test set
- Shared failure modes across **both** models:

| Failure type | Example | Why it fails |
|---|---|---|
| **Negation** | "not bad at all" → Negative | Double negation — both models predict wrong |
| **Sarcasm** | "Oh great, stopped working after a week" | Near 50% confidence — irony not in training signal |
| **Out-of-distribution** | Delivery/logistics text | Domain shift — model generalises the wrong features |

- BiLSTM: **overconfident on short inputs** ("It is okay" → 88% Negative)
- TF-IDF: better **calibrated** near the decision boundary

**Speaker note:**
> "Two hundred and twenty reviews were misclassified. The failure modes are instructive. Negation is a shared weakness — 'not bad at all' gets predicted as Negative by both models, because 'bad' dominates the signal. Sarcasm is the hardest case — the model sits near 50% confidence, which actually shows it's uncertain, not wrong in a bad way. Out-of-distribution text — like delivery logistics reviews — shows the limit of the training domain. BiLSTM is overconfident on short inputs; TF-IDF is better calibrated near the boundary. These failure modes motivate our future work."

**Visual — 🍌 Nano Banana #7: Error Failure Mode Table**
> Create a professional three-column table diagram in Lucidchart flowchart style, white background. Three rows plus header. Header: "Failure Type" (dark navy) | "Example Review" (dark navy) | "Root Cause" (dark navy). Row 1 (light red background): "Negation" | '"not bad at all" → Negative ❌' | "Double negation confuses bag-of-words and sequence models". Row 2 (light orange background): "Sarcasm" | '"Oh great, stopped working after a week"' | "Irony not represented in training labels". Row 3 (light yellow background): "Out-of-distribution" | "Delivery logistics review" | "Domain shift beyond the 4 trained categories". Title above table: "ReviewPulse — Shared Failure Modes (220 misclassified)". Output as image.

*(~60s)*

---

## Slide 9 — Live Demo (120s)

**Content:**
```
streamlit run app.py
```

Demo script (follow in order):
1. **Clear positive:** *"This blender is incredible…"* → Positive ~98% (BiLSTM)
2. **Clear negative:** *"Broke after two days…"* → Negative ~99%
3. **Negation trap:** *"This is not bad at all"* → both predict Negative ← discuss
4. **Sarcasm:** *"Oh great, another product that stopped working"* → low confidence ← discuss
5. **💡 Generate** button — random sample

**Speaker note:**
> "Let's run the app live. [Open browser to localhost:8501] The sidebar shows our logo, the main panel has two model options — baseline is default, BiLSTM is optional. [Run case 1 — clear positive] High confidence, correct. [Run case 2 — clear negative] Both models agree, very high confidence. [Run case 3 — negation] Both predict Negative — this is the failure mode we just described. [Run case 4 — sarcasm] Notice the confidence drops to 52–64% — the model is uncertain, which is honest. [Hit Generate] This loads a random sample from our 10 acceptance test cases."

**Visual:** Live app — no image prompt needed. Have `streamlit run app.py` already running before the presentation starts.

*(~120s)*

---

## Slide 10 — Ethics & Limitations (75s)

**Content:**

| Concern | Detail |
|---|---|
| **Label quality** | Labels from filenames, not human raters — audited explicitly: 0 ambiguous rows |
| **Domain bias** | 4 product categories only — generalisation to other domains is untested |
| **Negation & sarcasm** | Systematic failure mode in both models |
| **Confidence calibration** | BiLSTM logits are uncalibrated — 99% ≠ 99% accuracy |
| **Dataset age** | Blitzer et al. 2007 — ~20-year-old reviews; language patterns may have shifted |
| **Deployment risk** | App should not be used for high-stakes decisions without human review |

**Speaker note:**
> "Ethics is not an afterthought — it's part of the design. We audited labels explicitly and found zero conflicts. But the dataset is from 2007, which means the language of product reviews has shifted — emoji, slang, and platform conventions weren't in the training data. BiLSTM confidence values aren't calibrated — 98% confidence doesn't mean 98% accuracy. And the app is a demo, not a decision system. Any production deployment of a sentiment classifier needs human oversight, especially in high-stakes contexts like content moderation or product recalls."

**Visual:** Use Slide 5 Nano Banana table format, no new image prompt needed. Ethics table fits cleanly as a formatted slide.

*(~75s)*

---

## Slide 11 — Future Work (45s)

**Content:**
1. **DistilBERT / RoBERTa** — contextual embeddings for negation and sarcasm
2. **Confidence calibration** — Platt scaling or temperature scaling on BiLSTM logits
3. **More domains** — extend beyond the 4 Blitzer categories
4. **Explainability** — LIME or attention visualisation to show *why* the model predicted each label

**Speaker note:**
> "Four clear next steps. DistilBERT would handle negation and sarcasm through contextual attention — it's the obvious upgrade. Calibration would make confidence scores trustworthy. More domains would validate generalisation. And LIME or attention visualisation would make the model explainable — which matters if you're presenting results to a non-technical stakeholder."

**Visual:** Clean numbered list slide — no image prompt needed.

*(~45s)*

---

## Slide 12 — Summary & Questions (30s)

**Content:**
- Full pipeline: raw data → preprocessing → two trained models → live web app
- **Classical model wins on test** — TF-IDF F1 81.9% vs BiLSTM 80.3%
- **Neural model demonstrates architecture** — satisfies rubric, shows ML maturity
- 117 unit tests · real error analysis · ethics documented
- **GitHub:** `github.com/lfariabr/review-pulse`

**Speaker note:**
> "To summarise: we built a complete ML pipeline from raw data to a deployed app. The honest result is that the simpler model generalises better on the test set — but the BiLSTM satisfies the neural network requirement and demonstrates we understand architecture design. 117 tests, a real error analysis, ethics reviewed. The code is on GitHub. We're happy to take questions."

**Visual:** Summary bullet slide — no image prompt needed. ReviewPulse `logo-icon.png` bottom-right.

*(~30s)*

---

## Nano Banana Summary

| # | Slide | Description | Status |
|---|---|---|---|
| #1 | 2 | Sentiment spectrum — gradient bar with review quotes | New — prompt above |
| #2 | 3 | Domain balance grouped bar chart | New — prompt above |
| #3 | 4 | Preprocessing flowchart (7 steps) | New — prompt above |
| #4 | 5 | BiLSTM architecture vertical flow diagram | New — prompt above |
| #5 | 6 | Training curve — val F1 vs train F1 over 10 epochs | New — prompt above |
| #6 | 7 | Model comparison grouped bar chart | New — prompt above |
| #7 | 8 | Error failure mode table | New — prompt above |
| — | 1 | Title — use logo.jpeg | No prompt needed |
| — | 9 | Live demo — run app | No prompt needed |
| — | 10 | Ethics — text table | No prompt needed |
| — | 11 | Future work — numbered list | No prompt needed |
| — | 12 | Summary — bullets + logo | No prompt needed |

---

## Verification Checklist

- [ ] 12 slides total — timed at 12–15 min
- [ ] Every key pipeline stage represented (data → preprocessing → model → training → results → demo → ethics)
- [ ] Both models covered with honest comparison
- [ ] Live demo slide with exact script and fallback note
- [ ] Each slide has: Content, Speaker note, Visual instruction or Nano Banana prompt
- [ ] 7 Nano Banana prompts — all Lucidchart style, image output
- [ ] Ethics covered explicitly (Slide 10)
- [ ] Future work motivates DistilBERT as natural extension
- [ ] GitHub link on summary slide

---

## Design Notes

- **Style:** Clean white background + dark navy headers (matches Lucidchart Nano Banana format)
- **Accent colour:** Deep blue / purple for BiLSTM, green for positive results, red for negative
- **Logo:** `logo.jpeg` wordmark on title slide · `logo-icon.png` bottom-right on summary
- **Font:** Sans-serif throughout — no decorative fonts
- **Speaker allocation:** All slots marked TBD — assign during team sync before submission

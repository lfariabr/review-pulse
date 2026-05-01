# Demo Acceptance Test Cases — ReviewPulse

Facilitator-style test inputs for the live demo and presentation.
All outputs recorded from real trained models (baseline checkpoint `outputs/baseline.joblib`, BiLSTM checkpoint `outputs/bilstm.pt` epoch 9).

Run the app with:
```bash
streamlit run app.py
```

---

## Test Cases

### 1 — Clear Positive
> *Expected: Positive, high confidence on both models.*

**Input:**
```
This blender is absolutely incredible. Smoothies in 30 seconds, easy to clean, still going strong after six months.
```

| Model | Label | Confidence |
|---|---|---|
| TF-IDF baseline | Positive review | 73.8% |
| BiLSTM + GloVe | Positive review | 97.9% |

---

### 2 — Clear Negative
> *Expected: Negative, high confidence on both models.*

**Input:**
```
Broke after two days. Cheap plastic, terrible build quality. Complete waste of money — do not buy.
```

| Model | Label | Confidence |
|---|---|---|
| TF-IDF baseline | Negative review | 95.9% |
| BiLSTM + GloVe | Negative review | 99.6% |

---

### 3 — Short Ambiguous
> *Expected: low confidence — "it is okay" carries weak sentiment signal.*

**Input:**
```
It is okay I guess.
```

| Model | Label | Confidence |
|---|---|---|
| TF-IDF baseline | Negative review | 54.8% |
| BiLSTM + GloVe | Negative review | 88.1% |

**Note:** Both models predict negative, but TF-IDF confidence is near the 50% decision boundary — correctly uncertain. BiLSTM is overconfident on very short inputs; a known limitation of sequence models with minimal context.

---

### 4 — Negation Trap
> *Expected: Positive — "not bad" and "quite good" are positive. A model without negation handling would fail.*

**Input:**
```
This is not bad at all, actually quite good.
```

| Model | Label | Confidence |
|---|---|---|
| TF-IDF baseline | Negative review | 66.9% |
| BiLSTM + GloVe | Negative review | 74.6% |

**Note:** Both models misclassify this. Negation expansion (`n't` → `not`) is applied in preprocessing, but double negation ("not bad") remains a hard case. Good discussion point for the ethics/limitations section.

---

### 5 — Domain-Shifted (Books)
> *Expected: Positive — vocabulary ("plot twist", "thriller") is outside the core product-review lexicon but sentiment is clear.*

**Input:**
```
The plot twist in chapter 12 completely blindsided me. One of the best thrillers I have read in years.
```

| Model | Label | Confidence |
|---|---|---|
| TF-IDF baseline | Positive review | 69.4% |
| BiLSTM + GloVe | Positive review | 86.2% |

---

### 6 — Outside Training Distribution (Logistics)
> *Expected: uncertain — this is about delivery, not product quality. Neither label is clearly correct.*

**Input:**
```
Delivery was late but the driver apologised. Packaging was undamaged.
```

| Model | Label | Confidence |
|---|---|---|
| TF-IDF baseline | Negative review | 63.2% |
| BiLSTM + GloVe | Negative review | 92.7% |

**Note:** Both predict negative — "late" likely drives this. Demonstrates distribution shift risk when the model is deployed on non-product-review text.

---

### 7 — Mixed Sentiment
> *Expected: the overall review skews negative ("very disappointed overall") — but this is genuinely ambiguous.*

**Input:**
```
Great camera but the battery life is absolutely terrible. Very disappointed overall.
```

| Model | Label | Confidence |
|---|---|---|
| TF-IDF baseline | Negative review | 62.8% |
| BiLSTM + GloVe | Negative review | 77.5% |

---

### 8 — Very Short Positive
> *Expected: Positive — "Love it!" is an unambiguous signal despite having only two tokens.*

**Input:**
```
Love it!
```

| Model | Label | Confidence |
|---|---|---|
| TF-IDF baseline | Positive review | 94.8% |
| BiLSTM + GloVe | Positive review | 86.1% |

---

### 9 — Very Short Negative
> *Expected: Negative — single-word review.*

**Input:**
```
Junk.
```

| Model | Label | Confidence |
|---|---|---|
| TF-IDF baseline | Negative review | 67.2% |
| BiLSTM + GloVe | Negative review | 94.4% |

---

### 10 — Sarcasm
> *Expected: negative intent — but sarcasm is extremely hard for bag-of-words and sequence models alike.*

**Input:**
```
Oh great, another product that stopped working after a week. Just what I needed.
```

| Model | Label | Confidence |
|---|---|---|
| TF-IDF baseline | Negative review | 52.5% |
| BiLSTM + GloVe | Negative review | 64.6% |

**Note:** Both correctly predict negative but with low confidence. TF-IDF is near the boundary — "great" and "needed" carry positive weight. A transformer model with contextual embeddings (e.g. DistilBERT) would likely handle this better.

---

## Summary

| Case | Baseline | BiLSTM | Agreement |
|---|---|---|---|
| Clear positive | ✅ | ✅ | ✅ |
| Clear negative | ✅ | ✅ | ✅ |
| Short ambiguous | ⚠️ low conf | ❌ overconfident | ✅ same label |
| Negation trap | ❌ | ❌ | ✅ same error |
| Domain-shifted | ✅ | ✅ | ✅ |
| Outside distribution | ⚠️ | ⚠️ | ✅ same label |
| Mixed sentiment | ✅ | ✅ | ✅ |
| Very short positive | ✅ | ✅ | ✅ |
| Very short negative | ✅ | ✅ | ✅ |
| Sarcasm | ⚠️ low conf | ⚠️ low conf | ✅ |

**Key talking points for the presentation:**
- Both models agree on 9/10 cases — consistent behaviour across architectures
- Negation trap is a shared failure — addressable with improved negation preprocessing
- Sarcasm and out-of-distribution inputs are the natural next frontier — motivates DistilBERT as a future extension
- BiLSTM tends toward higher confidence, even on weak inputs — a calibration consideration

# Individual Contribution Report — ISY503 Assessment 3
**Target length:** 250 words ±10% per report

---

## Slide Assignment

| Presenter | Slides | Content | Approx. time |
|---|---|---|---|
| **Luis** | 1, 4, 5, 6, 7, 12 | Title · Preprocessing · Architecture · Training · Results · Summary | ~6.25 min |
| **Victor** | 3, 8, 10 | Dataset · Error Analysis · Ethics & Limitations | ~3.5 min |
| **Samiran** | 2, 9, 11 | Problem Statement · Live Demo · Future Work | ~3.75 min |

For speaker notes and suggested wording per slide — work from `docs/presentation-outline.md`.

---

## Luis Faria — A00187785

### Student Details

- **Name:** Luis G. B. A. Faria
- **Student ID:** A00187785
- **Slides presented:** 1, 4, 5, 6, 7, 12 (Title · Preprocessing · Architecture · Training · Results · Summary)
- **GitHub:** https://github.com/lfariabr/review-pulse

### Team Contribution Table

| Team Member | Student ID | Main Contribution | % |
|---|---|---|---:|
| Luis Faria | A00187785 | Full technical implementation, app, tests, docs | 65% |
| Victor Meneses | A00179705 | Dataset, error analysis, ethics presentation | 17.5% |
| Samiran Shrestha | A00106473 | Problem framing, live demo, future work | 17.5% |
| **Total** | | | **100%** |

### Draft Report (~250 words)

My primary contribution to ReviewPulse was the full technical implementation of the ML pipeline — nine source modules, the Streamlit web app, 117 unit tests, and all project documentation.

On the data side, I built the pseudo-XML parser (`parser.py`), the preprocessing pipeline with negation expansion (`preprocess.py`), and the vocabulary builder and PyTorch DataLoaders (`dataset.py`). I then implemented both models: the TF-IDF + Logistic Regression baseline (`baseline.py`) and the bidirectional LSTM initialised with GloVe 100-dimensional embeddings (`model.py`). The training loop (`train.py`) uses Adam with gradient clipping, F1-based checkpointing, and Apple MPS device support. I implemented evaluation with confusion matrix and error analysis (`evaluate.py`), a unified inference API (`inference.py`), and the Streamlit web app with model selector and sample review generator (`app.py`). I also produced the presentation outline, 10 acceptance test cases with real model outputs, and this document.

An important ethical consideration is that the Blitzer et al. (2007) dataset uses filename-derived labels rather than human raters. Because the dataset uses filename-derived labels rather than direct human annotation, we audited for possible rating/text conflicts and ambiguous boundary cases. In this dataset, we found zero ambiguous rows, but the risk remains relevant in broader sentiment classification settings. BiLSTM confidence values are also uncalibrated — 98% confidence does not imply 98% accuracy — and the model generalises poorly to out-of-distribution text such as logistics reviews. Any production deployment requires human oversight and periodic label audits.

I estimate my contribution at 65% as the primary technical implementer. Victor contributed 17.5% covering dataset analysis, error analysis, and ethics. Samiran contributed 17.5% covering the problem framing, live demo delivery, and future work.

### APA References

- Blitzer, J., Dredze, M., & Pereira, F. (2007). Biographies, Bollywood, Boom-boxes and Blenders: Domain adaptation for sentiment classification. In *Proceedings of the 45th Annual Meeting of the ACL* (pp. 440–447). ACL. https://aclanthology.org/P07-1056/
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
- Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the dangers of stochastic parrots: Can language models be too big? In *Proceedings of the 2021 ACM FAccT Conference* (pp. 610–623). https://doi.org/10.1145/3442188.3445922

---

## Victor Meneses — A00179705

### Student Details

- **Name:** Victor Meneses
- **Student ID:** A00179705
- **Slides presented:** 3, 8, 10 (Dataset · Error Analysis · Ethics & Limitations)

### Team Contribution Table

| Team Member | Student ID | Main Contribution | % |
|---|---|---|---:|
| Luis Faria | A00187785 | Full technical implementation, app, tests, docs | 65% |
| Victor Meneses | A00179705 | Dataset, error analysis, ethics presentation | 17.5% |
| Samiran Shrestha | A00106473 | Problem framing, live demo, future work | 17.5% |
| **Total** | | | **100%** |

### Draft Report (~250 words) — suggested content

My primary contribution to ReviewPulse was presenting the dataset analysis, error analysis, and ethical considerations, and contributing to [any additional work: research, documentation, team coordination].

I presented slides 3, 8, and 10. Slide 3 covers the Blitzer et al. (2007) dataset — 8,000 Amazon reviews across four domains (Books, DVDs, Electronics, Kitchen), perfectly balanced at 50/50 positive and negative, with labels derived from filenames rather than star ratings. I explained the stratified 70/15/15 split and the result of our label audit: zero ambiguous or conflicting rows, confirming the training signal is clean.

Slide 8 covers the 220 misclassified examples and their failure modes. Negation is a shared weakness — "not bad at all" gets predicted as Negative by both models because "bad" dominates the signal. Sarcasm produces low confidence near 50%, which is actually honest uncertainty rather than a hard error. Out-of-distribution text such as logistics reviews shows the limit of the training domain.

Slide 10 addresses ethics: label noise risk from filename-based labelling, domain bias across only four product categories, uncalibrated BiLSTM confidence scores, the 20-year age of the dataset, and the deployment risk of using the app for high-stakes decisions without human review.

An important ethical consideration I want to highlight is dataset age. The Blitzer et al. (2007) data was collected approximately 20 years ago; emoji, platform-specific slang, and short-form text are common today but absent from training. Periodic retraining on modern review data would reduce this distributional shift risk.

I estimate my contribution at 17.5% covering dataset analysis, error analysis, and ethics. Luis contributed 65% as the primary technical implementer. Samiran contributed 17.5% covering the problem framing, live demo delivery, and future work.

### APA References

- Blitzer, J., Dredze, M., & Pereira, F. (2007). Biographies, Bollywood, Boom-boxes and Blenders: Domain adaptation for sentiment classification. In *Proceedings of the 45th Annual Meeting of the ACL* (pp. 440–447). ACL. https://aclanthology.org/P07-1056/
- Pang, B., & Lee, L. (2008). Opinion mining and sentiment analysis. *Foundations and Trends in Information Retrieval, 2*(1–2), 1–135. https://doi.org/10.1561/1500000011

---

## Samiran Shrestha — A00106473

### Student Details

- **Name:** Samiran Shrestha
- **Student ID:** A00106473
- **Slides presented:** 2, 9, 11 (Problem Statement · Live Demo · Future Work)

### Team Contribution Table

| Team Member | Student ID | Main Contribution | % |
|---|---|---|---:|
| Luis Faria | A00187785 | Full technical implementation, app, tests, docs | 65% |
| Victor Meneses | A00179705 | Dataset, error analysis, ethics presentation | 17.5% |
| Samiran Shrestha | A00106473 | Problem framing, live demo, future work | 17.5% |
| **Total** | | | **100%** |

### Draft Report (~250 words) — suggested content

My primary contribution to ReviewPulse was framing the problem for the audience, running the live demo, and presenting the future work roadmap, as well as contributing to [any additional work: research, documentation, team coordination].

I presented slides 2, 9, and 11. Slide 2 establishes the commercial motivation for sentiment analysis — product feedback loops, brand monitoring, recommendation systems — and articulates the core challenge: real reviews are messy. They vary in length from two words to over 800, span multiple domains, use negation ("not bad at all"), and employ sarcasm ("oh great, another broken product"). I framed our goal as building two systems and letting the data decide which one wins.

Slide 9 is the live demo. I ran the Streamlit app live, demonstrating a clear positive, a clear negative, the negation failure mode, and the sarcasm case where both models sit near 50% confidence — which is honest uncertainty, not a hard error. I also showed the Generate button to load a random acceptance test case.

Slide 11 presents four concrete next steps: DistilBERT or RoBERTa for contextual embeddings that would handle negation and sarcasm through attention; Platt scaling for confidence calibration; additional training domains to validate generalisation; and LIME for explainability.

An important ethical consideration I focused on is that binary sentiment classification is reductive. Real reviews express nuanced opinions — mixed, hedged, or ironic — that a positive/negative label cannot capture. Deploying a binary classifier in high-stakes contexts risks suppressing legitimate nuance.

I estimate my contribution at 17.5% covering the problem framing, live demo delivery, and future work. Luis contributed 65% as the primary technical implementer. Victor contributed 17.5% covering dataset analysis, error analysis, and ethics.

### APA References

- Blitzer, J., Dredze, M., & Pereira, F. (2007). Biographies, Bollywood, Boom-boxes and Blenders: Domain adaptation for sentiment classification. In *Proceedings of the 45th Annual Meeting of the ACL* (pp. 440–447). ACL. https://aclanthology.org/P07-1056/
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019* (pp. 4171–4186). https://doi.org/10.18653/v1/N19-1423

---

## Notes for all team members

- **Word count:** 250 words ±10% (225–275). The draft sections above are calibrated to this target — trim or expand as needed.
- **Percentage split:** Must total exactly 100%. Agree as a team before submitting individually.
- **Ethics paragraph:** Each report includes a distinct ethical angle — do not duplicate across reports.
- **Speaker notes:** Work from `docs/presentation-outline.md` first for slide-specific wording.
- **GitHub evidence:** Luis's commit history is the primary evidence record. Reference specific commits or issues if asked to justify contribution claims.

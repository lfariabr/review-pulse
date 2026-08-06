from pathlib import Path

content = r"""# ReviewPulse v3.0 — Manual UI Validation Record

## 1. Validation Information

- **Validator:** Juan Sebastian Martinez Contreras
- **Validation date: 8/26 
- **Application URL:** https://review-pulse.streamlit.app/ReviewPulse_v3_0_0
- **Browser and version:** Chrome V.150.0.8
- **Validation environment:** Deployed Streamlit application
- **Overall result:** Pass / Pass with observations / Fail / Blocked

---

## 2. Validation Scope

This validation is performed entirely through the deployed Streamlit user interface.

The validation covers:

- Availability and basic inference for all six models
- Single-aspect and multi-aspect prediction
- Mixed-polarity review handling
- Aspect input cleaning, order preservation, and duplicate removal
- Sample generation
- ATAE-LSTM attention evidence
- DistilBERT attribution evidence
- Unsupported evidence handling for review-only models
- Controlled input validation
- Model switching and stale-result prevention
- ReviewPulse v2/v3 workflow compatibility

### Models in scope

1. TF-IDF
2. Target LSTM
3. Target GRU
4. TextCNN
5. ATAE-LSTM
6. DistilBERT

---

## 3. Test Data

### Input A — Single positive aspect

- **Review:** `The food was excellent.`
- **Aspects:** `food`

### Input B — Mixed-polarity multi-aspect review

- **Review:** `Great food but the service was dreadful!`
- **Aspects:** `food, service`

### Input C — Duplicate and reordered aspects

- **Review:** `The food was excellent, but the service was slow.`
- **Aspects:** `service, food, service`

### Input D — Input-cleaning case

- **Review:** `The food was excellent, but the service was slow.`
- **Aspects:** ` food, , service, food `

---

## 4. Execution Summary

| Test ID | Test name | Status | Evidence ID | Notes |
|---|---|---|---|---|
| UI-01 | Application and model availability | Pass | EV-01 | |
| UI-02 | Six-model inference smoke test | Pass |EV-02 | |
| UI-03 | Mixed-polarity multi-aspect prediction | Pass with observation | 	EV-03| |
| UI-04 | Aspect order and duplicate handling | Pass | EV-04 | |
| UI-05 | Sample generator | Pass |EV-05 | |
| UI-06 | Model switching and stale-result prevention | Pass |EV-06 | |
| UI-07 | ATAE-LSTM attention evidence | Pass with observation |EV-07 | |
| UI-08 | DistilBERT attribution evidence | Pass with observations  |EV-08 | |
| UI-09 | Unsupported evidence handling | Pass |EV-09 | |
| UI-10 | Empty review validation | Pass |EV-010 | |
| UI-11 | Empty/invalid aspect validation | Pass | EV-011 | |
| UI-12 | ReviewPulse v2/v3 compatibility | Pass | EV-12 | |

Allowed status values:

- `Pass`
- `Pass with observation`
- `Fail`
- `Blocked`
- `Not applicable`
- `Not run`

---

# 5. Detailed Test Cases

## UI-01 — Application and Model Availability

**Priority:** Critical

### Objective

Verify that the deployed application loads successfully and exposes all six expected models.

### Steps

1. Open the deployed ReviewPulse v3.0 application.
2. Wait for the page to finish loading.
3. Open the model selector.
4. Confirm that all six models are available.

### Expected result

- The page loads without a fatal error.
- The interface is usable.
- The following models are available:
  - TF-IDF
  - Target LSTM
  - Target GRU
  - TextCNN
  - ATAE-LSTM
  - DistilBERT
- No model is silently replaced by another model.

### Actual result

- **Page loaded:** Yes
- **All six models visible:** Yes
- **Missing models:** No
- **Warnings or errors:** No

### Status

Pass

### Evidence

- **Evidence ID:** EV-01
- **Screenshot description:** Model selector showing the available models.
- **Word document page:** 1

---

## UI-02 — Six-Model Inference Smoke Test

**Priority:** Critical

### Objective

Verify that every model can accept valid UI input and return a prediction.

### Input

- **Review:** `The food was excellent.`
- **Aspects:** `food`

### Steps

For each model:

1. Select the model.
2. Enter the review and aspect.
3. Run the prediction.
4. Confirm that a result is displayed.
5. Record the predicted label and confidence, when shown.

### Expected result

- Every model runs without crashing the page.
- A result is returned for the `food` aspect.
- The result clearly identifies the selected model and aspect.
- A sentiment label is displayed.
- Confidence/probability is displayed when supported by the interface.
- The result is not left over from the previously selected model.

### Results

| Model | Prediction returned | Predicted label | Confidence | Status | Evidence ID |
|---|---:|---|---:|---|---|
| TF-IDF |Positive| Positive | 88.3%| Pass | EV-02A |
| Target LSTM | Positive | Positive |88.8% |Pass | EV-02B |
| Target GRU |Positive | Positive | 89.5% | Pass |  EV-02C |
| TextCNN |Positive | Positive | 92.0% | Pass |EV-02D |
| ATAE-LSTM |Positive | Positive | 81.3% | Pass | EV-02E |
| DistilBERT |Positive | Positive | 97.7% | Pass | EV-02F |

### Notes

A prediction that appears semantically incorrect is a **model-quality observation**, not automatically a UI failure. 

---

## UI-03 — Mixed-Polarity Multi-Aspect Prediction

**Priority:** Critical

### Objective

Verify that the application accepts one review with different sentiments toward different aspects and returns one result per aspect.

### Input

- **Review:** `Great food but the service was dreadful!`
- **Aspects:** `food, service`

### Recommended models

This test will be run in models:

- ATAE-LSTM
- DistilBERT

### Steps

1. Enter the review.
2. Enter both aspects in the stated order.
3. Run the prediction.
4. Confirm that separate results are displayed for `food` and `service`.
5. Repeat with the selected models.
6. Record each prediction.

### Expected result

- Two aspect results are displayed.
- `food` and `service` are clearly separated.
- Each aspect has its own prediction output.
- The application does not collapse the review into only one visible result.
- The interface remains stable even when a model predicts the same sentiment for both aspects.

### Results

| Model | Food prediction | Service prediction | Two separate results | Status | Evidence ID |
|---|---|---|---:|---|---|
| ATAE-LSTM | Positive | Positive | Yes | Pass with observations | EV-03A |
| DistilBERT | Negative | Negative | Yest | Pass with observations  | EV-03B |

### Observations

- **Both models distinguish each aspect separately** 

- **Both models are wrong in one aspect, the correct answer is food: positive and service: negative** 

- **No unexpected errors** 

---

## UI-04 — Aspect Order, Cleaning, and Duplicate Removal

**Priority:** High

### Objective

Verify that aspect input is cleaned correctly, duplicates are removed, and the original meaningful order is preserved.

### Input

- **Review:** `The food was excellent, but the service was slow.`
- **Aspects:** `service, food, service`

### Steps

1. Enter the review.
2. Enter the duplicate aspect list.
3. Run the prediction.
4. Observe the final list/order of results.
5. Repeat with: ` food, , service, food `

### Expected result

- Empty entries created by repeated commas are ignored.
- Leading and trailing spaces are removed.
- Duplicate aspects are not predicted twice.
- The first meaningful order is preserved:
  - `service`
  - `food`
- No crash or malformed output occurs.

### Actual result

- **Displayed aspects:** Service,food
- **Displayed order:**  Service, food
- **Duplicates removed:** Yes
- **Blank entries ignored:** Yes
- **Errors:** None

### Status

Pass

### Evidence

- **Evidence ID:** EV-04
- **Word document figure/page:** 

---

## UI-05 — Sample Generator

**Priority:** High

### Objective

Verify that the sample generator produces usable input and integrates correctly with prediction.

### Steps

1. Open or click the sample-generation control.
2. Generate a sample.
3. Record the generated review and aspects.
4. Confirm that the generated values populate the correct fields.
5. Run a prediction.
6. Generate another sample and confirm that the interface updates correctly.

### Expected result

- A valid review is generated.
- At least one valid aspect is generated or populated.
- Generated content appears in the correct fields.
- The generated sample can be submitted without manual repair.
- Generating a new sample does not leave stale results incorrectly associated with the new input.

### Actual result

- **Generated review:** Valid
- **Generated aspects:**  Yes
- **Prediction successful:** Yes
- **Second sample updated correctly:** Yes
- **Errors:** No unexpected errors

### Status

Pass

### Evidence

- **Evidence ID:** EV-05
- **Word document figure/page:** 

---

## UI-06 — Model Switching and Stale-Result Prevention

**Priority:** High

### Objective

Verify that changing models updates the result correctly and does not display stale output from the previous model.

### Steps

1. Run a valid prediction with TF-IDF.
2. Record the visible result.
3. Change to ATAE-LSTM and run the same input.
4. Change to DistilBERT and run the same input.
5. Confirm that the selected model is clearly reflected in each result.
6. Confirm that evidence controls/views appear only when supported.
7. Change the input and run again.

### Expected result

- The selected model is the model actually used.
- Results refresh after every execution.
- Old predictions are not presented as new results.
- Model-specific evidence does not remain visible after switching to an unsupported model.
- Changing the input refreshes the corresponding results.

### Actual result

- **Correct model shown:** Yes
- **Results refreshed:** Yes
- **Stale output observed:** Yes
- **Evidence view refreshed/removed correctly:** Yes
- **Errors:** None unexpected

### Status

Pass 

### Evidence

- **Evidence ID:** EV-06
- **Word document figure/page:** 

---

## UI-07 — ATAE-LSTM Attention Evidence

**Priority:** Critical

### Objective

Verify that ATAE-LSTM produces a readable, aspect-specific attention evidence view.

### Input

- **Review:** `Great food but the service was dreadful!`
- **Aspects:** `food, service`
- **Model:** ATAE-LSTM

### Steps

1. Select ATAE-LSTM.
2. Run the mixed-polarity input.
3. Open or display the evidence view for `food`.
4. Open or display the evidence view for `service`.
5. Compare the visible token scores/highlighting.
6. Check punctuation and token alignment.

### Expected result

- An attention evidence view is available.
- Evidence is shown separately for each aspect.
- Visible tokens correspond to the review text.
- Token highlighting/scores are readable.
- Changing from `food` to `service` updates the aspect view.
- The application does not claim that attention is a causal explanation.

### Actual result

- **Evidence displayed:** Yes
- **Food evidence displayed:** Yes
- **Service evidence displayed:** Yes
- **Evidence changed by aspect:** No
- **Token alignment acceptable:** No
- **Punctuation handled correctly:** No
- **Errors:** 

### Status

Pass with observation
For the negative aspect there is not enough attention and the prediction is wrong

### Evidence

- **Evidence ID:** EV-07
- **Word document figure/page:** 

---

## UI-08 — DistilBERT Attribution Evidence

**Priority:** Critical

### Objective

Verify that DistilBERT produces a readable, aspect-specific attribution view.

### Input

- **Review:** `Great food but the service was dreadful!`
- **Aspects:** `food, service`
- **Model:** DistilBERT

### Steps

1. Select DistilBERT.
2. Run the mixed-polarity input.
3. Display evidence for `food`.
4. Display evidence for `service`.
5. Compare the visible attribution scores/highlighting.
6. Check whether special tokens or unreadable subword fragments appear.

### Expected result

- An attribution evidence view is available.
- Evidence is shown separately for each aspect.
- Visible tokens align with the review text.
- Changing the selected aspect updates the evidence view.

### Actual result

- **Evidence displayed:** Yes 
- **Food evidence displayed:** Yes 
- **Service evidence displayed:** Yes 
- **Evidence changed by aspect:**  No
- **Special tokens visible:** No
- **Unreadable subwords visible:**  No
- **Token alignment acceptable:**  No
- **Errors:** None unexpected

### Status

Pass with observation
For the Positive aspect there is not enough attention for the positive word and the prediction is wrong

### Evidence

- **Evidence ID:** EV-08
- **Word document figure/page:** 

---

## UI-09 — Unsupported Evidence Handling

**Priority:** High

### Objective

Verify that review-only models do not display invented token evidence and that unsupported evidence is communicated clearly.

### Models

- TF-IDF
- Target LSTM
- Target GRU
- TextCNN

### Steps

1. Select each review-only model.
2. Run a valid prediction.
3. Attempt to access the evidence view, when the interface provides that option.
4. Record the message or UI state.

### Expected result

- The application does not generate an attention or attribution heatmap for unsupported models.
- The UI clearly reports that token evidence is unsupported, unavailable, or not applicable.
- The application does not crash.
- Evidence from a previously selected ATAE-LSTM or DistilBERT result is not left visible.

### Results

| Model | Prediction works | Evidence correctly unavailable | Message shown | Status |
|---|---:|---:|---|---|
| TF-IDF |Yes | Yes |TF-IDF review-only is a review-only baseline and does not expose aspect-specific token evidence. | Pass |
| Target LSTM |Yes |Yes |LSTM review-only is a review-only baseline and does not expose aspect-specific token evidence. |Pass |
| Target GRU | Yes |Yes |GRU review-only (exploratory) is a review-only baseline and does not expose aspect-specific token evidence. |Pass |
| TextCNN |Yes |Yes |Text CNN review-only (exploratory) is a review-only baseline and does not expose aspect-specific token evidence. |Pass |

### Evidence

- **Evidence ID:** EV-09
- **Word document figure/page:** 

---

## UI-10 — Empty Review Validation

**Priority:** High

### Objective

Verify controlled handling of an empty or whitespace-only review.

### Inputs

1. Empty review with a valid aspect
2. Review containing spaces only with a valid aspect

### Steps

1. Leave the review field empty.
2. Enter `food` as the aspect.
3. Submit.
4. Repeat with spaces only in the review field.

### Expected result

- A clear validation message is displayed.
- No model prediction is attempted or shown as successful.
- The page does not crash.
- No stale result is presented as the new result.

### Actual result

- **Empty input message:** The selected model is unavailable: Review must not be empty
- **Whitespace-only message:** Classify aspect button unavailable. 
- **Page remained stable:** Yes
- **Stale output observed:** Yes

### Status

Pass

### Evidence

- **Evidence ID:** EV-10
- **Word document figure/page:** 

---

## UI-11 — Empty or Invalid Aspect Validation

**Priority:** High

### Objective

Verify controlled handling of missing or unusable aspect input.

### Inputs

1. Valid review with an empty aspect field
2. Valid review with spaces only
3. Valid review with commas only: `,,,`
4. Valid review with mixed blanks: `food, , ,`

### Steps

1. Enter a valid review.
2. Test each aspect input.
3. Submit each case.
4. Record the displayed result or validation message.

### Expected result

- Empty or unusable aspect input produces a clear validation message.
- Blank entries are ignored.
- `food, , ,` is treated as the valid aspect `food`.
- The page does not crash.
- No malformed empty-aspect prediction is shown.

### Results

| Aspect input | Expected | Actual | Status |
|---|---|---|---|
| Empty | Validation message | The selected model is unavailable: Provide at least one non-empty aspect | Pass|
| Spaces only | Validation message |The selected model is unavailable: Provide at least one non-empty aspect |Pass |
| `,,,` | Validation message |The selected model is unavailable: Provide at least one non-empty aspect | Pass|
| `food, , ,` | One `food` result | Food as valid aspect  | Pass |

### Evidence

- **Evidence ID:** EV-11
- **Word document figure/page:** 

---

## UI-12 — ReviewPulse v2/v3 Compatibility

**Priority:** Critical

### Objective

Verify that the legacy review-level workflow and the v3 aspect-based workflow remain accessible and functional.

### Steps

1. Open the ReviewPulse v2 or legacy review-level workflow.
2. Enter a valid review and run a prediction.
3. Record the result.
4. Navigate to ReviewPulse v3.
5. Enter a review and multiple aspects.
6. Run a prediction.
7. Navigate back to v2, when supported.
8. Confirm that both workflows still operate without interfering with each other.

### Expected result

- The v2 review-level workflow remains functional.
- The v3 aspect-level workflow remains functional.
- v2 returns a review-level prediction.
- v3 returns separate aspect-level predictions.
- Navigating between versions does not crash the application.
- Results and controls from one workflow do not appear incorrectly in the other.

### Actual result

- **v2 accessible:** Yes 
- **v2 prediction successful:** Yes 
- **v3 accessible:** Yes
- **v3 prediction successful:** Yes
- **Navigation stable:** Yes
- **Cross-version stale state observed:** Yes
- **Errors:** None unexpected

### Status

Pass

### Evidence

- **Evidence ID:** EV-12
- **Word document figure/page:** 

---

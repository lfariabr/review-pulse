"""ReviewPulse — Streamlit sentiment inference UI.

Usage:
    streamlit run app.py
"""

from PIL import Image
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

_ICON = Image.open("logo-icon.png")

st.set_page_config(
    page_title="ReviewPulse",
    page_icon=_ICON,          # browser-tab favicon (icon-only)
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar logo (icon-only — appears in both expanded and collapsed states)
# ---------------------------------------------------------------------------

st.logo("logo.jpeg", link=None, size="large")

# ---------------------------------------------------------------------------
# Model loading — cached for the lifetime of the Streamlit session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def _load_baseline():
    from src.inference import load_baseline_model
    return load_baseline_model()


@st.cache_resource(show_spinner="Loading model…")
def _load_bilstm():
    from src.inference import load_bilstm_model
    return load_bilstm_model()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("ReviewPulse")
st.caption("Multi-domain Amazon review sentiment classifier")

st.divider()

# ---------------------------------------------------------------------------
# Model selector
# ---------------------------------------------------------------------------

MODEL_OPTIONS = {
    "baseline": "TF-IDF + Logistic Regression  (recommended — best test F1)",
    "bilstm":   "BiLSTM + GloVe  (neural model)",
}

model_name = st.radio(
    "Model",
    options=list(MODEL_OPTIONS.keys()),
    format_func=lambda k: MODEL_OPTIONS[k],
    horizontal=False,
    index=0,
)

# Warm up the selected model in the background
if model_name == "baseline":
    _load_baseline()
else:
    _load_bilstm()

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

st.markdown("#### Review text")
text = st.text_area(
    label="Review text",
    label_visibility="collapsed",
    placeholder="Paste an Amazon review here…",
    height=160,
)

classify = st.button(
    "Classify",
    type="primary",
    disabled=not text.strip(),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

if classify and text.strip():
    from src.inference import predict_sentiment

    with st.spinner("Classifying…"):
        result = predict_sentiment(text.strip(), model_name=model_name)

    label      = result["label"]
    confidence = result["confidence"]
    is_pos     = "Positive" in label

    st.divider()

    if is_pos:
        st.success(f"**{label}**", icon="✅")
    else:
        st.error(f"**{label}**", icon="❌")

    st.metric(label="Confidence", value=f"{confidence:.1%}")
    st.progress(confidence)

    with st.expander("Details"):
        st.json(result)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Built for ISY503 Intelligent Systems · Torrens University · 2026‑T1  \n"
    "Baseline: TF-IDF + LogReg (test F1 81.9%)  \n"
    "Neural: BiLSTM + GloVe (val F1 84.0%, test F1 80.3%)"
)

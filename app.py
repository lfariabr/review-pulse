"""ReviewPulse — Streamlit sentiment inference UI.

Usage:
    streamlit run app.py
"""

from PIL import Image
import streamlit as st

from src.app_service import (
    DISTILBERT_UNAVAILABLE_MSG,
    MODEL_OPTIONS,
    is_distilbert_available,
    warm_up_model,
)
from src.config import MODEL_DISTILBERT

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
# Sidebar — logo + spacer
# ---------------------------------------------------------------------------

st.logo("logo-icon.png", link=None)   # icon in the top-left chrome
st.sidebar.image("logo.jpeg", width='content')

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("ReviewPulse")
st.caption("Multi-domain Amazon review sentiment classifier")

st.divider()

# ---------------------------------------------------------------------------
# Model selector
# ---------------------------------------------------------------------------

model_name = st.radio(
    "Model",
    options=list(MODEL_OPTIONS.keys()),
    format_func=lambda k: MODEL_OPTIONS[k],
    horizontal=False,
    index=0,
)

# Warm up the selected model; check DistilBERT loaded successfully.
_distilbert_available = True
if model_name == MODEL_DISTILBERT:
    if not is_distilbert_available():
        _distilbert_available = False
        st.warning(DISTILBERT_UNAVAILABLE_MSG, icon="⚠️")
else:
    warm_up_model(model_name)

# ---------------------------------------------------------------------------
# Sample reviews
# ---------------------------------------------------------------------------

from src.utils.samples import get_random_sample

# Session state key for the text area value
if "review_text" not in st.session_state:
    st.session_state["review_text"] = ""


def _load_random_sample():
    """Load a random sample into session state."""
    current = st.session_state.get("review_text", "")
    st.session_state["review_text"] = get_random_sample(current)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

label_col, btn_col = st.columns([5, 1])
with label_col:
    st.markdown("#### Review text")
with btn_col:
    st.button("💡 Generate", help="Load a random sample review", on_click=_load_random_sample)

text = st.text_area(
    label="Review text",
    label_visibility="collapsed",
    placeholder="Paste any review text here and we'll tell you if it's positive or negative…",
    height=160,
    key="review_text",
)

classify = st.button(
    "Classify",
    type="primary",
    disabled=not text.strip(),
    width='content',
)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

if classify and text.strip() and _distilbert_available:
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
    "Neural: BiLSTM + GloVe (val F1 84.0%, test F1 80.3%)  \n"
    "Transformer: DistilBERT (val F1 87.8%, test F1 88.6%)"
)

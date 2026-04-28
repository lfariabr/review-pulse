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
# Sidebar — logo + spacer
# ---------------------------------------------------------------------------

st.logo("logo-icon.png", link=None)   # icon in the top-left chrome
st.sidebar.image("logo.jpeg", use_container_width=True)

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


@st.cache_resource(show_spinner="Loading model…")
def _load_distilbert():
    try:
        from src.inference import load_distilbert_model
        return load_distilbert_model()
    except (ImportError, FileNotFoundError, RuntimeError):
        return None


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
    "distilbert": "DistilBERT_base_uncased  (Hugging Face model)",
}

MODEL_LOADERS = {
    "baseline": _load_baseline,
    "bilstm": _load_bilstm,
    "distilbert": _load_distilbert,
}

model_name = st.radio(
    "Model",
    options=list(MODEL_OPTIONS.keys()),
    format_func=lambda k: MODEL_OPTIONS[k],
    horizontal=False,
    index=0,
)

# Warm up the selected model; check DistilBERT loaded successfully.
_distilbert_available = True
if model_name == "distilbert":
    if _load_distilbert() is None:
        _distilbert_available = False
        st.warning(
            "DistilBERT is currently unavailable — the checkpoint or `transformers` "
            "dependency could not be loaded. Select **Baseline** or **BiLSTM** to continue.",
            icon="⚠️",
        )
else:
    MODEL_LOADERS[model_name]()

# ---------------------------------------------------------------------------
# Sample reviews
# ---------------------------------------------------------------------------

_SAMPLES = [
    # positive
    "This blender is incredible — smoothies in under 30 seconds, easy to clean, and still going strong after six months of daily use.",
    "Absolutely love this book. The writing is sharp, the characters feel real, and I stayed up until 2 AM to finish it. Highly recommended.",
    "Perfect headphones for the price. Sound quality rivals sets twice the cost, and the battery easily lasts two full days.",
    "The kitchen knife set exceeded my expectations. Razor sharp out of the box, balanced grip, and the block keeps everything organised.",
    "Bought this for my daughter's birthday and she hasn't put it down. Great educational toy that actually holds a child's attention.",
    # negative
    "Arrived with a cracked screen and the seller took three weeks to respond. Complete waste of money — avoid.",
    "The straps broke on the second use. Cheap stitching, flimsy buckles. Returned immediately and won't be buying from this brand again.",
    "Sound cuts out every few minutes. I thought it was a pairing issue but the replacement unit had the same problem. Terrible quality control.",
    "This DVD player skips on every disc I try. The remote is unresponsive half the time. Returned after one day.",
    "Poorly written instructions, missing hardware, and customer support just copy-pasted the same unhelpful reply three times. Deeply frustrating.",
]

# Session state key for the text area value
if "review_text" not in st.session_state:
    st.session_state["review_text"] = ""


def _load_random_sample():
    import random
    current = st.session_state.get("review_text", "")
    candidates = [s for s in _SAMPLES if s != current]
    st.session_state["review_text"] = random.choice(candidates)


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
    use_container_width=True,
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
    "Transformer: DistilBERT (val F1 87%, test F1 88%)"
)

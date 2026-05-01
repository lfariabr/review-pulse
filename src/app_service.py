"""Streamlit app service layer for ReviewPulse.

Provides cached model loading and DistilBERT availability helpers so that
app.py stays focused on layout, input, and result display.
"""

import streamlit as st

from src.config import MODEL_BASELINE, MODEL_BILSTM, MODEL_DISTILBERT

MODEL_OPTIONS: dict[str, str] = {
    MODEL_BASELINE:   "TF-IDF + Logistic Regression  (recommended — best test F1)",
    MODEL_BILSTM:     "BiLSTM + GloVe  (neural model)",
    MODEL_DISTILBERT: "DistilBERT_base_uncased  (Hugging Face model)",
}

DISTILBERT_UNAVAILABLE_MSG = (
    "DistilBERT is currently unavailable — the checkpoint or `transformers` "
    "dependency could not be loaded. Select **Baseline** or **BiLSTM** to continue."
)


@st.cache_resource(show_spinner="Loading model…")
def load_baseline():
    """Load and cache the TF-IDF + LogReg baseline."""
    from src.inference import load_baseline_model
    return load_baseline_model()


@st.cache_resource(show_spinner="Loading model…")
def load_bilstm():
    """Load and cache the BiLSTM + GloVe model."""
    from src.inference import load_bilstm_model
    return load_bilstm_model()


@st.cache_resource(show_spinner="Loading model…")
def load_distilbert():
    """Load and cache the DistilBERT model, returning None on any failure."""
    try:
        from src.inference import load_distilbert_model
        return load_distilbert_model()
    except (ImportError, FileNotFoundError, RuntimeError):
        return None


_MODEL_LOADERS = {
    MODEL_BASELINE:   load_baseline,
    MODEL_BILSTM:     load_bilstm,
    MODEL_DISTILBERT: load_distilbert,
}


def warm_up_model(model_name: str) -> bool:
    """Trigger cached loading for *model_name*. Returns False if unavailable."""
    loader = _MODEL_LOADERS.get(model_name)
    if loader is None:
        return False
    return loader() is not None


def is_distilbert_available() -> bool:
    """Return True when the DistilBERT checkpoint and dependencies load successfully."""
    return load_distilbert() is not None

"""Shared constants for ReviewPulse.

Single source of truth for model names, artifact paths, and prediction
threshold. Import from here rather than redefining in each module.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

BASELINE_PATH          = OUTPUTS_DIR / "baseline.joblib"
BILSTM_CHECKPOINT_PATH = OUTPUTS_DIR / "bilstm.pt"
VOCAB_PATH             = OUTPUTS_DIR / "vocab.json"
DISTILBERT_PATH        = OUTPUTS_DIR / "distilbert.pt"

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

MODEL_BASELINE   = "baseline"
MODEL_BILSTM     = "bilstm"
MODEL_DISTILBERT = "distilbert"

ALL_MODELS = (MODEL_BASELINE, MODEL_BILSTM, MODEL_DISTILBERT)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

PRED_THRESHOLD = 0.5

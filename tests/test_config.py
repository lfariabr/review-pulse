# .venv/bin/pytest tests/test_config.py -v

"""Contract tests for src/config.py.

These tests document what downstream modules depend on. If a constant
changes name, type, or value the failure here pinpoints the breakage
before it reaches inference, evaluate, or the app.
"""

import subprocess
import sys
from pathlib import Path

from src.config import (
    ALL_MODELS,
    BASELINE_PATH,
    BILSTM_CHECKPOINT_PATH,
    DISTILBERT_PATH,
    MODEL_BASELINE,
    MODEL_BILSTM,
    MODEL_DISTILBERT,
    OUTPUTS_DIR,
    PRED_THRESHOLD,
    VOCAB_PATH,
)

_PROJECT_ROOT = str(Path(__file__).parent.parent)


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

def test_all_path_constants_are_path_instances():
    for path in (OUTPUTS_DIR, BASELINE_PATH, BILSTM_CHECKPOINT_PATH, VOCAB_PATH, DISTILBERT_PATH):
        assert isinstance(path, Path)


def test_outputs_dir_name():
    assert OUTPUTS_DIR.name == "outputs"


def test_all_artifact_paths_are_under_outputs_dir():
    for path in (BASELINE_PATH, BILSTM_CHECKPOINT_PATH, VOCAB_PATH, DISTILBERT_PATH):
        assert path.parent == OUTPUTS_DIR


def test_baseline_path_extension():
    assert BASELINE_PATH.suffix == ".joblib"


def test_bilstm_path_extension():
    assert BILSTM_CHECKPOINT_PATH.suffix == ".pt"


def test_vocab_path_extension():
    assert VOCAB_PATH.suffix == ".json"


def test_distilbert_path_extension():
    assert DISTILBERT_PATH.suffix == ".pt"


# ---------------------------------------------------------------------------
# Model name constants
# ---------------------------------------------------------------------------

def test_model_name_constants_are_nonempty_strings():
    for name in (MODEL_BASELINE, MODEL_BILSTM, MODEL_DISTILBERT):
        assert isinstance(name, str) and name.strip()


def test_model_names_are_distinct():
    assert len({MODEL_BASELINE, MODEL_BILSTM, MODEL_DISTILBERT}) == 3


def test_all_models_contains_each_constant():
    assert MODEL_BASELINE   in ALL_MODELS
    assert MODEL_BILSTM     in ALL_MODELS
    assert MODEL_DISTILBERT in ALL_MODELS


def test_all_models_length():
    assert len(ALL_MODELS) == 3


def test_all_models_no_duplicates():
    assert len(set(ALL_MODELS)) == len(ALL_MODELS)


# ---------------------------------------------------------------------------
# Prediction threshold
# ---------------------------------------------------------------------------

def test_pred_threshold_is_float():
    assert isinstance(PRED_THRESHOLD, float)


def test_pred_threshold_in_open_unit_interval():
    assert 0.0 < PRED_THRESHOLD < 1.0


# ---------------------------------------------------------------------------
# Module boundary — config must import nothing from src.*
# ---------------------------------------------------------------------------

def test_config_imports_no_other_src_modules():
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import src.config; "
         "bad = [m for m in sys.modules "
         "       if m.startswith('src.') and m not in ('src.config', 'src')]; "
         "assert not bad, f'src.config pulled in: {bad}'"],
        capture_output=True, text=True, cwd=_PROJECT_ROOT,
    )
    assert result.returncode == 0, (
        f"src.config has unexpected imports:\n{result.stderr}"
    )

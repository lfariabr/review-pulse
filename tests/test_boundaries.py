# .venv/bin/pytest tests/test_boundaries.py -v

"""Module boundary tests.

Each test verifies that a module does not pull in code from a concern
it must not own. Failures here indicate coupling that will make the
refactor series unsafe.
"""

import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).parent.parent)


def _assert_no_imports(module: str, forbidden: list[str], stub_streamlit: bool = False) -> None:
    """Subprocess-import *module* and fail if any *forbidden* prefix appears in sys.modules."""
    preamble = ""
    if stub_streamlit:
        preamble = (
            "import sys, types; "
            "stub = types.ModuleType('streamlit'); "
            "stub.cache_resource = lambda *a, **kw: (a[0] if a and callable(a[0]) else lambda fn: fn); "
            "sys.modules['streamlit'] = stub; "
        )
    check = (
        f"bad = [m for m in sys.modules "
        f"       if any(m == r or m.startswith(r + '.') for r in {forbidden!r})]; "
        f"assert not bad, f'{{bad}}'"
    )
    result = subprocess.run(
        [sys.executable, "-c", f"{preamble}import sys; import {module}; {check}"],
        capture_output=True, text=True, cwd=_PROJECT_ROOT,
    )
    assert result.returncode == 0, (
        f"{module!r} imported forbidden module(s):\n{result.stderr or result.stdout}"
    )


# ---------------------------------------------------------------------------
# src.config — dependency root; must import nothing from the rest of src
# ---------------------------------------------------------------------------

def test_config_does_not_import_any_src_module():
    _assert_no_imports(
        "src.config",
        forbidden=["src.inference", "src.training", "src.evaluation",
                   "src.models", "src.tokenization",
                   "src.data.parser", "src.data.preprocess"],
    )


# ---------------------------------------------------------------------------
# src.inference — prediction layer; must not touch training or evaluation
# ---------------------------------------------------------------------------

def test_inference_does_not_import_train():
    _assert_no_imports("src.inference", forbidden=["src.training"])


def test_inference_does_not_import_evaluate():
    _assert_no_imports("src.inference", forbidden=["src.evaluation"])


def test_inference_does_not_import_parser():
    _assert_no_imports("src.inference", forbidden=["src.parser", "src.data.parser"])


# ---------------------------------------------------------------------------
# src.app.service — presentation service; must not touch training
# ---------------------------------------------------------------------------

def test_app_service_does_not_import_train():
    _assert_no_imports(
        "src.app.service",
        forbidden=["src.training"],
        stub_streamlit=True,
    )


def test_app_service_does_not_import_evaluate():
    _assert_no_imports(
        "src.app.service",
        forbidden=["src.evaluation"],
        stub_streamlit=True,
    )


def test_app_service_does_not_import_parser():
    _assert_no_imports(
        "src.app.service",
        forbidden=["src.parser", "src.data.parser"],
        stub_streamlit=True,
    )

# .venv/bin/pytest tests/test_app_service.py -v

"""Tests for src/app_service.py."""

import sys
import types
import pytest

from src.config import MODEL_BASELINE, MODEL_BILSTM, MODEL_DISTILBERT


# ---------------------------------------------------------------------------
# Streamlit stub — prevents real Streamlit from running during tests
# ---------------------------------------------------------------------------

def _make_st_stub():
    """Return a minimal streamlit stub with cache_resource as a no-op decorator."""
    stub = types.ModuleType("streamlit")
    stub.cache_resource = lambda *args, **kwargs: (
        (lambda fn: fn) if not args else args[0]
        if callable(args[0]) else (lambda fn: fn)
    )
    return stub


@pytest.fixture(autouse=True)
def _patch_streamlit(monkeypatch):
    """Replace streamlit with a minimal stub before each test."""
    stub = _make_st_stub()
    monkeypatch.setitem(sys.modules, "streamlit", stub)
    # Force reload so app_service picks up the stub
    if "src.app_service" in sys.modules:
        monkeypatch.delitem(sys.modules, "src.app_service")
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_service():
    import importlib
    return importlib.import_module("src.app_service")


# ---------------------------------------------------------------------------
# MODEL_OPTIONS
# ---------------------------------------------------------------------------

def test_model_options_contains_all_three_models():
    svc = _import_service()
    assert MODEL_BASELINE   in svc.MODEL_OPTIONS
    assert MODEL_BILSTM     in svc.MODEL_OPTIONS
    assert MODEL_DISTILBERT in svc.MODEL_OPTIONS


def test_model_options_values_are_nonempty_strings():
    svc = _import_service()
    for label in svc.MODEL_OPTIONS.values():
        assert isinstance(label, str) and label.strip()


def test_distilbert_unavailable_msg_is_nonempty():
    svc = _import_service()
    assert isinstance(svc.DISTILBERT_UNAVAILABLE_MSG, str)
    assert svc.DISTILBERT_UNAVAILABLE_MSG.strip()


# ---------------------------------------------------------------------------
# is_distilbert_available
# ---------------------------------------------------------------------------

def test_is_distilbert_available_true_when_loader_returns_object(monkeypatch):
    svc = _import_service()
    monkeypatch.setattr(svc, "load_distilbert", lambda: object())
    assert svc.is_distilbert_available() is True


def test_is_distilbert_available_false_when_loader_returns_none(monkeypatch):
    svc = _import_service()
    monkeypatch.setattr(svc, "load_distilbert", lambda: None)
    assert svc.is_distilbert_available() is False


# ---------------------------------------------------------------------------
# warm_up_model
# ---------------------------------------------------------------------------

def test_warm_up_model_returns_true_for_successful_load(monkeypatch):
    svc = _import_service()
    monkeypatch.setattr(svc, "load_baseline", lambda: object())
    assert svc.warm_up_model(MODEL_BASELINE) is True


def test_warm_up_model_returns_false_when_loader_returns_none(monkeypatch):
    svc = _import_service()
    monkeypatch.setattr(svc, "load_distilbert", lambda: None)
    monkeypatch.setattr(svc, "_MODEL_LOADERS", {MODEL_DISTILBERT: svc.load_distilbert})
    assert svc.warm_up_model(MODEL_DISTILBERT) is False


def test_warm_up_model_returns_false_for_unknown_model_name(monkeypatch):
    svc = _import_service()
    assert svc.warm_up_model("nonexistent_model") is False


def test_warm_up_model_covers_all_registered_models(monkeypatch):
    svc = _import_service()
    called = []

    def _fake_loader():
        called.append(True)
        return object()

    monkeypatch.setattr(svc, "_MODEL_LOADERS", {
        MODEL_BASELINE:   _fake_loader,
        MODEL_BILSTM:     _fake_loader,
        MODEL_DISTILBERT: _fake_loader,
    })

    for name in (MODEL_BASELINE, MODEL_BILSTM, MODEL_DISTILBERT):
        result = svc.warm_up_model(name)
        assert result is True

    assert len(called) == 3


# ---------------------------------------------------------------------------
# load_distilbert swallows expected errors (patches the real loader)
# ---------------------------------------------------------------------------

def _make_raiser(exc_type, msg="error"):
    def _raise():
        raise exc_type(msg)
    return _raise


@pytest.mark.parametrize("exc_type,msg", [
    (ImportError,       "no transformers"),
    (FileNotFoundError, "no checkpoint"),
    (RuntimeError,      "corrupt checkpoint"),
    (OSError,           "hf download failed"),
])
def test_load_distilbert_swallows_exception(monkeypatch, exc_type, msg):
    """load_distilbert() must return None (not raise) for all expected error types."""
    import src.inference as inference_module
    monkeypatch.setattr(inference_module, "load_distilbert_model", _make_raiser(exc_type, msg))
    svc = _import_service()
    assert svc.load_distilbert() is None

"""Compatibility package for the existing ``src.inference`` module."""

from pathlib import Path

_legacy_module_path = Path(__file__).resolve().parent.parent / "inference.py"
exec(compile(_legacy_module_path.read_text(), str(_legacy_module_path), "exec"), globals())

del Path, _legacy_module_path

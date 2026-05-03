# .venv/bin/pytest tests/test_backward_compat.py -v

"""Backward compatibility tests.

Verify that old import paths (src.parser, src.preprocess, src.features)
still work via re-export wrappers, even after internal modules moved to src.data.

These tests are not meant to be comprehensive; they verify that the wrapper
re-exports are alive and functioning. Implementation tests are in the new paths.
"""

import pytest


def test_can_import_from_old_parser_path():
    """Verify src.parser re-export wrapper is importable."""
    from src.parser import load_all_domains, parse_review_file
    assert callable(load_all_domains)
    assert callable(parse_review_file)


def test_can_import_from_old_preprocess_path():
    """Verify src.preprocess re-export wrapper is importable."""
    from src.preprocess import audit_labels, clean_text, split_data
    assert callable(audit_labels)
    assert callable(clean_text)
    assert callable(split_data)


def test_can_import_from_old_features_path():
    """Verify src.features re-export wrapper is importable."""
    from src.features import class_balance, domain_balance
    assert callable(class_balance)
    assert callable(domain_balance)


def test_old_parser_imports_delegate_to_new_path():
    """Verify that old imports actually delegate to src.data.parser."""
    from src.parser import load_all_domains as old_load
    from src.data.parser import load_all_domains as new_load
    # Both should be the same function object
    assert old_load is new_load


def test_old_preprocess_imports_delegate_to_new_path():
    """Verify that old imports actually delegate to src.data.preprocess."""
    from src.preprocess import clean_text as old_clean
    from src.data.preprocess import clean_text as new_clean
    # Both should be the same function object
    assert old_clean is new_clean


def test_old_features_imports_delegate_to_new_path():
    """Verify that old imports actually delegate to src.data.features."""
    from src.features import class_balance as old_balance
    from src.data.features import class_balance as new_balance
    # Both should be the same function object
    assert old_balance is new_balance

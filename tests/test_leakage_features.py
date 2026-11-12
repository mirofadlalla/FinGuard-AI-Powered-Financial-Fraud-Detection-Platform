"""Tests protecting against feature and label leakage."""

import pytest


def test_isfraud_not_in_model_features(meta):
    """isFraud must NEVER appear in model feature names."""
    names = meta["feature_names"]
    leakers = [n for n in names if "fraud" in n.lower() or n == "isFraud"]
    assert leakers == [], f"Label leakage found: {leakers}"


def test_transaction_id_not_in_model_features(meta):
    """TransactionID is a row identifier, not a predictive signal."""
    names = meta["feature_names"]
    assert "TransactionID" not in names, "TransactionID leaked into features"


def test_isfraud_not_in_preprocessor_num_cols(preprocessor):
    """isFraud must not appear in preprocessor's numeric columns."""
    leakers = [c for c in preprocessor.num_cols if "fraud" in c.lower()]
    assert leakers == [], f"isFraud in num_cols: {leakers}"


def test_isfraud_not_in_preprocessor_freq_cols(preprocessor):
    """isFraud must not appear in freq_cols."""
    leakers = [c for c in preprocessor.freq_cols if "fraud" in c.lower()]
    assert leakers == [], f"isFraud in freq_cols: {leakers}"

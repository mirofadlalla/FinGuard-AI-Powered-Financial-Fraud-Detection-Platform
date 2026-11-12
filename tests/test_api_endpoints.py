"""Tests verifying API prediction consistency, multi-transaction sanity, and serving parity."""

import numpy as np
import pytest
import xgboost as xgb


def test_model_loading_and_config(booster, meta):
    """Loaded XGBoost booster must match meta feature count."""
    assert booster.num_features() == len(meta["feature_names"])


def test_fraud_probability_range(booster, test_data):
    """Predicted fraud probability must strictly lie within [0.0, 1.0]."""
    X_val = test_data["X_val"][:100]
    dmat = xgb.DMatrix(X_val, missing=np.nan)
    preds = booster.predict(dmat)
    assert (preds >= 0.0).all()
    assert (preds <= 1.0).all()
    assert not np.isnan(preds).any()


def test_multi_transaction_sanity(booster, test_data, meta):
    """Model must NOT flag everything as fraud; majority of payments should be approved."""
    X_test = test_data["X_test"][:1000]
    dmat = xgb.DMatrix(X_test, missing=np.nan)
    probs = booster.predict(dmat)
    review_thr = meta["thresholds"]["review"]
    flagged = probs >= review_thr
    flag_rate = flagged.mean()
    # Fraud rate in IEEE-CIS is ~3.5%; flag rate should be reasonable (< 25%)
    assert flag_rate < 0.25, f"Flag rate {flag_rate:.2%} is unrealistically high"

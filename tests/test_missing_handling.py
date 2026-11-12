"""Tests verifying robust missing-value handling across the pipeline and model."""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb


def _make_empty_row(preprocessor):
    all_cols = (
        list(getattr(preprocessor, "freq_cols", []))
        + list(getattr(preprocessor, "na_cols", []))
        + list(getattr(preprocessor, "num_cols", []))
        + list(getattr(preprocessor, "v_block_reps", []))
    )
    return pd.DataFrame([{c: np.nan for c in set(all_cols)}])


def test_all_nan_row_produces_finite_features(preprocessor):
    """An all-NaN transaction must produce a finite or NaN array, never crash."""
    empty_df = _make_empty_row(preprocessor)
    x = preprocessor.transform(empty_df).values[0]
    assert len(x) == len(preprocessor.feature_names)
    assert not np.isinf(x).any(), "Infinite values generated from NaN input"


def test_model_predicts_on_all_nan_row(booster, preprocessor):
    """XGBoost must natively score an all-missing row without error."""
    empty_df = _make_empty_row(preprocessor)
    x = preprocessor.transform(empty_df).values[0]
    dmat = xgb.DMatrix(x.reshape(1, -1), missing=np.nan)
    prob = float(booster.predict(dmat)[0])
    assert 0.0 <= prob <= 1.0
    assert not np.isnan(prob)


def test_missing_indicator_features_present(meta):
    """_na suffix features must exist for columns with high missingness in training."""
    na_cols = [n for n in meta["feature_names"] if n.endswith("_na")]
    assert len(na_cols) > 0, "No _na indicator columns found"

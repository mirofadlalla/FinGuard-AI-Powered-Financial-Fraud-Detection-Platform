"""Tests verifying categorical frequency encoding and unseen value mapping."""

import numpy as np
import pandas as pd
import pytest


def test_freq_cols_have_freq_suffix(preprocessor):
    """Every frequency column must be mapped to c_freq in feature names."""
    for c in preprocessor.freq_cols:
        assert f"{c}_freq" in preprocessor.feature_names, f"{c}_freq missing from feature_names"


def test_unseen_category_maps_to_zero(preprocessor):
    """An unseen category value must map to frequency 0.0 (rare), not crash."""
    all_cols = (
        list(getattr(preprocessor, "freq_cols", []))
        + list(getattr(preprocessor, "na_cols", []))
        + list(getattr(preprocessor, "num_cols", []))
        + list(getattr(preprocessor, "v_block_reps", []))
    )
    row = {c: np.nan for c in set(all_cols)}
    for c in preprocessor.freq_cols:
        row[c] = "__completely_unseen_category_xyz__"
    df = pd.DataFrame([row])
    x = preprocessor.transform(df)
    for c in preprocessor.freq_cols:
        val = x[f"{c}_freq"].iloc[0]
        assert not np.isnan(val) and not np.isinf(val)


def test_freq_maps_contain_series(preprocessor):
    """freq_maps dictionary must hold non-empty frequency Series for categories."""
    assert len(preprocessor.freq_maps) == len(preprocessor.freq_cols)
    for col, fm in preprocessor.freq_maps.items():
        assert isinstance(fm, pd.Series)

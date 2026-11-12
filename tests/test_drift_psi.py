"""Tests verifying Population Stability Index (PSI) drift monitoring."""

import numpy as np
import pytest

from monitoring.drift import (
    calculate_psi,
    feature_psi,
    make_reference,
    make_reference_column,
    psi_level,
)


def test_psi_is_small_for_same_data_and_big_for_shifted_data():
    rng = np.random.default_rng(1)
    train = rng.normal(0, 1, 20000)
    ref = make_reference_column(train)
    assert calculate_psi(ref, rng.normal(0, 1, 20000)) < 0.1
    assert calculate_psi(ref, rng.normal(1.5, 1, 20000)) > 0.25


def test_psi_stable_for_identical_data():
    rng = np.random.default_rng(42)
    data = rng.exponential(scale=2.0, size=5000)
    ref = make_reference_column(data)
    assert calculate_psi(ref, data) < 0.01


def test_feature_psi_returns_all_features():
    rng = np.random.default_rng(7)
    names = ["a", "b", "c"]
    X = rng.normal(size=(1000, 3))
    ref = make_reference(X, names)
    result = feature_psi(ref, X, names)
    assert set(result.keys()) == set(names)
    for v in result.values():
        assert v >= 0


def test_psi_level_boundaries():
    assert psi_level(0.05) == "stable"
    assert psi_level(0.15) == "small shift"
    assert psi_level(0.30) == "BIG SHIFT"


def test_make_reference_column_handles_all_nan():
    ref = make_reference_column(np.array([np.nan, np.nan, np.nan]))
    assert isinstance(ref["edges"], list)
    assert isinstance(ref["props"], list)

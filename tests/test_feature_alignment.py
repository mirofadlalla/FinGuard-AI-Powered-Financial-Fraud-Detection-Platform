"""Tests verifying exact feature count, names, and ordering between model and preprocessor."""

import pytest


def test_model_and_preprocessor_same_feature_count(meta, preprocessor):
    """Model and preprocessor must expose exactly the same number of features."""
    assert len(meta["feature_names"]) == len(preprocessor.feature_names), (
        f"Model has {len(meta['feature_names'])} features, "
        f"preprocessor has {len(preprocessor.feature_names)}"
    )


def test_model_and_preprocessor_same_feature_names(meta, preprocessor):
    """Every feature expected by the model must be produced by the preprocessor."""
    model_set = set(meta["feature_names"])
    prep_set = set(preprocessor.feature_names)
    assert model_set == prep_set, f"Feature mismatch: diff={model_set ^ prep_set}"


def test_model_and_preprocessor_same_feature_order(meta, preprocessor):
    """Feature order matters for matrix feeding; order must match 1-to-1."""
    assert meta["feature_names"] == preprocessor.feature_names, (
        "Feature ordering differs between meta.json and preprocessor.pkl"
    )


def test_preprocessor_mean_std_shape(meta, preprocessor):
    """Preprocessor mean and std must match feature count and contain no zeros in std."""
    n = len(meta["feature_names"])
    assert len(preprocessor.mean) == n
    assert len(preprocessor.std) == n
    assert (preprocessor.std == 0).sum() == 0, "Zero standard deviation found"

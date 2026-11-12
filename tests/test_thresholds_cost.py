"""Tests verifying decision thresholds, cost calculations, and business savings."""

import numpy as np
import pytest

from evaluation.metrics import (
    best_f1_threshold,
    cost_threshold,
    no_model_cost,
    precision_threshold,
    report,
    summarize,
)


def _make_scores(seed=0):
    rng = np.random.default_rng(seed)
    n = 2000
    y = rng.integers(0, 2, n).astype("int8")
    p = np.where(y == 1, rng.uniform(0.4, 1.0, n), rng.uniform(0.0, 0.6, n))
    amount = rng.uniform(10, 500, n).astype("float32")
    return y, p, amount


def test_no_model_cost_equals_sum_of_fraud_amounts():
    y, p, amount = _make_scores()
    expected = float(amount[y == 1].sum())
    assert np.isclose(no_model_cost(y, amount), expected)


def test_best_f1_threshold_gives_valid_range():
    y, p, _ = _make_scores()
    thr = best_f1_threshold(y, p)
    assert 0.0 <= thr <= 1.0


def test_cost_threshold_is_better_than_no_model():
    y, p, amount = _make_scores()
    thr, cost = cost_threshold(y, p, amount)
    assert cost < no_model_cost(y, amount)


def test_precision_threshold_target_met():
    y, p, amount = _make_scores()
    thr = precision_threshold(y, p, target=0.5)
    flagged = p >= thr
    if flagged.sum() > 0:
        prec = float(y[flagged].sum()) / int(flagged.sum())
        assert prec >= 0.5 or thr == 1.0


def test_report_returns_expected_keys():
    y, p, amount = _make_scores()
    r = report(y, p, amount, thr=0.5, name="test")
    for key in ["model", "PR-AUC", "ROC-AUC", "precision", "recall", "F1", "FP", "FN", "cost", "saved%"]:
        assert key in r

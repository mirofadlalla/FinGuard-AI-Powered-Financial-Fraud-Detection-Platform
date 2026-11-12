"""Tests protecting against temporal data leakage (looking at future rows)."""

import numpy as np
import pandas as pd
import pytest

from features.ieee import add_past_aggs, rolling_count


def test_aggregates_only_use_the_past():
    """Verify cumulative aggregates on client groups only look at strictly prior rows."""
    df = pd.DataFrame({
        "k": ["a", "b", "a", "a", "b", "a", "a"],
        "TransactionAmt": [10, 5, 20, 30, 7, np.nan, 50.0]
    })
    add_past_aggs(df, "k", "g", ["TransactionAmt"], ["TransactionAmt"])
    for i in range(len(df)):
        before = df.iloc[:i]
        past = before[before.k == df.k[i]].TransactionAmt.dropna()
        expected = past.mean() if len(past) else np.nan
        assert np.isclose(df.g_agg_TransactionAmt_mean[i], expected, equal_nan=True)


def test_rolling_count_matches_brute_force():
    """Verify vectorized glued-key rolling count matches brute force implementation."""
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 30, 1500)
    t = np.sort(rng.integers(0, 2_000_000, 1500)).astype("int64")
    fast = rolling_count(codes, t, 86400)
    brute = [((codes == codes[i]) & (t < t[i]) & (t >= t[i] - 86400)).sum() for i in range(len(t))]
    assert np.array_equal(fast.astype(int), brute)


def test_rolling_count_overflow_guard():
    """rolling_count should raise OverflowError when glued key code would overflow int64."""
    BIG = 10 ** 9
    max_safe_code = 9_200_000_000_000_000_000 // BIG
    bad_code = max_safe_code + 1
    codes = np.array([bad_code, bad_code])
    t = np.array([0, 1], dtype="int64")
    with pytest.raises(OverflowError):
        rolling_count(codes, t, 10)

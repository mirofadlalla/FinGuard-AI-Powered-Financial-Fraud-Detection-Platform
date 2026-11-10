"""Tests that protect us from the worst bug in fraud models: looking at the future."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finguard.assistant import find_transaction_id
from finguard.drift import make_reference_column, psi
from finguard.ieee import add_past_aggs, rolling_count


def test_aggregates_only_use_the_past():
    df = pd.DataFrame({"k": ["a", "b", "a", "a", "b", "a", "a"],
                       "TransactionAmt": [10, 5, 20, 30, 7, np.nan, 50.0]})
    add_past_aggs(df, "k", "g", ["TransactionAmt"], ["TransactionAmt"])
    for i in range(len(df)):
        before = df.iloc[:i]
        past = before[before.k == df.k[i]].TransactionAmt.dropna()
        expected = past.mean() if len(past) else np.nan
        assert np.isclose(df.g_agg_TransactionAmt_mean[i], expected, equal_nan=True)


def test_rolling_count_matches_brute_force():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 30, 1500)
    t = np.sort(rng.integers(0, 2_000_000, 1500)).astype("int64")
    fast = rolling_count(codes, t, 86400)
    brute = [((codes == codes[i]) & (t < t[i]) & (t >= t[i] - 86400)).sum() for i in range(len(t))]
    assert np.array_equal(fast.astype(int), brute)


def test_psi_is_small_for_same_data_and_big_for_shifted_data():
    rng = np.random.default_rng(1)
    train = rng.normal(0, 1, 20000)
    ref = make_reference_column(train)
    assert psi(ref, rng.normal(0, 1, 20000)) < 0.1
    assert psi(ref, rng.normal(1.5, 1, 20000)) > 0.25


def test_assistant_finds_the_transaction_id():
    assert find_transaction_id("Why was transaction #3487632 flagged?") == 3487632
    assert find_transaction_id("hello") is None

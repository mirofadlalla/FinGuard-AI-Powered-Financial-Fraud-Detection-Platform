"""Tests that protect us from the worst bug in fraud models: looking at the future.

Also covers: PSI drift, business metrics, database layer, Cache, Feedback, LiveMonitor,
quality_stats, and model architecture.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finguard.assistant import find_transaction_id
from finguard.drift import make_reference_column, psi, feature_psi, make_reference, level
from finguard.ieee import add_past_aggs, rolling_count
from finguard.metrics import (
    best_f1_threshold,
    cost_threshold,
    precision_threshold,
    report,
    no_model_cost,
    summarize,
)
from finguard.db import Cache, make_engine, save_decision, get_decision, recent_stats, save_feedback, quality_stats

# Skip marker for torch-dependent tests (model architecture).
# torch itself is NOT imported here — only inside tests that need it.
_no_torch = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torch"),
    reason="torch not installed",
)


# ─────────────────────────── data-leakage tests ────────────────────────────

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


def test_rolling_count_overflow_guard():
    """rolling_count should raise OverflowError when the glued key would overflow int64."""
    # craft codes that exceed 9.2e18 / BIG
    BIG = 10 ** 9
    max_safe_code = 9_200_000_000_000_000_000 // BIG
    bad_code = max_safe_code + 1
    codes = np.array([bad_code, bad_code])
    t = np.array([0, 1], dtype="int64")
    with pytest.raises(OverflowError):
        rolling_count(codes, t, 10)


# ─────────────────────────── drift / PSI tests ─────────────────────────────

def test_psi_is_small_for_same_data_and_big_for_shifted_data():
    rng = np.random.default_rng(1)
    train = rng.normal(0, 1, 20000)
    ref = make_reference_column(train)
    assert psi(ref, rng.normal(0, 1, 20000)) < 0.1
    assert psi(ref, rng.normal(1.5, 1, 20000)) > 0.25


def test_psi_stable_for_identical_data():
    rng = np.random.default_rng(42)
    data = rng.exponential(scale=2.0, size=5000)
    ref = make_reference_column(data)
    assert psi(ref, data) < 0.01   # same data → PSI ≈ 0


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
    assert level(0.05) == "stable"
    assert level(0.15) == "small shift"
    assert level(0.30) == "BIG SHIFT"


def test_make_reference_column_handles_all_nan():
    ref = make_reference_column(np.array([np.nan, np.nan, np.nan]))
    # Should not raise; edges will be empty
    assert isinstance(ref["edges"], list)
    assert isinstance(ref["props"], list)


# ─────────────────────────── metrics tests ─────────────────────────────────

def _make_scores(seed=0):
    rng = np.random.default_rng(seed)
    n = 2000
    y = rng.integers(0, 2, n).astype("int8")
    # fraud scores slightly higher than legit
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
    # Using that threshold, precision should be >= 0.5 (or threshold == 1.0 if impossible)
    flagged = p >= thr
    if flagged.sum() > 0:
        prec = float(y[flagged].sum()) / int(flagged.sum())
        assert prec >= 0.5 or thr == 1.0


def test_report_returns_expected_keys():
    y, p, amount = _make_scores()
    r = report(y, p, amount, thr=0.5, name="test")
    for key in ["model", "PR-AUC", "ROC-AUC", "precision", "recall", "F1", "FP", "FN", "cost", "saved%"]:
        assert key in r


def test_summarize_aggregates_multiple_seeds():
    rows = []
    for seed in range(3):
        y, p, amount = _make_scores(seed)
        rows.append(report(y, p, amount, thr=0.5, name="XGB"))
    df_runs = pd.DataFrame(rows)
    df_runs["model"] = "XGB"
    summary = summarize(df_runs)
    assert len(summary) == 1
    assert "±" in summary["PR-AUC"].iloc[0]


# ─────────────────────────── database / cache tests ─────────────────────────

@pytest.fixture
def sqlite_engine(tmp_path):
    return make_engine(f"sqlite:///{tmp_path}/test.db")


def test_save_and_retrieve_decision(sqlite_engine):
    result = {
        "txn_id": 12345,
        "fraud_prob": 0.87,
        "decision": "FRAUD_BLOCK",
        "model_version": "v-test",
        "factors": [{"feature": "C1", "shap": 0.4}],
    }
    save_decision(sqlite_engine, result)
    row = get_decision(sqlite_engine, 12345)
    assert row is not None
    assert row["decision"] == "FRAUD_BLOCK"
    assert round(row["fraud_prob"], 2) == 0.87


def test_get_decision_returns_none_for_missing(sqlite_engine):
    assert get_decision(sqlite_engine, 99999) is None


def test_recent_stats_empty_db(sqlite_engine):
    stats = recent_stats(sqlite_engine)
    assert stats == {"n": 0}


def test_recent_stats_with_data(sqlite_engine):
    for i, (decision, prob) in enumerate([("APPROVE", 0.05), ("FRAUD_REVIEW", 0.45), ("FRAUD_BLOCK", 0.92)]):
        save_decision(sqlite_engine, {
            "txn_id": i + 1000,
            "fraud_prob": prob,
            "decision": decision,
            "model_version": "v-test",
            "factors": [],
        })
    stats = recent_stats(sqlite_engine)
    assert stats["n"] == 3
    assert "APPROVE" in stats["decisions"]
    assert 0 < stats["mean_prob"] < 1


def test_cache_disabled_when_no_redis():
    """Cache gracefully degrades when Redis is unavailable (no URL set)."""
    cache = Cache(url=None)
    assert not cache.enabled
    assert cache.get("any_key") is None
    cache.set("any_key", {"foo": "bar"})   # should not raise


# ─────────────────────────── model architecture tests ───────────────────────
# These tests require torch; they are automatically skipped when torch is absent
# (pytest.importorskip at module level already handles this, but local imports
# keep the test self-contained for readability).

@_no_torch
def test_res_block_output_shape():
    import torch
    from finguard.models import ResBlock
    block = ResBlock(dim=64, dropout=0.0)
    x = torch.randn(8, 64)
    out = block(x)
    assert out.shape == (8, 64)


@_no_torch
def test_fraud_mlp_output_shape():
    import torch
    from finguard.models import FraudMLP
    mlp = FraudMLP(n_features=50, hidden=64, n_blocks=2, dropout=0.0)
    x = torch.randn(16, 50)
    out = mlp(x)
    assert out.shape == (16,)   # one logit per row


@_no_torch
def test_fraud_mlp_zero_blocks_is_plain_mlp():
    import torch
    from finguard.models import FraudMLP
    mlp = FraudMLP(n_features=20, hidden=32, n_blocks=0, dropout=0.0)
    x = torch.randn(4, 20)
    out = mlp(x)
    assert out.shape == (4,)


# ─────────────────────────── assistant tests ────────────────────────────────

def test_assistant_finds_the_transaction_id():
    assert find_transaction_id("Why was transaction #3487632 flagged?") == 3487632
    assert find_transaction_id("hello") is None
    assert find_transaction_id("transaction 1234 looks suspicious") == 1234


def test_assistant_ignores_short_numbers():
    """Numbers shorter than 4 digits should be ignored."""
    assert find_transaction_id("3 items in cart") is None
    assert find_transaction_id("score: 99 out of 100") is None


# ─────────────────────── feedback / quality_stats tests ─────────────────────

def test_save_and_retrieve_feedback(sqlite_engine):
    from finguard.db import save_feedback, quality_stats
    # Save a decision first
    save_decision(sqlite_engine, {
        "txn_id": 5001, "fraud_prob": 0.85, "decision": "FRAUD_BLOCK",
        "model_version": "v-test", "factors": [],
    })
    save_feedback(sqlite_engine, 5001, is_fraud=True)
    stats = quality_stats(sqlite_engine)
    assert stats["labeled"] == 1
    assert stats["tp"] == 1
    assert stats["precision"] == 1.0
    assert stats["recall"] == 1.0


def test_quality_stats_false_positive(sqlite_engine):
    from finguard.db import save_feedback, quality_stats
    save_decision(sqlite_engine, {
        "txn_id": 5002, "fraud_prob": 0.9, "decision": "FRAUD_BLOCK",
        "model_version": "v-test", "factors": [],
    })
    save_feedback(sqlite_engine, 5002, is_fraud=False)   # false alarm
    stats = quality_stats(sqlite_engine)
    assert stats["fp"] >= 1
    assert stats["precision"] < 1.0


def test_quality_stats_empty(sqlite_engine):
    from finguard.db import quality_stats
    stats = quality_stats(sqlite_engine)
    assert stats["labeled"] == 0
    assert stats["precision"] is None
    assert stats["recall"] is None


# ─────────────────────── LiveMonitor in-memory quality ───────────────────────

def test_live_monitor_quality():
    from finguard.monitor import LiveMonitor
    import tempfile, json, os

    # Build a minimal reference.json so LiveMonitor can be constructed
    with tempfile.TemporaryDirectory() as tmp:
        ref = {"__score__": {"edges": [], "props": [1.0, 0.0]}}
        (Path(tmp) / "reference.json").write_text(json.dumps(ref))
        mon = LiveMonitor(tmp, feature_names=[], review_thr=0.5)

    # Simulate 3 scored transactions (no features needed for quality test)
    mon._window = {1001: ("FRAUD_BLOCK", 0.9), 1002: ("APPROVE", 0.1), 1003: ("FRAUD_REVIEW", 0.6)}

    # Feedback: 1001 is true fraud (TP), 1002 is true fraud (FN), 1003 is legit (FP)
    mon._feedback = {1001: True, 1002: True, 1003: False}

    q = mon.quality()
    assert q["labeled"] == 3
    assert q["tp"] == 1   # 1001 flagged + fraud
    assert q["fp"] == 1   # 1003 flagged + legit
    assert q["fn"] == 1   # 1002 approved + fraud
    assert round(q["precision"], 4) == 0.5    # 1/(1+1)
    assert round(q["recall"], 4) == 0.5       # 1/(1+1)


def test_live_monitor_add_feedback_triggers_gauge_update():
    """add_feedback should not raise even when Prometheus gauges are present."""
    from finguard.monitor import LiveMonitor
    import tempfile, json

    with tempfile.TemporaryDirectory() as tmp:
        ref = {"__score__": {"edges": [], "props": [1.0, 0.0]}}
        (Path(tmp) / "reference.json").write_text(json.dumps(ref))
        mon = LiveMonitor(tmp, feature_names=[], review_thr=0.5)

    mon._window[9999] = ("FRAUD_BLOCK", 0.95)
    mon.add_feedback(9999, is_fraud=True)   # must not raise
    q = mon.quality()
    assert q["labeled"] == 1

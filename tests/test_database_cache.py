"""Tests verifying database persistence, statistics, feedback, and Redis cache handling."""

import pytest

from database.db import (
    Cache,
    get_decision,
    quality_stats,
    recent_stats,
    save_decision,
    save_feedback,
)


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
    cache.set("any_key", {"foo": "bar"})


def test_save_and_retrieve_feedback(sqlite_engine):
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
    save_decision(sqlite_engine, {
        "txn_id": 5002, "fraud_prob": 0.9, "decision": "FRAUD_BLOCK",
        "model_version": "v-test", "factors": [],
    })
    save_feedback(sqlite_engine, 5002, is_fraud=False)
    stats = quality_stats(sqlite_engine)
    assert stats["fp"] >= 1
    assert stats["precision"] < 1.0


def test_quality_stats_empty(sqlite_engine):
    stats = quality_stats(sqlite_engine)
    assert stats["labeled"] == 0
    assert stats["precision"] is None
    assert stats["recall"] is None

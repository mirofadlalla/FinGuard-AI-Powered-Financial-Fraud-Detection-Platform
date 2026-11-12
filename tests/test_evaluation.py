"""Unit tests for the evaluation showcase helper (no trained model required)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finguard import APPROVE, BLOCK, REVIEW
from finguard.evaluation import decide_batch, pick_showcase, summary_counts


def test_decide_batch_thresholds():
    probs = np.array([0.01, 0.20, 0.90])
    d = decide_batch(probs, review_thr=0.10, block_thr=0.80)
    assert list(d) == [APPROVE, REVIEW, BLOCK]


def test_pick_showcase_covers_all_nonempty_buckets():
    ids = np.arange(6)
    y = np.array([1, 1, 1, 0, 0, 0])
    probs = np.array([0.95, 0.40, 0.01, 0.01, 0.40, 0.95])
    amounts = np.ones(6)
    decisions = decide_batch(probs, 0.10, 0.80)
    rows = pick_showcase(ids, y, probs, amounts, decisions, n_per_bucket=1, seed=0)
    buckets = {r.bucket for r in rows}
    assert "fraud / BLOCK" in buckets
    assert "fraud / REVIEW" in buckets
    assert "fraud / APPROVE (missed)" in buckets
    assert "legit / APPROVE" in buckets
    assert "legit / REVIEW (false alarm)" in buckets
    assert "legit / BLOCK (severe FP)" in buckets


def test_summary_counts():
    y = np.array([1, 1, 0, 0])
    decisions = np.array([BLOCK, APPROVE, APPROVE, REVIEW])
    s = summary_counts(y, decisions)
    assert s["caught"] == 1
    assert s["missed"] == 1
    assert s["false_alarms"] == 1
    assert s["true_negatives"] == 1
    assert s["recall"] == 0.5

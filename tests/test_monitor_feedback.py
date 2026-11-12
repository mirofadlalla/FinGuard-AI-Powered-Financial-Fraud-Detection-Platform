"""Tests verifying LiveMonitor quality tracking and Prometheus gauge updates."""

import json
import tempfile
from pathlib import Path
import pytest

from monitoring.monitor import LiveMonitor


def test_live_monitor_quality():
    with tempfile.TemporaryDirectory() as tmp:
        ref = {"__score__": {"edges": [], "props": [1.0, 0.0]}}
        (Path(tmp) / "reference.json").write_text(json.dumps(ref))
        mon = LiveMonitor(tmp, feature_names=[], review_thr=0.5)

    mon._window = {1001: ("FRAUD_BLOCK", 0.9), 1002: ("APPROVE", 0.1), 1003: ("FRAUD_REVIEW", 0.6)}
    mon._feedback = {1001: True, 1002: True, 1003: False}

    q = mon.quality()
    assert q["labeled"] == 3
    assert q["tp"] == 1
    assert q["fp"] == 1
    assert q["fn"] == 1
    assert round(q["precision"], 4) == 0.5
    assert round(q["recall"], 4) == 0.5


def test_live_monitor_add_feedback_triggers_gauge_update():
    with tempfile.TemporaryDirectory() as tmp:
        ref = {"__score__": {"edges": [], "props": [1.0, 0.0]}}
        (Path(tmp) / "reference.json").write_text(json.dumps(ref))
        mon = LiveMonitor(tmp, feature_names=[], review_thr=0.5)

    mon._window[9999] = ("FRAUD_BLOCK", 0.95)
    mon.add_feedback(9999, is_fraud=True)
    q = mon.quality()
    assert q["labeled"] == 1

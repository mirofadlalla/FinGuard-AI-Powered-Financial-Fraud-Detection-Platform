"""System monitoring: Prometheus metrics + a live drift monitor.

What we watch (Prometheus scrapes /metrics, Grafana draws it, alerts.yml raises alarms):
  * SYSTEM : request rate, errors, latency, cache hit ratio
  * MODEL  : decisions per type, score distribution, share of flagged transactions
  * DATA   : PSI of the live scores and of the live features compared with training
  * QUALITY: live precision / recall from the real outcomes that arrive later (feedback)
"""
import json
from collections import deque
from pathlib import Path
import numpy as np
from prometheus_client import Counter, Gauge, Histogram

from finguard.drift import feature_psi, level, psi

#  system ----------
REQUESTS = Counter("finguard_requests_total", "HTTP requests", ["endpoint", "method", "status"])

LATENCY = Histogram("finguard_request_latency_seconds", "Request latency", ["endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10))

CACHE = Counter("finguard_cache_total", "Redis cache lookups", ["result"])             # hit / miss
ASSISTANT = Counter("finguard_assistant_total", "Assistant answers", ["source"])       # llm / template

# ---------- model ----------
DECISIONS = Counter("finguard_decisions_total", "Decisions taken", ["decision"])

SCORES = Histogram("finguard_fraud_score", "Fraud probability of scored transactions",
                   buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0))

MODEL_INFO = Gauge("finguard_model_info", "Model version that is loaded (value is always 1)", ["version"])

BASELINE = Gauge("finguard_baseline", "What the model achieved on the test set when it was trained", ["metric"])

FLAGGED_RECENT = Gauge("finguard_flagged_share_recent", "Share of flagged transactions in the live window")

# ---------- data drift ----------
SCORE_PSI = Gauge("finguard_score_psi", "PSI of the live fraud scores vs validation")
FEATURES_BIG_SHIFT = Gauge("finguard_features_big_shift", "Number of features with PSI > 0.25 in the live window")
FEATURE_PSI = Gauge("finguard_feature_psi", "PSI of the most drifted features", ["feature"])

# ---------- quality (needs real outcomes) ----------
LIVE_PRECISION = Gauge("finguard_live_precision", "Precision of flagged transactions with a known outcome")
LIVE_RECALL = Gauge("finguard_live_recall", "Recall on transactions with a known outcome")
LABELED = Gauge("finguard_labeled_transactions", "Transactions with a known real outcome")


def set_baselines(meta):
    """Remember how the model looked at training time, so alerts can compare live numbers with it."""
    MODEL_INFO.labels(meta["version"]).set(1)
    for key in ("share_flagged", "review_precision", "review_recall"):
        if key in meta.get("metrics", {}):
            BASELINE.labels(key).set(meta["metrics"][key])


def update_quality_gauges(stats):
    """Publish live quality numbers to Prometheus from a quality_stats() result dict."""
    LABELED.set(stats["labeled"])
    if stats["precision"] is not None:
        LIVE_PRECISION.set(stats["precision"])
    if stats["recall"] is not None:
        LIVE_RECALL.set(stats["recall"])


class LiveMonitor:
    """Keeps the last N scored transactions and compares them with training distribution (PSI).

    Also tracks in-memory feedback (real outcomes) for a fast quality estimate that does not
    require a DB query on every request.  The DB-backed quality_stats() function is the
    authoritative source; this in-memory buffer is just for the /monitoring/performance endpoint.
    """

    def __init__(self, model_dir, feature_names, review_thr, window=2000, min_rows=200, every=100):
        path = Path(model_dir) / "reference.json"
        self.reference = json.loads(path.read_text()) if path.exists() else None
        self.names, self.review_thr = feature_names, review_thr
        self.rows, self.scores = deque(maxlen=window), deque(maxlen=window)
        self.min_rows, self.every, self.seen = min_rows, every, 0
        self.last = {"rows_in_window": 0}
        # Maps txn_id -> (decision, prob) for the current window so we can join with feedback
        self._window: dict[int, tuple[str, float]] = {}
        # Maps txn_id -> bool for feedback received in the current window
        self._feedback: dict[int, bool] = {}
        # Track published Prometheus feature labels to avoid FEATURE_PSI.clear()
        self._published_features: set = set()

    def add(self, txn_id: int, x, prob: float, decision: str):
        """Call this for every NEW scored transaction."""
        self.rows.append(np.asarray(x, dtype="float32"))
        self.scores.append(float(prob))
        self._window[txn_id] = (decision, float(prob))
        self.seen += 1
        DECISIONS.labels(decision).inc()
        SCORES.observe(prob)
        if self.seen % self.every == 0:
            self.refresh()

    def add_feedback(self, txn_id: int, is_fraud: bool):
        """Record the real outcome for a transaction (called from the /feedback endpoint)."""
        self._feedback[txn_id] = is_fraud
        self._publish_quality()

    def quality(self) -> dict:
        """In-memory live precision / recall from the feedback received so far."""
        joined = [(self._window[tid], label)
                  for tid, label in self._feedback.items()
                  if tid in self._window]
        if not joined:
            return {"labeled": 0, "precision": None, "recall": None}
        tp = sum(1 for (d, _), f in joined if d != "APPROVE" and f)
        fp = sum(1 for (d, _), f in joined if d != "APPROVE" and not f)
        fn = sum(1 for (d, _), f in joined if d == "APPROVE" and f)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        return {"labeled": len(joined), "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec, 4), "recall": round(rec, 4)}

    def _publish_quality(self):
        stats = self.quality()
        update_quality_gauges(stats)

    def refresh(self):
        """Recompute drift numbers and publish them as Prometheus gauges."""
        if self.reference is None or len(self.rows) < self.min_rows:
            return self.last
        X, scores = np.vstack(self.rows), np.array(self.scores)
        feature_drift = feature_psi(self.reference, X, self.names)
        score_drift = psi(self.reference["__score__"], scores)
        top = sorted(feature_drift.items(), key=lambda kv: -kv[1])[:10]

        SCORE_PSI.set(score_drift)
        FEATURES_BIG_SHIFT.set(sum(v > 0.25 for v in feature_drift.values()))
        FLAGGED_RECENT.set(float((scores >= self.review_thr).mean()))

        # Update top-10 feature PSI gauges in-place (zero-out stale labels without calling .clear())
        top_names = {name for name, _ in top}
        for stale in self._published_features - top_names:
            FEATURE_PSI.labels(stale).set(0.0)
        for name, value in top:
            FEATURE_PSI.labels(name).set(value)
        self._published_features = top_names

        self.last = {
            "rows_in_window": len(scores),
            "score_psi": round(score_drift, 4),
            "score_drift": level(score_drift),
            "features_big_shift": int(sum(v > 0.25 for v in feature_drift.values())),
            "flagged_share": round(float((scores >= self.review_thr).mean()), 4),
            "top_drifted_features": [{"feature": n, "psi": round(v, 3), "level": level(v)} for n, v in top],
        }
        return self.last

"""FinGuard API.

Run (from the project root with PYTHONPATH=src):
    uvicorn finguard.api:app --reload

Or via Docker:
    docker compose up --build

Endpoints
---------
GET  /health                    model version, store size, Redis status
GET  /score/{txn_id}            score a stored transaction (cached)
POST /predict                   score a raw feature dict (not persisted)
GET  /decisions/{txn_id}        retrieve a saved decision + SHAP factors
POST /investigate               NL answer: "Why was transaction 3487632 flagged?"
POST /feedback                  record real outcome (is_fraud) for a scored transaction
GET  /monitoring/summary        last 1000 decisions, mean score (60 s Redis cache)
GET  /monitoring/drift          live PSI drift window from LiveMonitor
GET  /monitoring/performance    live precision / recall from confirmed feedback
GET  /metrics                   Prometheus scrape endpoint
"""
import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from finguard import assistant
from finguard.db import (
    Cache, get_decision, make_engine, quality_stats, recent_stats,
    save_decision, save_feedback,
)
from finguard.monitor import (
    ASSISTANT, CACHE, LATENCY, REQUESTS,
    LiveMonitor, set_baselines, update_quality_gauges,
)
from finguard.serving import APPROVE, FeatureStore, Model

log = logging.getLogger(__name__)
state: dict = {}


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app):
    """Load model, feature store, DB, cache and monitoring once at startup."""
    model = Model(os.getenv("MODEL_DIR", "artifacts/model"))
    state["model"] = model
    state["store"] = FeatureStore(os.getenv("FEATURE_FILE", "data/ieee_ready.npz"))
    state["db"]    = make_engine()
    state["cache"] = Cache()

    # Wire monitoring
    monitor = LiveMonitor(
        model_dir=os.getenv("MODEL_DIR", "artifacts/model"),
        feature_names=model.feature_names,
        review_thr=model.review_thr,
    )
    state["monitor"] = monitor
    set_baselines(model.meta)      # publish training-time baselines to Prometheus

    log.info("FinGuard started — model %s", model.version)
    yield
    log.info("FinGuard shutting down")


app = FastAPI(title="FinGuard", version="2.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request-level middleware: latency + request counter
# ---------------------------------------------------------------------------

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    endpoint = request.url.path
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    LATENCY.labels(endpoint).observe(elapsed)
    REQUESTS.labels(endpoint, request.method, str(response.status_code)).inc()
    return response


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_row(txn_id: int, x) -> dict:
    """Score one row, explain if suspicious, persist the decision, feed the monitor."""
    model   = state["model"]
    monitor = state["monitor"]

    prob     = float(model.predict(x.reshape(1, -1))[0])
    decision = model.decide(prob)
    factors  = model.explain(x) if decision != APPROVE else []

    result = {
        "txn_id":        int(txn_id),
        "fraud_prob":    round(prob, 4),
        "decision":      decision,
        "model_version": model.version,
        "factors":       factors,
    }
    save_decision(state["db"], result)
    monitor.add(txn_id, x, prob, decision)   # feeds Prometheus counters + drift window
    return result


def _score_by_id(txn_id: int) -> dict:
    """Look up a transaction in the feature store and score it (with Redis cache)."""
    cache  = state["cache"]
    cached = cache.get(f"score:{txn_id}")
    if cached:
        CACHE.labels("hit").inc()
        return {**cached, "from_cache": True}
    CACHE.labels("miss").inc()
    x = state["store"].get(txn_id)
    if x is None:
        raise HTTPException(404, f"transaction {txn_id} not found in the feature store")
    result = _score_row(txn_id, x)
    cache.set(f"score:{txn_id}", result)
    return {**result, "from_cache": False}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check: model version, store size, Redis availability."""
    return {
        "status":                "ok",
        "model_version":         state["model"].version,
        "transactions_in_store": len(state["store"]),
        "redis_cache":           state["cache"].enabled,
    }


@app.get("/score/{txn_id}")
def score(txn_id: int):
    """Score a transaction that is already in the feature store."""
    return _score_by_id(txn_id)


class Features(BaseModel):
    features: dict[str, float | None]  # feature name -> scaled value (missing ones become NaN)


@app.post("/predict")
def predict(body: Features):
    """Score a raw feature dict directly (for testing / real-time integration).

    This endpoint is NOT persisted to the database — use /score for production traffic.
    """
    model   = state["model"]
    names   = model.feature_names
    unknown = [k for k in body.features if k not in set(names)]
    if unknown:
        raise HTTPException(422, f"unknown features: {unknown[:5]}")
    x = np.array(
        [np.nan if body.features.get(n) is None else body.features[n] for n in names],
        dtype="float32",
    )
    prob     = float(model.predict(x.reshape(1, -1))[0])
    decision = model.decide(prob)
    return {
        "fraud_prob": round(prob, 4),
        "decision":   decision,
        "factors":    model.explain(x) if decision != APPROVE else [],
    }


@app.get("/decisions/{txn_id}")
def decision(txn_id: int):
    """Retrieve the saved decision (with SHAP factors) for a transaction scored before."""
    row = get_decision(state["db"], txn_id)
    if row is None:
        raise HTTPException(404, "this transaction has not been scored yet")
    return row


class Question(BaseModel):
    question: str


@app.post("/investigate")
def investigate(body: Question):
    """Ask the assistant in plain English.  Example: 'Why was transaction 3487632 flagged?'"""
    txn_id = assistant.find_transaction_id(body.question)
    if txn_id is None:
        raise HTTPException(422, "please include a transaction id in the question")

    result = get_decision(state["db"], txn_id)
    # Re-score if never seen or if it was flagged but SHAP factors are missing.
    if result is None or (result["decision"] != APPROVE and not result.get("factors")):
        result = _score_by_id(txn_id)

    answer_text, source = assistant.answer(body.question, result)
    ASSISTANT.labels(source).inc()
    return {"txn_id": txn_id, "answer": answer_text}


class FeedbackBody(BaseModel):
    txn_id:   int
    is_fraud: bool


@app.post("/feedback")
def feedback(body: FeedbackBody):
    """Record the real outcome for a scored transaction (e.g. from a chargeback system).

    Used to compute live precision / recall.  Call as outcomes become known —
    typically hours or days after the original decision.
    """
    # Persist to DB for durability
    save_feedback(state["db"], body.txn_id, body.is_fraud)
    # Feed in-memory monitor for fast quality gauge updates
    state["monitor"].add_feedback(body.txn_id, body.is_fraud)
    return {"txn_id": body.txn_id, "is_fraud": body.is_fraud, "recorded": True}


@app.get("/monitoring/summary")
def monitoring_summary():
    """Last 1000 decisions: decision-type counts + mean fraud score vs. training baseline.

    Result is cached in Redis for 60 s.
    """
    cache  = state["cache"]
    cached = cache.get("monitoring:summary")
    if cached:
        return {**cached, "from_cache": True}
    stats = recent_stats(state["db"])
    stats["training_mean_prob_on_validation"] = state["model"].meta.get("val_mean_prob")
    cache.set("monitoring:summary", stats, ttl=60)
    return {**stats, "from_cache": False}


@app.get("/monitoring/drift")
def monitoring_drift():
    """Live data-drift report: PSI of scores and features vs. training distribution.

    Updated every 100 scored transactions (configurable via LiveMonitor.every).
    Returns 'rows_in_window: 0' until the minimum window size (200 rows) is reached.
    """
    return state["monitor"].last


@app.get("/monitoring/performance")
def monitoring_performance():
    """Live model quality: precision and recall computed from confirmed feedback.

    Returns null for precision / recall until at least one feedback record is received.
    For the authoritative DB-backed version (uses the full history) see quality_stats().
    """
    # In-memory monitor (fast, current window only)
    mem_stats = state["monitor"].quality()
    # DB-backed stats (slower, full history, authoritative)
    db_stats  = quality_stats(state["db"])
    # Publish to Prometheus
    update_quality_gauges(db_stats)
    return {
        "window":    mem_stats,    # last N transactions in the rolling window
        "all_time":  db_stats,     # all feedback ever recorded
    }


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint.  Add this URL to your prometheus.yml scrape_configs."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

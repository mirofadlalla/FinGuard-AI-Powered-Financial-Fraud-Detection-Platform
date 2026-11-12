"""FinGuard HTTP API.

    uvicorn api.app:app --reload
    docker compose up --build
"""
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from assistant import assistant
from database.db import get_decision, quality_stats, recent_stats, save_feedback
from monitoring.monitor import ASSISTANT, LATENCY, REQUESTS, update_quality_gauges
from api.raw_score import PreprocessorUnavailable, RawScoreError, score_raw_row
from api.runtime import lifespan, reload_model, runtime
from api.schemas import Features, FeedbackBody, Question
from api.scoring import TransactionNotFound, features_to_vector, score_by_id
from api.serving import APPROVE

app = FastAPI(title="FinGuard", version="2.0", lifespan=lifespan)

# In production, replace "*" with the actual frontend origin(s), e.g.:
#   allow_origins=["https://your-dashboard.example.com"]
_CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC = Path(__file__).parent / "static"
if _STATIC.exists():
    app.mount("/ui", StaticFiles(directory=str(_STATIC), html=True), name="ui")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui")


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    LATENCY.labels(request.url.path).observe(time.perf_counter() - start)
    REQUESTS.labels(request.url.path, request.method, str(response.status_code)).inc()
    return response


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": runtime.model.version,
        "transactions_in_store": len(runtime.store),
        "redis_cache": runtime.cache.enabled,
    }


@app.get("/score/{txn_id}")
def score(txn_id: int):
    try:
        return score_by_id(runtime, txn_id)
    except TransactionNotFound as e:
        raise HTTPException(404, str(e)) from e


@app.post("/predict")
def predict(body: Features):
    """Score a preprocessed feature dict. Not persisted — use /score or /predict/raw for that."""
    model = runtime.model
    unknown = [k for k in body.features if k not in set(model.feature_names)]
    if unknown:
        raise HTTPException(422, f"unknown features: {unknown[:5]}")
    x = features_to_vector(model.feature_names, body.features)
    prob = float(model.predict(x.reshape(1, -1))[0])
    decision = model.decide(prob)
    return {
        "fraud_prob": round(prob, 4),
        "decision": decision,
        "factors": model.explain(x) if decision != APPROVE else [],
    }


@app.post("/predict/raw")
async def predict_raw(request: Request):
    """Score a raw IEEE-CIS transaction (original CSV field names)."""
    try:
        row = await request.json()
    except Exception:
        raise HTTPException(422, "Request body must be valid JSON.")
    use_store = request.query_params.get("use_store", "true").lower() != "false"
    debug = request.query_params.get("debug", "").lower() == "true"
    try:
        return score_raw_row(runtime, row, use_store=use_store, debug=debug)
    except PreprocessorUnavailable as e:
        raise HTTPException(503, str(e)) from e
    except RawScoreError as e:
        raise HTTPException(422, str(e)) from e


def _saved_decision(txn_id: int):
    row = get_decision(runtime.db, txn_id)
    if row is None:
        raise HTTPException(404, "this transaction has not been scored yet")
    return row


@app.get("/decision/{txn_id}")
@app.get("/decisions/{txn_id}")
def decision(txn_id: int):
    return _saved_decision(txn_id)


@app.post("/investigate")
def investigate(body: Question):
    txn_id = assistant.find_transaction_id(body.question)
    if txn_id is None:
        raise HTTPException(422, "please include a transaction id in the question")

    result = get_decision(runtime.db, txn_id)
    if result is None or (result["decision"] != APPROVE and not result.get("factors")):
        try:
            result = score_by_id(runtime, txn_id)
        except TransactionNotFound as e:
            raise HTTPException(404, str(e)) from e

    answer_text, source = assistant.answer(body.question, result)
    ASSISTANT.labels(source).inc()
    return {"txn_id": txn_id, "answer": answer_text}


@app.post("/feedback")
def feedback(body: FeedbackBody):
    save_feedback(runtime.db, body.txn_id, body.is_fraud)
    runtime.monitor.add_feedback(body.txn_id, body.is_fraud)
    return {"txn_id": body.txn_id, "is_fraud": body.is_fraud, "recorded": True}


@app.get("/monitoring/summary")
def monitoring_summary():
    cache = runtime.cache
    cached = cache.get("monitoring:summary")
    if cached:
        return {**cached, "from_cache": True}
    stats = recent_stats(runtime.db)
    stats["training_mean_prob_on_validation"] = runtime.model.meta.get("val_mean_prob")
    cache.set("monitoring:summary", stats, ttl=60)
    return {**stats, "from_cache": False}


@app.get("/monitoring/drift")
def monitoring_drift():
    return runtime.monitor.last


@app.get("/monitoring/performance")
def monitoring_performance():
    mem_stats = runtime.monitor.quality()
    db_stats = quality_stats(runtime.db)
    update_quality_gauges(db_stats)
    return {"window": mem_stats, "all_time": db_stats}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/admin/reload", include_in_schema=False)
def admin_reload():
    try:
        reload_model(runtime)
    except RuntimeError as e:
        raise HTTPException(503, f"{e} — try again shortly") from e
    return {
        "reloaded": True,
        "model_version": runtime.model.version,
        "features": len(runtime.model.feature_names),
    }

"""FinGuard API.   Run:  uvicorn finguard.api:app --reload   (from the project folder, with PYTHONPATH=src)"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from finguard import assistant
from finguard.db import Cache, get_decision, make_engine, recent_stats, save_decision
from finguard.serving import APPROVE, FeatureStore, Model

state = {}


@asynccontextmanager
async def lifespan(app):
    """Runs once when the API starts."""
    state["model"] = Model(os.getenv("MODEL_DIR", "artifacts/model"))
    state["store"] = FeatureStore(os.getenv("FEATURE_FILE", "data/ieee_ready.npz"))
    state["db"] = make_engine()
    state["cache"] = Cache()
    yield


app = FastAPI(title="FinGuard", version="2.0", lifespan=lifespan)


def score_row(txn_id, x):
    """Score one row, explain it if it is suspicious, save the decision."""
    model = state["model"]
    prob = float(model.predict(x.reshape(1, -1))[0])
    decision = model.decide(prob)
    # explanations are slower, so we only compute them for transactions that are not approved
    factors = model.explain(x) if decision != APPROVE else []
    result = {"txn_id": int(txn_id), "fraud_prob": round(prob, 4), "decision": decision,
              "model_version": model.version, "factors": factors}
    save_decision(state["db"], result)
    return result


def score_by_id(txn_id):
    """Look up a transaction in the feature store and score it (with Redis cache)."""
    cache = state["cache"]
    cached = cache.get(f"score:{txn_id}")
    if cached:
        return {**cached, "from_cache": True}
    x = state["store"].get(txn_id)
    if x is None:
        raise HTTPException(404, f"transaction {txn_id} is not in the feature store")
    result = score_row(txn_id, x)
    cache.set(f"score:{txn_id}", result)
    return {**result, "from_cache": False}


@app.get("/health")
def health():
    return {"status": "ok", "model_version": state["model"].version,
            "transactions_in_store": len(state["store"]), "redis_cache": state["cache"].enabled}


@app.get("/score/{txn_id}")
def score(txn_id: int):
    """Score a transaction that is already in the feature store."""
    return score_by_id(txn_id)


class Features(BaseModel):
    features: dict[str, float | None]     # feature name -> scaled value (missing ones become NaN)


@app.post("/predict")
def predict(body: Features):
    """Score a feature dictionary directly (advanced use / testing)."""
    import numpy as np
    names = state["model"].feature_names
    unknown = [k for k in body.features if k not in names]
    if unknown:
        raise HTTPException(422, f"unknown features: {unknown[:5]}")
    x = np.array([np.nan if body.features.get(n) is None else body.features[n] for n in names], dtype="float32")
    model = state["model"]
    prob = float(model.predict(x.reshape(1, -1))[0])
    decision = model.decide(prob)
    return {"fraud_prob": round(prob, 4), "decision": decision,
            "factors": model.explain(x) if decision != APPROVE else []}


@app.get("/decisions/{txn_id}")
def decision(txn_id: int):
    """The saved decision (with its explanation) of a transaction scored before."""
    row = get_decision(state["db"], txn_id)
    if row is None:
        raise HTTPException(404, "this transaction was not scored yet")
    return row


class Question(BaseModel):
    question: str


@app.post("/investigate")
def investigate(body: Question):
    """Ask the assistant, ex: 'Why was transaction 3487632 flagged?'"""
    txn_id = assistant.find_transaction_id(body.question)
    if txn_id is None:
        raise HTTPException(422, "please include a transaction id in the question")
    result = get_decision(state["db"], txn_id)
    if result is None or (result["decision"] != APPROVE and not result["factors"]):
        result = score_by_id(txn_id)
    return {"txn_id": txn_id, "answer": assistant.answer(body.question, result)}


@app.get("/monitoring/summary")
def monitoring():
    """Quick health of the model in production: how many decisions of each type, mean score vs training.

    Result is cached in Redis for 60 s to avoid a table scan on every call.
    """
    cache = state["cache"]
    cached = cache.get("monitoring:summary")
    if cached:
        return {**cached, "from_cache": True}
    stats = recent_stats(state["db"])
    stats["training_mean_prob_on_validation"] = state["model"].meta.get("val_mean_prob")
    cache.client and cache.set("monitoring:summary", stats)   # TTL inherits cache.ttl (1 h), fine for monitoring
    # Override with a short TTL so staleness is bounded to 60 s
    try:
        if cache.client:
            import json
            cache.client.setex("monitoring:summary", 60, json.dumps(stats))
    except Exception:
        pass
    return {**stats, "from_cache": False}

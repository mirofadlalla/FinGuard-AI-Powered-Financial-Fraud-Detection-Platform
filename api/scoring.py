"""Score a feature vector, persist the decision, and optionally cache by transaction id."""
from __future__ import annotations

import time

import numpy as np

from database.db import get_decision, save_decision
from monitoring.monitor import CACHE
from api.runtime import AppRuntime
from api.serving import APPROVE


class TransactionNotFound(LookupError):
    """Raised when a transaction id is neither in the feature store nor in saved decisions."""


def parse_txn_id(raw) -> int | None:
    try:
        if raw is None:
            return None
        return int(raw)
    except (TypeError, ValueError):
        return None


def score_row(rt: AppRuntime, txn_id: int, x) -> dict:
    """Score one row, explain if suspicious, persist, and feed the live monitor."""
    model = rt.model
    monitor = rt.monitor

    prob = float(model.predict(x.reshape(1, -1))[0])
    decision = model.decide(prob)
    factors = model.explain(x) if decision != APPROVE else []

    result = {
        "txn_id": int(txn_id),
        "fraud_prob": round(prob, 4),
        "decision": decision,
        "model_version": model.version,
        "factors": factors,
    }
    save_decision(rt.db, result)
    monitor.add(txn_id, x, prob, decision)
    return result


def remember(rt: AppRuntime, txn_id: int | None, x) -> dict:
    """Score, save the decision, and keep the vector for later /score lookups."""
    x = np.asarray(x, dtype="float32").reshape(-1)
    if txn_id is None:
        txn_id = int(time.time() * 1000) % 1_000_000_000 + 9_000_000_000
    rt.store.put(txn_id, x)
    return score_row(rt, txn_id, x)


def score_by_id(rt: AppRuntime, txn_id: int) -> dict:
    """Look up a transaction in the feature store and score it (with Redis cache)."""
    cache = rt.cache
    cached = cache.get(f"score:{txn_id}")
    if cached:
        CACHE.labels("hit").inc()
        return {**cached, "from_cache": True}
    CACHE.labels("miss").inc()
    x = rt.store.get(txn_id)
    if x is None:
        saved = get_decision(rt.db, txn_id)
        if saved is not None:
            return {**saved, "from_cache": False, "feature_source": "saved_decision"}
        raise TransactionNotFound(
            f"transaction {txn_id} not found. POST it to /predict/raw first "
            "(only train_transaction.csv IDs are preloaded; Kaggle test.csv IDs are stored after you predict them)."
        )
    result = score_row(rt, txn_id, np.asarray(x, dtype="float32"))
    cache.set(f"score:{txn_id}", result)
    return {**result, "from_cache": False}


def features_to_vector(feature_names: list[str], features: dict[str, float | None]) -> np.ndarray:
    """Build a model-ordered vector from a name→value dict. Unknown keys are the caller's problem."""
    return np.array(
        [np.nan if features.get(n) is None else features[n] for n in feature_names],
        dtype="float32",
    )

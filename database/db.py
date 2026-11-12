"""Database (PostgreSQL in Docker, SQLite when you run it locally) + Redis cache."""
import json
import logging
import os
from datetime import datetime, timezone
from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, JSON, String, create_engine, select
from sqlalchemy.orm import Session, declarative_base

log = logging.getLogger(__name__)

Base = declarative_base()


class Decision(Base):
    """One row per scored transaction."""
    __tablename__ = "decisions"
    txn_id = Column(BigInteger, primary_key=True)
    fraud_prob = Column(Float, nullable=False)
    decision = Column(String(20), nullable=False)
    model_version = Column(String(50))
    factors = Column(JSON)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    def as_dict(self):
        return {"txn_id": int(self.txn_id), "fraud_prob": round(self.fraud_prob, 4), "decision": self.decision,
                "model_version": self.model_version, "factors": self.factors or []}


class Feedback(Base):
    """Real outcome for a transaction (e.g. chargeback confirmed = fraud, cleared = legit).

    These arrive asynchronously (hours / days after the decision) and are used to compute
    live precision and recall so we can tell if the model quality is degrading in production.
    """
    __tablename__ = "feedback"
    txn_id = Column(BigInteger, primary_key=True)
    is_fraud = Column(Boolean, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


def make_engine(url=None):
    url = url or os.getenv("DATABASE_URL", "sqlite:///finguard.db")
    engine = create_engine(url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    return engine


def save_decision(engine, result):
    """Persist a scored transaction. Logs a warning if the transaction was already scored (re-score)."""
    with Session(engine) as session:
        existing = session.get(Decision, int(result["txn_id"]))
        if existing is not None:
            log.warning(
                "Re-scoring transaction %s (previous decision: %s, new decision: %s)",
                result["txn_id"], existing.decision, result["decision"],
            )
        session.merge(Decision(txn_id=result["txn_id"], fraud_prob=result["fraud_prob"], decision=result["decision"],
                               model_version=result["model_version"], factors=result["factors"]))
        session.commit()


def get_decision(engine, txn_id):
    with Session(engine) as session:
        row = session.get(Decision, int(txn_id))
        return row.as_dict() if row else None


def recent_stats(engine, limit=1000):
    """Summary of the last `limit` decisions (used for monitoring)."""
    with Session(engine) as session:
        rows = session.execute(select(Decision.decision, Decision.fraud_prob)
                               .order_by(Decision.created_at.desc()).limit(limit)).all()
    if not rows:
        return {"n": 0}
    counts = {}
    for decision, _ in rows:
        counts[decision] = counts.get(decision, 0) + 1
    probs = [p for _, p in rows]
    return {"n": len(rows), "mean_prob": round(sum(probs) / len(probs), 4), "decisions": counts}


def save_feedback(engine, txn_id: int, is_fraud: bool):
    """Record the real outcome for a transaction (upsert)."""
    with Session(engine) as session:
        session.merge(Feedback(txn_id=txn_id, is_fraud=is_fraud))
        session.commit()


def quality_stats(engine, limit=1000):
    """Live precision / recall computed from feedback joined with decisions.

    Only transactions that have BOTH a decision AND a real outcome are counted.
    Returns a dict with keys: labeled, tp, fp, fn, precision, recall.
    """
    with Session(engine) as session:
        rows = session.execute(
            select(Decision.decision, Feedback.is_fraud)
            .join(Feedback, Decision.txn_id == Feedback.txn_id)
            .order_by(Decision.created_at.desc())
            .limit(limit)
        ).all()

    if not rows:
        return {"labeled": 0, "precision": None, "recall": None}

    flagged_fraud = sum(1 for d, f in rows if d != "APPROVE" and f)
    flagged_legit = sum(1 for d, f in rows if d != "APPROVE" and not f)
    missed_fraud  = sum(1 for d, f in rows if d == "APPROVE" and f)

    tp, fp, fn = flagged_fraud, flagged_legit, missed_fraud
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    return {
        "labeled":   len(rows),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
    }


class Cache:
    """Tiny Redis cache. If Redis is not available the API still works, just without cache."""

    def __init__(self, url=None, ttl=3600):
        self.ttl, self.client = ttl, None
        url = url or os.getenv("REDIS_URL")
        if url:
            try:
                import redis
                self.client = redis.Redis.from_url(url, socket_connect_timeout=1)
                self.client.ping()
            except Exception:
                self.client = None

    @property
    def enabled(self):
        return self.client is not None

    def get(self, key):
        try:
            value = self.client.get(key) if self.client else None
            return json.loads(value) if value else None
        except Exception:
            log.debug("Redis GET failed for key %r", key, exc_info=True)
            return None

    def set(self, key, value, ttl: int | None = None):
        """Persist *value* under *key* with an optional per-call TTL override."""
        try:
            if self.client:
                self.client.setex(key, ttl if ttl is not None else self.ttl, json.dumps(value))
        except Exception:
            log.debug("Redis SET failed for key %r", key, exc_info=True)

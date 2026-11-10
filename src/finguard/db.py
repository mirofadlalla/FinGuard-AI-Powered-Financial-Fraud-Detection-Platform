"""Database (PostgreSQL in Docker, SQLite when you run it locally) + Redis cache."""
import json
import logging
import os
from datetime import datetime, timezone
from sqlalchemy import BigInteger, Column, DateTime, Float, JSON, String, create_engine, func, select
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

    def set(self, key, value):
        try:
            if self.client:
                self.client.setex(key, self.ttl, json.dumps(value))
        except Exception:
            log.debug("Redis SET failed for key %r", key, exc_info=True)

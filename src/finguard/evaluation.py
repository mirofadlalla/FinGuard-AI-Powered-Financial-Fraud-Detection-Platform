"""Pick a small, labeled showcase of test transactions across risk levels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finguard import APPROVE, BLOCK, REVIEW


@dataclass
class ShowcaseRow:
    txn_id: int
    label: str
    fraud_prob: float
    decision: str
    amount: float
    bucket: str


BUCKETS = (
    ("fraud / BLOCK", True, BLOCK),
    ("fraud / REVIEW", True, REVIEW),
    ("fraud / APPROVE (missed)", True, APPROVE),
    ("legit / APPROVE", False, APPROVE),
    ("legit / REVIEW (false alarm)", False, REVIEW),
    ("legit / BLOCK (severe FP)", False, BLOCK),
)


def bucket_mask(y, decisions, want_fraud: bool, want_decision: str) -> np.ndarray:
    label_ok = (y == 1) if want_fraud else (y == 0)
    return label_ok & (decisions == want_decision)


def pick_showcase(ids, y, probs, amounts, decisions, n_per_bucket: int = 2, seed: int = 0) -> list[ShowcaseRow]:
    """Deterministically sample a few rows from each (label, decision) cell."""
    rng = np.random.default_rng(seed)
    rows: list[ShowcaseRow] = []
    for bucket, want_fraud, want_decision in BUCKETS:
        idx = np.flatnonzero(bucket_mask(y, decisions, want_fraud, want_decision))
        if len(idx) == 0:
            continue
        take = min(n_per_bucket, len(idx))
        chosen = rng.choice(idx, size=take, replace=False)
        chosen.sort()
        for i in chosen:
            rows.append(
                ShowcaseRow(
                    txn_id=int(ids[i]),
                    label="FRAUD" if y[i] == 1 else "LEGIT",
                    fraud_prob=float(probs[i]),
                    decision=str(decisions[i]),
                    amount=float(amounts[i]),
                    bucket=bucket,
                )
            )
    return rows


def decide_batch(probs, review_thr: float, block_thr: float) -> np.ndarray:
    decisions = np.full(len(probs), APPROVE, dtype=object)
    decisions[probs >= review_thr] = REVIEW
    decisions[probs >= block_thr] = BLOCK
    return decisions


def score_split(model, X) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(model.predict(X), dtype="float64")
    return probs, decide_batch(probs, model.review_thr, model.block_thr)


def summary_counts(y, decisions) -> dict:
    fraud = y == 1
    legit = y == 0
    flagged = decisions != APPROVE
    n_fraud = int(fraud.sum())
    caught = int((fraud & flagged).sum())
    missed = int((fraud & ~flagged).sum())
    fp = int((legit & flagged).sum())
    tn = int((legit & ~flagged).sum())
    return {
        "n": int(len(y)),
        "fraud": n_fraud,
        "legit": int(legit.sum()),
        "caught": caught,
        "missed": missed,
        "false_alarms": fp,
        "true_negatives": tn,
        "recall": round(caught / max(n_fraud, 1), 4),
        "false_positive_rate": round(fp / max(int(legit.sum()), 1), 4),
    }

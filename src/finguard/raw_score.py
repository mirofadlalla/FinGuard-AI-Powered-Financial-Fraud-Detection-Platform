"""Score a raw IEEE-CIS JSON row (cold-start preprocessing + quality notes)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from finguard.raw_features import fill_identity, normalize_keys, prepare_raw_dataframe
from finguard.runtime import AppRuntime
from finguard.scoring import parse_txn_id, remember

# History/velocity features are always NaN (or cold-started) on a single raw row.
VEL_PREFIXES = (
    "uid_count_", "card1_count_", "card_amt_vs_mean", "uid_amt_vs_mean",
    "combo_", "_agg_", "uid_txn_",
)

# Empirically measured recall on the held-out test set when velocity features are NaN.
FULL_RECALL = 63.67
NOVEL_RECALL = 33.86


class PreprocessorUnavailable(RuntimeError):
    pass


class RawScoreError(ValueError):
    pass


def is_velocity_feature(name: str) -> bool:
    return any(p in name for p in VEL_PREFIXES)


def align_to_model(x: np.ndarray, prep, names: list[str]) -> np.ndarray:
    feat_map = {
        n: float(x[prep.feature_names.index(n)]) if n in prep.feature_names else float("nan")
        for n in names
    }
    return np.array([feat_map[n] for n in names], dtype="float32")


def coverage(x: np.ndarray) -> tuple[int, float]:
    nan_count = int(np.isnan(x).sum())
    n = len(x)
    pct = round((n - nan_count) / n * 100, 1) if n else 0.0
    return nan_count, pct


def velocity_quality(names: list[str], x: np.ndarray) -> tuple[int, float, str]:
    vel_nan = int(sum(1 for n, v in zip(names, x) if np.isnan(v) and is_velocity_feature(n)))
    nan_count = int(np.isnan(x).sum())
    recall_deg = round(FULL_RECALL - NOVEL_RECALL, 1) if vel_nan > 0 else 0.0
    quality = "full" if nan_count == 0 else "degraded"
    return vel_nan, recall_deg, quality


def _quality_payload(model, x, *, feature_source: str, assumed: list[str] | None = None) -> dict:
    names = model.feature_names
    nan_count, cov = coverage(x)
    vel_nan, recall_deg, quality = velocity_quality(names, x)
    return {
        "nan_features": nan_count,
        "total_features": len(names),
        "coverage_pct": cov,
        "score_quality": quality,
        "velocity_features_nan": vel_nan,
        "recall_degradation_pct": recall_deg,
        "review_threshold": model.review_thr,
        "block_threshold": model.block_thr,
        "feature_source": feature_source,
        "reliable": not assumed,
        "history_assumed": assumed or [],
    }


def _notes(*, vel_nan, nan_count, txn_dt, assumed, ident_filled, txn_id, model) -> list[str]:
    notes = []
    if vel_nan > 0:
        notes.append(
            f"{vel_nan} velocity/history features (uid_count_*, card_amt_vs_mean, "
            "combo_*, *_agg_*) are NaN — these require prior transaction history. "
            f"Fraud recall is ~{NOVEL_RECALL}% vs ~{FULL_RECALL}% with full features "
            f"(~{round(FULL_RECALL - NOVEL_RECALL):.0f}pp degradation). Use /score/{{txn_id}} for a "
            "fully-featured score."
        )
    elif nan_count > 0:
        notes.append("Some features are NaN. Use /score/{txn_id} for a fully-featured score.")
    if txn_dt is None:
        notes.append(
            "TransactionDT was not provided: hour_sin, hour_cos, dow, and Dn_D* "
            "features are NaN. Include TransactionDT for a better estimate."
        )
    if assumed:
        notes.insert(
            0,
            f"{len(assumed)} history features were not supplied and were set to their "
            "'first-time client' value (no past activity). The score is for this "
            "single row only — a typical legitimate payment often scores well below 1%.",
        )
    if ident_filled:
        notes.insert(0, "Empty identity fields were filled from train_identity.csv / test_identity.csv.")
    elif txn_id is not None and txn_id >= 3663549:
        notes.append(
            "This TransactionID is from Kaggle test.csv (not used in training). "
            "There is no label. Most of those rows are legitimate, so a score under the "
            f"{model.review_thr:.0%} review threshold is APPROVE — that is not 99% fraud."
        )
    return notes


def _debug_trace(model, df, x, assumed) -> dict:
    names = model.feature_names
    nan_names = [n for n, v in zip(names, x) if np.isnan(v)]
    contrib = model.booster.predict(
        xgb.DMatrix(x.reshape(1, -1), missing=np.nan),
        pred_contribs=True,
    )[0]
    order = np.argsort(-np.abs(contrib[:-1]))[:15]
    return {
        "review_threshold": model.review_thr,
        "block_threshold": model.block_thr,
        "base_logit": round(float(contrib[-1]), 3),
        "final_logit": round(float(contrib.sum()), 3),
        "has_identity": float(df["has_identity"].iloc[0]),
        "email_same": float(df["email_same"].iloc[0]),
        "nan_features": nan_names,
        "history_features_assumed": assumed,
        "top_shap": [
            {
                "feature": names[i],
                "z_value": None if np.isnan(x[i]) else round(float(x[i]), 3),
                "shap_logit": round(float(contrib[i]), 3),
            }
            for i in order
        ],
    }


def score_from_store(rt: AppRuntime, txn_id: int, x) -> dict:
    x = np.asarray(x, dtype="float32")
    saved = remember(rt, txn_id, x)
    extra = _quality_payload(rt.model, x, feature_source="feature_store", assumed=[])
    extra["reliable"] = True
    extra["velocity_features_nan"] = 0
    extra["recall_degradation_pct"] = 0.0
    extra["note"] = (
        "Scored from the stored full feature vector (history included). "
        "This ID is in the prepared train/val/test split of train_transaction.csv."
    )
    return {**saved, **extra}


def score_raw_row(rt: AppRuntime, row: dict, *, use_store: bool = True, debug: bool = False) -> dict:
    """Preprocess a raw IEEE-CIS dict and score it.

    If the TransactionID is already in the feature store and use_store is True, the
    stored (history-complete) vector is used instead of the cold-start row.
    """
    if not isinstance(row, dict):
        raise RawScoreError("Request body must be a JSON object.")

    prep = rt.preprocessor
    if prep is None:
        raise PreprocessorUnavailable(
            "Preprocessor not loaded. Run the training pipeline so preprocessor.pkl "
            "is in MODEL_DIR, then restart the API (or POST /admin/reload)."
        )

    model = rt.model
    row = normalize_keys(row)
    row, ident_filled = fill_identity(row, rt.identity)

    txn_id = parse_txn_id(row.get("TransactionID"))
    if txn_id is not None and use_store:
        x_store = rt.store.get(txn_id)
        if x_store is not None:
            return score_from_store(rt, txn_id, x_store)

    for c in getattr(prep, "freq_cols", []):
        if c not in prep.freq_maps:
            prep.freq_maps[c] = pd.Series(dtype="float64")

    txn_dt = row.get("TransactionDT")
    try:
        df, assumed = prepare_raw_dataframe(row, prep)
        x = prep.transform(df).values[0].astype("float32")
    except Exception as e:
        raise RawScoreError(f"Preprocessing failed: {e}") from e

    x_aligned = align_to_model(x, prep, model.feature_names)
    extra = _quality_payload(
        model, x_aligned,
        feature_source="raw_row_cold_start" if assumed else "raw_row",
        assumed=assumed,
    )
    notes = _notes(
        vel_nan=extra["velocity_features_nan"],
        nan_count=extra["nan_features"],
        txn_dt=txn_dt,
        assumed=assumed,
        ident_filled=ident_filled,
        txn_id=txn_id,
        model=model,
    )
    saved = remember(rt, txn_id, x_aligned)
    notes.insert(0, f"Saved as transaction {saved['txn_id']}. You can look it up with /score/{saved['txn_id']}.")
    trace = _debug_trace(model, df, x_aligned, assumed) if debug else None
    return {
        **saved,
        **extra,
        "trace": trace,
        "note": " | ".join(notes) if notes else None,
    }

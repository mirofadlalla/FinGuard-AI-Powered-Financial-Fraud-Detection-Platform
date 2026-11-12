"""Helpers for /predict/raw: make a single raw IEEE-CIS row look like a training row.

Why this exists
---------------
Training computes "history" features (velocity counts, amount-vs-usual, seconds since last
transaction, ...) from earlier rows. They are NEVER NaN in training. A single raw row has no
history, and sending NaN for them makes XGBoost follow its arbitrary default branch at every
split -> the score collapses to ~0 for every transaction.

The fix is to fill them with the value training gave to a client seen for the FIRST time
(this is exactly what add_history / rolling_count produce when there is no past row):
    counts            -> 0
    *_amt_vs_mean     -> 1.0
    *_secs_since_last -> -1
    *_agg_*           -> stay NaN (training also leaves them NaN for a first-seen client)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from finguard.ieee import (
    COMBO_COLS,
    D_COLS,
    ID_CAT_COLS,
    ID_NUM_COLS,
    V_COLS,
)

M_MAPPING = {"T": 1.0, "F": 0.0, "M0": 0.0, "M1": 1.0, "M2": 2.0}   # same as ieee.add_features
IDENTITY_KEYS = [f"id_{i:02d}" for i in range(1, 12)] + [f"id_{i}" for i in range(12, 39)] + ["DeviceType", "DeviceInfo"]

COLD_START = {
    "uid_count_1h": 0.0, "uid_count_1d": 0.0, "uid_count_7d": 0.0, "card1_count_1d": 0.0,
    "card_amt_vs_mean": 1.0, "uid_amt_vs_mean": 1.0,
    "card_secs_since_last": -1.0, "uid_secs_since_last": -1.0,
}


def cold_start_defaults(combo_cols):
    d = dict(COLD_START)
    d.update({c: 0.0 for c in combo_cols})
    return d


def normalize_keys(row: dict) -> dict:
    """Test CSVs use 'id-01'; training uses 'id_01'. Accept both."""
    return {k.replace("-", "_"): v for k, v in row.items()}


def is_missing(v) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def has_identity(row: dict) -> float:
    """1.0 only if at least one identity field actually has a value (training: row exists in identity file)."""
    return 1.0 if any(k in row and not is_missing(row[k]) for k in IDENTITY_KEYS) else 0.0


def card_key(row: dict) -> str:
    """Same string add_features builds: str(card1) + '_' + str(addr1) (addr1 is float -> '498.0')."""
    c1, a1 = row.get("card1"), row.get("addr1")
    c1 = "nan" if is_missing(c1) else str(int(c1)) if float(c1).is_integer() else str(c1)
    a1 = "nan" if is_missing(a1) else str(float(a1))
    return f"{c1}_{a1}"


def load_identity_table(data_dir):
    """Index identity rows by TransactionID from train/test identity CSVs if present."""
    frames = []
    data_dir = Path(data_dir)
    for name in ("train_identity.csv", "test_identity.csv"):
        path = data_dir / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.columns = [c.replace("-", "_") for c in df.columns]
        if "TransactionID" not in df.columns:
            continue
        frames.append(df)
    if not frames:
        return None
    ident = pd.concat(frames, ignore_index=True).drop_duplicates("TransactionID")
    return ident.set_index("TransactionID")


def fill_identity(row: dict, ident) -> tuple[dict, bool]:
    """Copy identity fields for this TransactionID when the payload left them empty."""
    if ident is None:
        return row, False
    tid = row.get("TransactionID")
    try:
        tid = int(tid)
    except (TypeError, ValueError):
        return row, False
    if tid not in ident.index:
        return row, False
    rec = ident.loc[tid]
    if isinstance(rec, pd.DataFrame):
        rec = rec.iloc[0]
    filled = False
    for col in IDENTITY_KEYS:
        if col not in rec.index:
            continue
        if not is_missing(row.get(col)):
            continue
        val = rec[col]
        if is_missing(val):
            continue
        if hasattr(val, "item"):
            val = val.item()
        row[col] = val
        filled = True
    return row, filled


def _scalar_missing(df, col) -> bool:
    return col not in df.columns or is_missing(df[col].iloc[0])


def prepare_raw_dataframe(row: dict, prep):
    """Build one training-shaped row from a raw IEEE-CIS dict.

    Returns (df, assumed_history_cols). `df` is ready for Preprocessor.transform.
    """
    row = normalize_keys(dict(row))
    row.pop("isFraud", None)
    txn_dt = row.pop("TransactionDT", None)
    # TransactionID is used only for identity / store lookup; drop before encoding.
    row.pop("TransactionID", None)

    df = pd.DataFrame([{k: (np.nan if v is None else v) for k, v in row.items()}])

    if txn_dt is not None:
        try:
            dt = float(txn_dt)
            hour = (dt / 3600) % 24
            day = dt / 86400
            df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            df["dow"] = int(day) % 7
            for dn in ("Dn_D2", "Dn_D4", "Dn_D10", "Dn_D11", "Dn_D15"):
                d_col = dn.replace("Dn_", "")
                if not _scalar_missing(df, d_col):
                    df[dn] = float(df[d_col].iloc[0]) - day
        except (TypeError, ValueError):
            pass

    if not _scalar_missing(df, "TransactionAmt"):
        df["amt_decimal"] = float(df["TransactionAmt"].iloc[0]) % 1

    p_em, r_em = row.get("P_emaildomain"), row.get("R_emaildomain")
    df["email_same"] = float((not is_missing(p_em)) and (not is_missing(r_em)) and p_em == r_em)

    # Count missing over the FULL training column lists (absent JSON keys = missing).
    df["na_count_D"] = sum(1 for c in D_COLS if _scalar_missing(df, c))
    df["na_count_V"] = sum(1 for c in V_COLS if _scalar_missing(df, c))
    id_cols = ID_NUM_COLS + ID_CAT_COLS + ["DeviceType", "DeviceInfo"]
    df["na_count_id"] = sum(1 for c in id_cols if _scalar_missing(df, c))

    df["has_identity"] = has_identity(row)
    df["card_key"] = card_key(row)

    for mcol in ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"):
        if mcol in df.columns:
            v = df[mcol].iloc[0]
            if isinstance(v, str):
                df[mcol] = df[mcol].map(M_MAPPING)
            df[mcol] = df[mcol].astype("float32")

    assumed = []
    for c, v in cold_start_defaults(COMBO_COLS).items():
        if _scalar_missing(df, c):
            df[c] = v
            assumed.append(c)

    all_prep_cols = (
        list(getattr(prep, "freq_cols", []))
        + list(getattr(prep, "na_cols", []))
        + list(getattr(prep, "num_cols", []))
        + list(getattr(prep, "v_block_reps", []))
    )
    missing = list(dict.fromkeys(c for c in all_prep_cols if c not in df.columns))
    if missing:
        extra = pd.DataFrame(np.nan, index=df.index, columns=missing)
        df = pd.concat([df, extra], axis=1)

    return df, assumed

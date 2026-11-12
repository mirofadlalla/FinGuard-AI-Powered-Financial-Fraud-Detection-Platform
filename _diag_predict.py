"""One-off diagnosis: raw /predict/raw path vs feature-store scores."""
import json, pickle, sys, math
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from finguard.ieee import D_COLS, V_COLS, ID_NUM_COLS, ID_CAT_COLS, COMBO_COLS
from finguard.raw_features import M_MAPPING, card_key, cold_start_defaults, has_identity, normalize_keys
from finguard.serving import Model

with open(ROOT / "artifacts/model/preprocessor.pkl", "rb") as f:
    prep = pickle.load(f)
model = Model(ROOT / "artifacts/model")
d = np.load(ROOT / "data/ieee_ready.npz", allow_pickle=True)
idx = {int(t): i for i, t in enumerate(d["id_test"])}
idx_tr = {int(t): i for i, t in enumerate(d["id_train"])}
idx_va = {int(t): i for i, t in enumerate(d["id_val"])}

print("n_features", len(model.feature_names))
print("review_thr", model.review_thr, "block_thr", model.block_thr)
print("test fraud rate", float(d["y_test"].mean()), "n_test", len(d["y_test"]))
print("train ids", d["id_train"][0], d["id_train"][-1], "test ids", d["id_test"][0], d["id_test"][-1])


def locate(txn_id):
    if txn_id in idx_tr:
        return "train", d["X_train"][idx_tr[txn_id]], int(d["y_train"][idx_tr[txn_id]])
    if txn_id in idx_va:
        return "val", d["X_val"][idx_va[txn_id]], int(d["y_val"][idx_va[txn_id]])
    if txn_id in idx:
        return "test", d["X_test"][idx[txn_id]], int(d["y_test"][idx[txn_id]])
    return None, None, None


def score_x(x):
    x = np.asarray(x, dtype="float32").reshape(1, -1)
    p = float(model.predict(x)[0])
    return p, model.decide(p), int(np.isnan(x).sum())


def serving_transform(row: dict):
    row = normalize_keys(dict(row))
    row.pop("TransactionID", None)
    row.pop("isFraud", None)
    txn_dt = row.pop("TransactionDT", None)
    df = pd.DataFrame([row]).replace({None: np.nan})
    if txn_dt is not None:
        dt = float(txn_dt)
        hour = (dt / 3600) % 24
        day = dt / 86400
        df["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        df["hour_cos"] = math.cos(2 * math.pi * hour / 24)
        df["dow"] = int(day) % 7
        for dn in ("Dn_D2", "Dn_D4", "Dn_D10", "Dn_D11", "Dn_D15"):
            d_col = dn.replace("Dn_", "")
            if d_col in df.columns and not pd.isna(df[d_col].iloc[0]):
                df[dn] = float(df[d_col].iloc[0]) - day
    if "TransactionAmt" in df.columns and not pd.isna(df["TransactionAmt"].iloc[0]):
        df["amt_decimal"] = float(df["TransactionAmt"].iloc[0]) % 1
    p_em, r_em = row.get("P_emaildomain"), row.get("R_emaildomain")
    df["email_same"] = float((not pd.isna(p_em)) and (not pd.isna(r_em)) and p_em == r_em)
    df["na_count_D"] = df[[c for c in D_COLS if c in df.columns]].isna().sum(axis=1)
    df["na_count_V"] = df[[c for c in V_COLS if c in df.columns]].isna().sum(axis=1)
    id_present = [c for c in ID_NUM_COLS + ID_CAT_COLS + ["DeviceType", "DeviceInfo"] if c in df.columns]
    df["na_count_id"] = df[id_present].isna().sum(axis=1) if id_present else np.nan
    df["has_identity"] = has_identity(row)
    df["card_key"] = card_key(row)
    for mcol in ("M1","M2","M3","M4","M5","M6","M7","M8","M9"):
        if mcol in df.columns:
            v = df[mcol].iloc[0]
            if isinstance(v, str):
                df[mcol] = df[mcol].map(M_MAPPING)
            df[mcol] = df[mcol].astype("float32")
    assumed = []
    for c, v in cold_start_defaults(COMBO_COLS).items():
        if c not in df.columns or pd.isna(df[c].iloc[0]):
            df[c] = v
            assumed.append(c)
    all_prep_cols = list(prep.freq_cols) + list(prep.na_cols) + list(prep.num_cols) + list(prep.v_block_reps)
    missing_cols = {c: float("nan") for c in all_prep_cols if c not in df.columns}
    if missing_cols:
        df = pd.concat([df, pd.DataFrame([missing_cols])], axis=1)
    print("  df shape after concat", df.shape, "n_missing_added", len(missing_cols), "assumed", len(assumed))
    x = prep.transform(df)
    print("  transform shape", x.shape)
    feat_map = {n: float(x.values[0][prep.feature_names.index(n)]) if n in prep.feature_names else float("nan")
                for n in model.feature_names}
    x_aligned = np.array([feat_map[n] for n in model.feature_names], dtype="float32")
    return x_aligned, assumed, df


def dump_row(name, path=None, row=None):
    print("\n====", name, "====")
    if path:
        row = json.loads(Path(path).read_text(encoding="utf-8"))
    txn_id = row.get("TransactionID")
    print("txn", txn_id, "isFraud", row.get("isFraud"), "amt", row.get("TransactionAmt"),
          "ProductCD", row.get("ProductCD"), "DT", row.get("TransactionDT"))
    split, x_store, y = locate(int(txn_id)) if txn_id is not None else (None, None, None)
    print("in store?", split, "label", y)
    if x_store is not None:
        p, dec, nans = score_x(x_store)
        print(f"STORE score={p:.6f} {dec} nans={nans}")
    x_raw, assumed, df = serving_transform(row)
    p, dec, nans = score_x(x_raw)
    print(f"RAW   score={p:.6f} {dec} nans={nans} assumed={len(assumed)}")
    # freq encoding sanity
    for c in ["card1", "card4", "ProductCD", "card_key", "addr1"]:
        if c in prep.freq_maps:
            raw_v = df[c].iloc[0] if c in df.columns else "MISSING_COL"
            mapped = prep.freq_maps[c].get(raw_v, "NOT_IN_MAP")
            # try alt types
            alts = []
            try:
                alts.append(("int", prep.freq_maps[c].get(int(float(raw_v)), None)))
            except Exception:
                pass
            try:
                alts.append(("float", prep.freq_maps[c].get(float(raw_v), None)))
            except Exception:
                pass
            print(f"  freq {c}={raw_v!r} type={type(raw_v).__name__} mapped={mapped} alts={alts[:2]}")
    if x_store is not None:
        xs = np.asarray(x_store, dtype="float32")
        both_finite = ~np.isnan(xs) & ~np.isnan(x_raw)
        diffs = np.abs(xs - x_raw)
        diffs[~both_finite] = 0
        top = np.argsort(-diffs)[:12]
        print("  max |store-raw| on finite:", float(diffs.max()), "n_disagree>", 1e-3, int((diffs > 1e-3).sum()))
        print("  nans store", int(np.isnan(xs).sum()), "raw", int(np.isnan(x_raw).sum()))
        only_store_nan = np.isnan(xs) & ~np.isnan(x_raw)
        only_raw_nan = ~np.isnan(xs) & np.isnan(x_raw)
        print("  only_store_nan", int(only_store_nan.sum()), "only_raw_nan", int(only_raw_nan.sum()))
        for i in top:
            if diffs[i] > 1e-4:
                print(f"    {model.feature_names[i]}: store={xs[i]:.4f} raw={x_raw[i]:.4f} d={diffs[i]:.4f}")
        for i in np.where(only_raw_nan)[0][:8]:
            print(f"    RAW NaN {model.feature_names[i]} store={xs[i]}")


dump_row("z45", ROOT / "data/z45.txt")
dump_row("f6", ROOT / "data/f6.txt")

# known fraud from store
fraud_ids = d["id_test"][d["y_test"] == 1][:5]
print("\n==== store frauds (full features) ====")
for tid in fraud_ids:
    p, dec, nans = score_x(d["X_test"][idx[int(tid)]])
    print(int(tid), f"{p:.4f}", dec)

# reconstruct one fraud from CSV if possible
print("\n==== reconstruct first test-split fraud from train_transaction.csv ====")
tid = int(fraud_ids[0])
# read just that row by scanning? usecols + skip is hard; use pandas with filter
# train file is huge — use a chunked scan
found = None
for chunk in pd.read_csv(ROOT / "data/train_transaction.csv", chunksize=50000):
    hit = chunk[chunk["TransactionID"] == tid]
    if len(hit):
        found = hit.iloc[0].replace({np.nan: None}).to_dict()
        break
print("found txn", tid, "isFraud", found.get("isFraud") if found else None)
if found:
    dump_row("csv_fraud", row=found)

# distribution of raw-path scores on a sample of test store rows with history zeroed/cold-started
print("\n==== if we COLD-START history on stored test frauds ====")
names = model.feature_names
vel = [i for i, n in enumerate(names) if any(p in n for p in (
    "uid_count_", "card1_count_", "card_amt_vs_mean", "uid_amt_vs_mean", "combo_", "_agg_",
    "card_secs_since_last", "uid_secs_since_last"))]
print("n vel-ish", len(vel), vel[:5], [names[i] for i in vel[:8]])
X = d["X_test"]
y = d["y_test"]
p_full = model.predict(X)
# can't easily cold-start in z-space; report full-feature stats
print("full mean p legit", float(p_full[y==0].mean()), "fraud", float(p_full[y==1].mean()))
print("full share p<0.01", float((p_full < 0.01).mean()), "fraud recall@0.01", float((p_full[y==1] >= 0.01).mean()))
print("full share p<1e-4", float((p_full < 1e-4).mean()), "median legit", float(np.median(p_full[y==0])))

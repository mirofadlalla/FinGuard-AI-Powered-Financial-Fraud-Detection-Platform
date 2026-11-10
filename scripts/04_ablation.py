"""Step 4: which groups of features REALLY help?
Train XGBoost with all features, then again without each group, and compare on the test set.

Run:  python scripts/04_ablation.py
Differences smaller than ~0.003 PR-AUC are just noise (one seed).
"""
import re
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", message=".*mismatched devices.*")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finguard.metrics import report, precision_threshold

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
(ROOT / "artifacts").mkdir(exist_ok=True)

d = np.load(data_dir / "ieee_ready.npz", allow_pickle=True)

names = list(d["feature_names"])

X_tr, y_tr = d["X_train"], d["y_train"]
X_va, y_va = d["X_val"], d["y_val"]
X_te, y_te, amt_te = d["X_test"], d["y_test"], d["amount_test"]


def group_of(name):
    """Which group does this feature belong to?"""
    if re.fullmatch(r"V\d+", name) or name.startswith(("Vblock_", "na_count_V")):
        return "V columns"
    if name.startswith(("id_", "DeviceType", "DeviceInfo", "has_identity", "na_count_id")):
        return "identity"
    if name.startswith("Dn_"):
        return "D normalized"
    if "_agg_" in name:
        return "client aggregates"
    if name.startswith("combo_"):
        return "combo counts"
    if name.startswith(("uid_", "card_count", "card_amt", "card_secs", "card1_count")):
        return "client history"
    if name.endswith("_freq"):
        return "frequency encoding"
    if re.fullmatch(r"[CDM]\d+(_na)?", name):
        return "C / D / M columns"
    return "base"


groups = np.array([group_of(n) for n in names])
print(pd.Series(groups).value_counts().to_string(), "\n")


def pick_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


device = pick_device()
ratio = (y_tr == 0).sum() / (y_tr == 1).sum()


def run(mask, label):
    """Train XGBoost on the chosen columns and report on the test set."""
    model = XGBClassifier(n_estimators=1500, max_depth=8, learning_rate=0.1, subsample=0.8,
                          colsample_bytree=0.5, scale_pos_weight=ratio, eval_metric="aucpr",
                          early_stopping_rounds=50, device=device, random_state=0)
    model.fit(X_tr[:, mask], y_tr, eval_set=[(X_va[:, mask], y_va)], verbose=False)
    p_val = model.predict_proba(X_va[:, mask])[:, 1]
    p_test = model.predict_proba(X_te[:, mask])[:, 1]
    thr = precision_threshold(y_va, p_val, 0.5)
    r = report(y_te, p_test, amt_te, thr, label)
    row = {"experiment": label, "n_features": int(mask.sum()), "PR-AUC": r["PR-AUC"],
           "ROC-AUC": r["ROC-AUC"], "recall@P50": r["recall"], "precision": r["precision"]}
    print(f"{label:34s} feats={row['n_features']:4d}  PR-AUC={row['PR-AUC']:.4f}  ROC-AUC={row['ROC-AUC']:.4f}  recall={row['recall@P50']:.3f}")
    return row


rows = [run(np.ones(len(names), dtype=bool), "ALL features")]
for g in sorted(set(groups) - {"base"}):
    rows.append(run(groups != g, f"without {g}"))
rows.append(run(groups == "base", "ONLY base features"))

result = pd.DataFrame(rows)
full = result.loc[0, "PR-AUC"]
result["PR-AUC change"] = (result["PR-AUC"] - full).round(4)
result.to_csv(ROOT / "artifacts" / "ablation.csv", index=False)
print("\n", result.to_string(index=False))
print("\nA big negative change when a group is removed = that group helps a lot.")

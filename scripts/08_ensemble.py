"""Step 8: final comparison table + ensembles of the models.

Uses the predictions saved by 02 (XGBoost), 03 (MLP) and 07 (GNN), averaged over seeds.
Ensemble = average of the models' PERCENTILES (the models use different probability scales,
so we first turn each score into 'what share of the validation scores is below it').

Run:  python scripts/08_ensemble.py
"""
import sys
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation.metrics import no_model_cost, operating_points, report

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
art_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "artifacts"

d = np.load(data_dir / "ieee_ready.npz", allow_pickle=True)
y_va, y_te, amt_va, amt_te = d["y_val"], d["y_test"], d["amount_val"], d["amount_test"]

files = {"XGBoost": "preds_xgb.npz", "MLP": "preds_mlp.npz", "GNN": "preds_gnn.npz"}
preds = {name: np.load(art_dir / f) for name, f in files.items() if (art_dir / f).exists()}
print("models found:", list(preds), "\n")
if len(preds) < 2:
    sys.exit("Need at least 2 prediction files. Run scripts 02, 03 and 07 first.")


def to_percentile(val_scores, scores):
    """Share of validation scores that are <= each score (same scale for every model)."""
    return np.searchsorted(np.sort(val_scores), scores, side="right") / len(val_scores)


rows = []
names = list(preds)
for size in range(1, len(names) + 1):
    for combo in combinations(names, size):
        p_val = np.mean([to_percentile(preds[m]["val"], preds[m]["val"]) for m in combo], axis=0)
        p_test = np.mean([to_percentile(preds[m]["val"], preds[m]["test"]) for m in combo], axis=0)
        for op_name, thr in operating_points(y_va, p_val, amt_va).items():
            rows.append(report(y_te, p_test, amt_te, thr, f"{' + '.join(combo)} @{op_name}"))

table = pd.DataFrame(rows)
print(f"Cost if we use NO model: {no_model_cost(y_te, amt_te):,.0f}\n")
print(table.to_string(index=False))
table.to_csv(art_dir / "ensemble_results.csv", index=False)

# short table for the README: one row per model, at the P50 threshold (precision about 50%)
short = table[table["model"].str.endswith("@P50-thr")][["model", "PR-AUC", "ROC-AUC", "precision", "recall"]]
print("\nFor the README (threshold chosen for ~50% precision):\n")
try:
    print(short.to_markdown(index=False))          # needs the 'tabulate' package
except ImportError:
    print(short.to_string(index=False))

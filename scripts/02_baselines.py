"""Step 2: baselines (Logistic Regression + XGBoost), XGBoost repeated with several seeds.

Run:  python scripts/02_baselines.py               (3 seeds)
      python scripts/02_baselines.py data 1        (quick run: 1 seed)
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", message=".*mismatched devices.*")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finguard.metrics import operating_points, report, summarize, no_model_cost, FP_COST

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
(ROOT / "artifacts").mkdir(exist_ok=True)

d = np.load(data_dir / "ieee_ready.npz", allow_pickle=True)

names = list(d["feature_names"])

# --- load the data ---
X_tr, y_tr = d["X_train"], d["y_train"]
X_va, y_va, amt_va = d["X_val"], d["y_val"], d["amount_val"]
X_te, y_te, amt_te = d["X_test"], d["y_test"], d["amount_test"]


def pick_device():
    """Use the GPU for XGBoost if PyTorch can see one."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


device = pick_device()
print("XGBoost device:", device)
ratio = (y_tr == 0).sum() / (y_tr == 1).sum()    # legit transactions for each fraud

runs = []
def evaluate(model, name, seed, fix_nan=False):
    # LogReg cannot handle NaN, so there we replace NaN by 0 (= the mean after scaling)
    fix = (lambda x: np.nan_to_num(x, nan=0.0)) if fix_nan else (lambda x: x)
    p_val = model.predict_proba(fix(X_va))[:, 1] # only the fraud probability second column
    p_test = model.predict_proba(fix(X_te))[:, 1]
    
    for op_name, thr in operating_points(y_va, p_val, amt_va).items():
        row = report(y_te, p_test, amt_te, thr, f"{name} @{op_name}")
        row["seed"] = seed
        runs.append(row)


# --- Logistic Regression (no randomness, so one run) ---
print("Training LogReg ...")
logreg = LogisticRegression(class_weight="balanced", max_iter=500)
logreg.fit(np.nan_to_num(X_tr, nan=0.0), y_tr)
evaluate(logreg, "LogReg", 0, fix_nan=True)

# --- XGBoost, one run per seed ---
for seed in range(n_seeds):
    print(f"Training XGBoost, seed {seed} ...")
    xgb = XGBClassifier(n_estimators=6000, max_depth=9, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.4, scale_pos_weight=ratio, eval_metric="aucpr",
                        early_stopping_rounds=100, device=device, random_state=seed)
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    print(f"   best round: {xgb.best_iteration}")
    evaluate(xgb, "XGBoost", seed)

runs = pd.DataFrame(runs)
runs.to_csv(ROOT / "artifacts" / "baseline_all_runs.csv", index=False)
summary = summarize(runs)
summary.to_csv(ROOT / "artifacts" / "baseline_results.csv", index=False)

print(f"\nCost if we use NO model (all fraud goes through): {no_model_cost(y_te, amt_te):,.0f}")
print(f"(a false alarm costs {FP_COST:.0f}; thresholds were chosen on the validation set)\n")
print(summary.to_string(index=False))

top = sorted(zip(names, xgb.feature_importances_), key=lambda x: -x[1])[:20]
print("\nTop 20 XGBoost features (last seed):")
for n, v in top:
    print(f"  {n:32s} {v:.3f}")

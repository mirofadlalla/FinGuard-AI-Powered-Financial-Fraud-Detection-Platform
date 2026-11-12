"""Step 5: train the FINAL model, track it with MLflow, save the bundle for the API.

What it does:
  1. trains XGBoost (early stopping on the validation set)
  2. chooses two thresholds on validation: REVIEW (lowest business cost) and BLOCK (precision >= 80%)
  3. logs params / metrics / artifacts to MLflow and registers the model
  4. promotion gate: the new model becomes "production" only if its test PR-AUC is not worse
     than the current production model. Only then the API folder artifacts/model is updated.

Run:  python scripts/05_train_final.py
      python scripts/05_train_final.py data quick      (tiny model, only to test that everything works)
"""
import json
import os
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", message=".*mismatched devices.*")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finguard.drift import make_reference
from finguard.metrics import cost_threshold, precision_threshold

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
quick = len(sys.argv) > 2 and sys.argv[2] == "quick"
MODEL_NAME = "finguard-xgb"

d = np.load(data_dir / "ieee_ready.npz", allow_pickle=True)
names = [str(n) for n in d["feature_names"]]
X_tr, y_tr = d["X_train"], d["y_train"]
X_va, y_va, amt_va = d["X_val"], d["y_val"], d["amount_val"]
X_te, y_te, amt_te = d["X_test"], d["y_test"], d["amount_test"]


def pick_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


params = {"n_estimators": 50 if quick else 5000, "max_depth": 9, "learning_rate": 0.05, "subsample": 0.8,
          "colsample_bytree": 0.4, "scale_pos_weight": float((y_tr == 0).sum() / (y_tr == 1).sum()),
          "eval_metric": "aucpr", "early_stopping_rounds": 100, "random_state": 0}

#  MLflow setup (local sqlite file by default, or a server if MLFLOW_TRACKING_URI is set) 
uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///" + (ROOT / "mlflow.db").as_posix())
mlflow.set_tracking_uri(uri)
if uri.startswith("sqlite") and mlflow.get_experiment_by_name("finguard") is None:
    mlflow.create_experiment("finguard", artifact_location=(ROOT / "mlartifacts").as_uri())
mlflow.set_experiment("finguard")

with mlflow.start_run() as run:
    run_id = run.info.run_id
    device = pick_device()
    print("XGBoost device:", device, "| MLflow:", uri)

    #  1. train 
    clf = XGBClassifier(device=device, **params)
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    p_val = clf.predict_proba(X_va)[:, 1]
    p_test = clf.predict_proba(X_te)[:, 1]
    print("best round:", clf.best_iteration)

    #  2. thresholds (chosen on validation only) 
    review_thr = cost_threshold(y_va, p_val, amt_va)[0]
    block_thr = precision_threshold(y_va, p_val, 0.80)

    #  3. test results 
    is_blocked = p_test >= block_thr
    is_review = (p_test >= review_thr) & ~is_blocked
    flagged = is_blocked | is_review
    metrics = {
        "test_pr_auc": average_precision_score(y_te, p_test),
        "test_roc_auc": roc_auc_score(y_te, p_test),
        "review_recall": float(y_te[flagged].sum() / y_te.sum()),            # fraud we send to review or block
        "review_precision": float(y_te[flagged].mean()) if flagged.any() else 0.0,
        "block_precision": float(y_te[is_blocked].mean()) if is_blocked.any() else 0.0,
        "share_flagged": float(flagged.mean()),
        "share_blocked": float(is_blocked.mean()),
        "cost_with_model": float(amt_te[(y_te == 1) & ~flagged].sum() + 5.0 * ((y_te == 0) & flagged).sum()),
        "cost_no_model": float(amt_te[y_te == 1].sum()),
        "best_iteration": float(clf.best_iteration),
    }
    for k, v in metrics.items():
        print(f"  {k:18s} {v:,.4f}")

    #  4. SHAP: which features matter most overall 
    booster = clf.get_booster()
    sample = X_te[np.random.default_rng(0).choice(len(X_te), min(2000, len(X_te)), replace=False)]
    shap_values = booster.predict(xgb.DMatrix(sample, missing=np.nan), pred_contribs=True)[:, :-1]
    shap_global = (pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
                   .sort_values("mean_abs_shap", ascending=False))

    #  5. save the bundle (model + settings + drift reference) 
    version = datetime.now().strftime("%Y%m%d-%H%M") + "-" + run_id[:6]
    bundle = ROOT / "artifacts" / "model_versions" / version
    bundle.mkdir(parents=True, exist_ok=True)
    booster.set_param({"device": "cpu"})                 # so the model loads on any machine
    booster.save_model(str(bundle / "model.ubj"))
    meta = {"version": version, "feature_names": names, "mlflow_run_id": run_id,
            "thresholds": {"review": float(review_thr), "block": float(block_thr)},
            "val_mean_prob": float(p_val.mean()), "metrics": metrics, "params": params}
    (bundle / "meta.json").write_text(json.dumps(meta, indent=2))
    (bundle / "reference.json").write_text(json.dumps(make_reference(X_tr, names, scores=p_val)))
    shap_global.to_csv(bundle / "shap_global.csv", index=False)
    # Save the preprocessor so the API can accept raw IEEE-CIS transactions.
    # Load from data/preprocessor.pkl (written by 01_prepare_data.py) so we get the
    # real fitted mean/std and freq_maps rather than the hollow shell that
    # save_preprocessor.py used to build by inferring from ieee_ready.npz.
    import pickle
    _prep_src = data_dir / "preprocessor.pkl"
    if not _prep_src.exists():
        raise FileNotFoundError(
            f"{_prep_src} not found.  Run  python scripts/01_prepare_data.py  first "
            "so the fitted Preprocessor (with real mean/std and freq_maps) is available."
        )
    with open(_prep_src, "rb") as _f:
        prep = pickle.load(_f)
    with open(bundle / "preprocessor.pkl", "wb") as _f:
        pickle.dump(prep, _f, protocol=4)
    print(f"Saved preprocessor → preprocessor.pkl  (freq_maps: {len(prep.freq_maps)}, "
          f"num_cols: {len(prep.num_cols)})")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = shap_global.head(20).iloc[::-1]
        plt.figure(figsize=(8, 6))
        plt.barh(top["feature"], top["mean_abs_shap"])
        plt.xlabel("mean |SHAP value| (log-odds)")
        plt.title("Top 20 features")
        plt.tight_layout()
        plt.savefig(bundle / "shap_global.png", dpi=120)
        plt.close()
    except Exception as e:
        print("(no SHAP plot:", e, ")")

    #  6. log to MLflow + register 
    mlflow.log_params({k: v for k, v in params.items()})
    mlflow.log_params({"n_features": len(names), "review_thr": round(review_thr, 5), "block_thr": round(block_thr, 5),
                       "device": device})
    mlflow.log_metrics(metrics)
    mlflow.log_artifacts(str(bundle), artifact_path="bundle")
    mlflow.xgboost.log_model(clf, artifact_path="model", registered_model_name=MODEL_NAME)

#  7. promotion gate 
client = MlflowClient()
new_version = [v for v in client.search_model_versions(f"name='{MODEL_NAME}'") if v.run_id == run_id][0].version
try:
    current = client.get_model_version_by_alias(MODEL_NAME, "production")
    old_pr_auc = client.get_run(current.run_id).data.metrics.get("test_pr_auc", 0.0)
except Exception:
    current, old_pr_auc = None, None

if current is None or metrics["test_pr_auc"] >= old_pr_auc:
    client.set_registered_model_alias(MODEL_NAME, "production", new_version)
    live = ROOT / "artifacts" / "model"                  # the folder the API loads
    if live.exists():
        shutil.rmtree(live)
    shutil.copytree(bundle, live)
    print(f"\nPROMOTED: {MODEL_NAME} v{new_version} is now production "
          f"(test PR-AUC {metrics['test_pr_auc']:.4f}" + ("" if old_pr_auc is None else f", before {old_pr_auc:.4f}") + ")")
else:
    client.set_registered_model_alias(MODEL_NAME, "challenger", new_version)
    print(f"\nNOT promoted: new PR-AUC {metrics['test_pr_auc']:.4f} < production {old_pr_auc:.4f}. Saved as 'challenger'.")
print("Bundle:", bundle)

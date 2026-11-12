"""ZenML Step: Export Bundle, Track with MLflow & Automated Production Promotion Gate."""

import json
import os
import pickle
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
from xgboost import XGBClassifier
from zenml import step

from monitoring.drift import make_reference


@step
def register_model_step(
    clf: XGBClassifier,
    metadata: Dict[str, Any],
    metrics: Dict[str, float],
    thresholds: Dict[str, float],
    data_dir: str = "data",
    model_name: str = "finguard-xgb",
) -> str:
    """Export artifacts (model.ubj, meta.json, reference.json, preprocessor.pkl),
    log to MLflow Tracking, and apply the production promotion gate.

    The gate is unconditional: every successful run promotes the new model.
    For a PR-AUC gate, compare metrics["test_pr_auc"] against the incumbent
    meta.json before calling shutil.copytree.

    Returns the version string (e.g. '20261112-1430-a3f1b2').
    """
    root = Path(__file__).resolve().parents[2]
    path = Path(data_dir)
    d = np.load(path / "ieee_ready.npz", allow_pickle=True)
    names = metadata["feature_names"]
    X_tr = d["X_train"]

    run_ts = datetime.now().strftime("%Y%m%d-%H%M")
    short_hash = uuid.uuid4().hex[:6]
    version = f"{run_ts}-{short_hash}"
    bundle = root / "artifacts" / "model_versions" / version
    bundle.mkdir(parents=True, exist_ok=True)

    # ── Save model ────────────────────────────────────────────────────────────
    booster = clf.get_booster()
    booster.set_param({"device": "cpu"})
    booster.save_model(str(bundle / "model.ubj"))

    # ── Write meta.json ───────────────────────────────────────────────────────
    meta = {
        "version": version,
        "feature_names": names,
        "thresholds": thresholds,
        "val_mean_prob": metrics.get("val_mean_prob", 0.0),
        "metrics": metrics,
        "params": metadata["params"],
    }
    (bundle / "meta.json").write_text(json.dumps(meta, indent=2))

    # ── Write drift reference ─────────────────────────────────────────────────
    (bundle / "reference.json").write_text(json.dumps(make_reference(X_tr, names)))

    # ── Copy preprocessor ─────────────────────────────────────────────────────
    prep_src = path / "preprocessor.pkl"
    if prep_src.exists():
        with open(prep_src, "rb") as f:
            prep = pickle.load(f)
        with open(bundle / "preprocessor.pkl", "wb") as f:
            pickle.dump(prep, f, protocol=4)

    # ── MLflow tracking (optional — works without a tracking server) ──────────
    try:
        import mlflow

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlflow.db")
        mlflow.set_tracking_uri(
            tracking_uri if tracking_uri.startswith("http") else f"sqlite:///{tracking_uri}"
        )
        mlflow.set_experiment(model_name)
        with mlflow.start_run(run_name=version):
            mlflow.log_params(metadata["params"])
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(bundle / "meta.json"))
        print(f"MLflow: logged run '{version}' to experiment '{model_name}'")
    except Exception as exc:
        print(f"MLflow logging skipped: {exc}")

    # ── Production promotion gate: update artifacts/model ────────────────────
    live = root / "artifacts" / "model"
    if live.exists():
        shutil.rmtree(live)
    shutil.copytree(bundle, live)

    print(f"Exported model bundle -> {bundle}")
    print(f"Promoted {model_name} version {version} -> {live}")
    return version

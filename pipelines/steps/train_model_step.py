"""ZenML Step: Train XGBoost Production Model with Early Stopping."""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from zenml import step


def pick_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@step
def train_model_step(
    data_dir: str = "data",
    n_estimators: int = 5000,
    max_depth: int = 9,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 100,
    quick: bool = False,
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """Trains production XGBoost classifier on prepared data with validation early stopping.
    
    Returns the trained model and run metadata.
    """
    path = Path(data_dir)
    npz_file = path / "ieee_ready.npz"
    if not npz_file.exists():
        raise FileNotFoundError(f"Prepared data not found at {npz_file}. Run prepare_data_step first.")
        
    print(f"Executing ZenML Step: Training XGBoost model from {npz_file}...")
    d = np.load(npz_file, allow_pickle=True)
    X_tr, y_tr = d["X_train"], d["y_train"]
    X_va, y_va = d["X_val"], d["y_val"]
    
    device = pick_device()
    n_est = 50 if quick else n_estimators
    scale_pos = float((y_tr == 0).sum() / (y_tr == 1).sum())
    
    params = {
        "n_estimators": n_est,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.4,
        "scale_pos_weight": scale_pos,
        "eval_metric": "aucpr",
        "early_stopping_rounds": early_stopping_rounds,
        "random_state": 0,
        "device": device,
    }
    
    clf = XGBClassifier(**params)
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    
    best_iteration = int(clf.best_iteration) if hasattr(clf, "best_iteration") and clf.best_iteration is not None else n_est
    print(f"Training completed. Best iteration: {best_iteration} (device: {device})")
    
    train_metadata = {
        "params": params,
        "best_iteration": best_iteration,
        "device": device,
        "n_features": int(X_tr.shape[1]),
        "train_rows": int(len(X_tr)),
        "feature_names": [str(n) for n in d["feature_names"]],
    }
    
    return clf, train_metadata

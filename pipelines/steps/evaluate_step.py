"""ZenML Step: Evaluate Model & Calibrate Validation Thresholds."""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from zenml import step

from evaluation.metrics import cost_threshold, precision_threshold


@step
def evaluate_step(
    clf: XGBClassifier,
    metadata: Dict[str, Any],
    data_dir: str = "data"
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Evaluates the model on validation & out-of-time test set.
    Calibrates REVIEW and BLOCK thresholds on the validation set.
    
    Returns (metrics, thresholds).
    """
    path = Path(data_dir)
    d = np.load(path / "ieee_ready.npz", allow_pickle=True)
    X_va, y_va, amt_va = d["X_val"], d["y_val"], d["amount_val"]
    X_te, y_te, amt_te = d["X_test"], d["y_test"], d["amount_test"]
    
    p_val = clf.predict_proba(X_va)[:, 1]
    p_test = clf.predict_proba(X_te)[:, 1]
    
    # Thresholds tuned strictly on validation set
    review_thr = float(cost_threshold(y_va, p_val, amt_va)[0])
    block_thr = float(precision_threshold(y_va, p_val, 0.80))
    
    # Out-of-time test evaluation
    is_blocked = p_test >= block_thr
    is_review = (p_test >= review_thr) & ~is_blocked
    flagged = is_blocked | is_review
    
    metrics = {
        "test_pr_auc": float(average_precision_score(y_te, p_test)),
        "test_roc_auc": float(roc_auc_score(y_te, p_test)),
        "review_recall": float(y_te[flagged].sum() / y_te.sum()),
        "review_precision": float(y_te[flagged].mean()) if flagged.any() else 0.0,
        "block_precision": float(y_te[is_blocked].mean()) if is_blocked.any() else 0.0,
        "share_flagged": float(flagged.mean()),
        "share_blocked": float(is_blocked.mean()),
        "cost_with_model": float(amt_te[(y_te == 1) & ~flagged].sum() + 5.0 * ((y_te == 0) & flagged).sum()),
        "cost_no_model": float(amt_te[y_te == 1].sum()),
        "val_mean_prob": float(p_val.mean()),
    }
    
    thresholds = {
        "review": review_thr,
        "block": block_thr,
    }
    
    print("\nZenML Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:,.4f}")
        
    return metrics, thresholds

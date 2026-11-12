import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

FP_COST = 5.0   # assumption: one false alarm costs $5 (review time + annoyed customer)


#  ways to choose the threshold (always choose on VAL, then use on TEST) 
def best_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
    return float(th[int(np.nanargmax(f1[:-1]))])


def cost_threshold(y, p, amount, fp_cost=FP_COST):
    """Threshold with the lowest money loss: missed fraud = lost amount, false alarm = fp_cost.

    Searches a grid of 400 candidate thresholds evenly spaced in [0.01, 0.99]
    (probability space, not quantile space).  Using quantiles of p would start
    the search at the 50th-percentile score, which for a well-calibrated model
    on a ~3.5% fraud dataset is a tiny number (~0.00004), producing an absurdly
    low review threshold that flags ~16% of all transactions.
    """
    best_t, best_c = 0.5, np.inf
    for t in np.linspace(0.01, 0.99, 400):
        flagged = p >= t
        cost = amount[(y == 1) & ~flagged].sum() + fp_cost * ((y == 0) & flagged).sum()
        if cost < best_c:
            best_t, best_c = float(t), float(cost)
    return best_t, best_c


def precision_threshold(y, p, target=0.5):
    """Lowest threshold where precision is still at least `target` (= the most recall we can get)."""
    pr, rc, th = precision_recall_curve(y, p)
    ok = np.where(pr[:-1] >= target)[0]
    return float(th[ok[0]]) if len(ok) else 1.0


def operating_points(y_val, p_val, amount_val):
    """Three ways to pick a threshold, all chosen on the validation set."""
    return {"F1-thr": best_f1_threshold(y_val, p_val),
            "cost-thr": cost_threshold(y_val, p_val, amount_val)[0],
            "P50-thr": precision_threshold(y_val, p_val, 0.5)}


#  results 
def no_model_cost(y, amount):
    """If we never flag anything, every fraud goes through."""
    return float(amount[y == 1].sum())


def report(y, p, amount, thr, name="", fp_cost=FP_COST):
    flagged = p >= thr
    tp = int(((y == 1) & flagged).sum())
    fp = int(((y == 0) & flagged).sum())
    fn = int(((y == 1) & ~flagged).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    cost = float(amount[(y == 1) & ~flagged].sum() + fp_cost * fp)
    return {"model": name,
            "PR-AUC": round(average_precision_score(y, p), 4),
            "ROC-AUC": round(roc_auc_score(y, p), 4),
            "precision": round(precision, 3), "recall": round(recall, 3),
            "F1": round(2 * precision * recall / max(precision + recall, 1e-9), 3),
            "FP": fp, "FN": fn, "cost": round(cost),
            "saved%": round(100 * (1 - cost / no_model_cost(y, amount)), 1)}


def summarize(runs):
    """Mean ± std over seeds, one row per model."""
    def fmt(g, col, nice):
        std = g[col].std(ddof=1) if len(g) > 1 else 0.0
        return f"{g[col].mean():{nice}} ± {std:{nice}}"

    rows = []
    for model, g in runs.groupby("model", sort=False):
        rows.append({"model": model, "PR-AUC": fmt(g, "PR-AUC", ".3f"), "ROC-AUC": fmt(g, "ROC-AUC", ".3f"),
                     "precision": fmt(g, "precision", ".3f"), "recall": fmt(g, "recall", ".3f"),
                     "F1": fmt(g, "F1", ".3f"), "cost": fmt(g, "cost", ",.0f"),
                     "saved%": fmt(g, "saved%", ".1f")})
    return pd.DataFrame(rows)

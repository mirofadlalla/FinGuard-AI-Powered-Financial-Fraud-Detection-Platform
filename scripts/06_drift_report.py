"""Step 6: drift report.

We pretend the test data arrives in 4 consecutive periods (like 4 weeks in production) and ask:
  * DATA drift:  did the inputs change?      -> PSI of every feature
  * SCORE drift: did the model scores change? -> PSI of the scores
  * MODEL drift: is the model still good?     -> PR-AUC per period (labels arrive later in real life)

Run:  python scripts/06_drift_report.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finguard.drift import feature_psi, level, psi
from finguard.serving import Model

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
model_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "artifacts" / "model"

model = Model(model_dir)
reference = json.loads((model_dir / "reference.json").read_text())
names = model.feature_names
d = np.load(data_dir / "ieee_ready.npz", allow_pickle=True)

# periods: validation first (the "calm" baseline), then the test data cut in 4 time slices
periods = {"validation": (d["X_val"], d["y_val"])}
for i, rows in enumerate(np.array_split(np.arange(len(d["y_test"])), 4), 1):
    periods[f"test period {i}"] = (d["X_test"][rows], d["y_test"][rows])

summary, last_psi = [], None
for name, (X, y) in periods.items():
    scores = model.predict(X)
    feature_drift = feature_psi(reference, X, names)
    score_drift = psi(reference["__score__"], scores)
    flagged = scores >= model.review_thr
    summary.append({"period": name, "rows": len(y), "fraud_rate": round(float(y.mean()), 4),
                    "mean_score": round(float(scores.mean()), 4), "share_flagged": round(float(flagged.mean()), 4),
                    "recall_at_review_thr": round(float(y[flagged].sum() / max(y.sum(), 1)), 3),
                    "pr_auc": round(float(average_precision_score(y, scores)), 4) if y.sum() else None,
                    "score_psi": round(score_drift, 4), "score_drift": level(score_drift),
                    "features_big_shift": sum(v > 0.25 for v in feature_drift.values()),
                    "features_small_shift": sum(0.10 <= v < 0.25 for v in feature_drift.values())})
    last_psi = feature_drift

report = pd.DataFrame(summary)
print(report.to_string(index=False))

top = pd.Series(last_psi).sort_values(ascending=False).head(15)
print("\nMost drifted features in the last period:")
for feature, value in top.items():
    print(f"  {feature:34s} PSI {value:.3f}  ({level(value)})")

# simple rule: retrain when the score distribution or the quality changed a lot
base = report.loc[0, "pr_auc"]
latest = report.iloc[-1]
reasons = []
if latest["score_psi"] > 0.25:
    reasons.append(f"score PSI {latest['score_psi']} > 0.25")
if base and latest["pr_auc"] is not None and latest["pr_auc"] < 0.8 * base:
    reasons.append(f"PR-AUC fell from {base} to {latest['pr_auc']} (more than 20%)")
print("\nRECOMMENDATION:", ("RETRAIN - " + "; ".join(reasons)) if reasons else "model looks stable, keep monitoring")

(ROOT / "artifacts").mkdir(exist_ok=True)
report.to_csv(ROOT / "artifacts" / "drift_report.csv", index=False)
top.rename("psi").to_csv(ROOT / "artifacts" / "drift_top_features.csv")

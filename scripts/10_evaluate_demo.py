"""Practical demo: score holdout transactions across fraud / legit and risk levels.

Uses the production XGBoost bundle (artifacts/model) and the time-split test set
in data/ieee_ready.npz. No API server required.

Run (from the project root):
    python scripts/10_evaluate_demo.py
    python scripts/10_evaluate_demo.py data artifacts/model
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finguard.evaluation import pick_showcase, score_split, summary_counts
from finguard.metrics import no_model_cost
from finguard.serving import Model

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
model_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "artifacts" / "model"
npz_path = data_dir / "ieee_ready.npz"


def _die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def main() -> None:
    if not npz_path.exists():
        _die(f"Missing {npz_path}. Run: python scripts/01_prepare_data.py")
    if not (model_dir / "model.ubj").exists():
        _die(f"Missing model in {model_dir}. Run: python scripts/05_train_final.py")

    model = Model(model_dir)
    d = np.load(npz_path, allow_pickle=True, mmap_mode="r")
    X = np.asarray(d["X_test"], dtype="float32")
    y = np.asarray(d["y_test"])
    ids = np.asarray(d["id_test"])
    amounts = np.asarray(d["amount_test"])

    print("FinGuard evaluation demo")
    print(f"  model     {model.version}")
    print(f"  features  {len(model.feature_names)}")
    print(f"  test rows {len(y):,}")
    print(f"  review >= {model.review_thr:.4f}  -> FRAUD_REVIEW")
    print(f"  block  >= {model.block_thr:.4f}  -> FRAUD_BLOCK")
    print()

    probs, decisions = score_split(model, X)
    pr_auc = average_precision_score(y, probs)
    roc = roc_auc_score(y, probs)
    stats = summary_counts(y, decisions)
    baseline = no_model_cost(y, amounts)
    flagged = decisions != "APPROVE"
    cost = float(amounts[(y == 1) & ~flagged].sum() + 5.0 * ((y == 0) & flagged).sum())
    saved = (baseline - cost) / baseline if baseline else 0.0

    print("Holdout metrics (time-split test set)")
    print(f"  PR-AUC              {pr_auc:.4f}   (random ~ fraud rate {y.mean():.3f})")
    print(f"  ROC-AUC             {roc:.4f}")
    print(f"  fraud caught        {stats['caught']:,} / {stats['fraud']:,}  recall {stats['recall']:.1%}")
    print(f"  missed fraud        {stats['missed']:,}")
    print(f"  false alarms        {stats['false_alarms']:,}  FPR {stats['false_positive_rate']:.1%}")
    print(f"  estimated $ saved   {saved:.1%} of no-model loss (${baseline:,.0f})")
    print()

    rows = pick_showcase(ids, y, probs, amounts, decisions, n_per_bucket=2, seed=0)
    print("Showcase - same model, different transaction types")
    print(f"{'bucket':<30} {'txn_id':>10} {'label':>6} {'prob':>8} {'decision':<14} {'amount':>10}")
    print("-" * 84)
    last = None
    for r in rows:
        if r.bucket != last:
            if last is not None:
                print()
            last = r.bucket
        print(
            f"{r.bucket:<30} {r.txn_id:>10} {r.label:>6} {r.fraud_prob:>8.4f} "
            f"{r.decision:<14} {r.amount:>10.2f}"
        )

    print()
    print("How to read this")
    print("  FRAUD + BLOCK/REVIEW  - the model ranked a labeled fraud above a threshold.")
    print("  FRAUD + APPROVE       - a miss: score stayed below review. Rare-event models do this.")
    print("  LEGIT + APPROVE       - typical payment; low probability.")
    print("  LEGIT + REVIEW/BLOCK  - false alarm; analysts would look at SHAP factors.")
    print()
    print("Replay any row through the API:")
    if rows:
        sample = rows[0].txn_id
        print(f"  curl http://localhost:8000/score/{sample}")
        print(f'  curl -X POST http://localhost:8000/investigate -H "Content-Type: application/json" \\')
        print(f'       -d \'{{"question": "Why was transaction {sample} flagged?"}}\'')


if __name__ == "__main__":
    main()

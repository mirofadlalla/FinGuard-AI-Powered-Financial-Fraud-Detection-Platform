# FinGuard v2 — Fraud Detection on IEEE-CIS (work in progress)

Real data: Kaggle IEEE-CIS Fraud Detection (590K transactions, 3.5% fraud).
Compares Logistic Regression, XGBoost and a PyTorch residual MLP, then serves the best model
with an explainable API. Metrics: PR-AUC, ROC-AUC, precision / recall / F1 and business cost. Accuracy is never used.

```
Transaction -> features -> XGBoost -> fraud probability -> APPROVE / FRAUD_REVIEW / FRAUD_BLOCK
                                                        -> SHAP top factors (only for flagged ones)
                                                        -> assistant: "Why was #123 flagged?"
FastAPI + PostgreSQL (decisions) + Redis (cache) + MLflow (experiments, registry) + PSI drift report
```

## Setup
1. Put `train_transaction.csv` and `train_identity.csv` inside `data/`
2. `pip install -r requirements.txt` (for the GPU use the CUDA command inside that file)

## Part 1 - models and features
```
python scripts/01_prepare_data.py     # features + time split  -> data/ieee_ready.npz
python scripts/02_baselines.py        # LogReg + XGBoost (3 seeds)
python scripts/03_train_mlp.py        # PyTorch MLP, 4 imbalance strategies x 5 seeds
python scripts/04_ablation.py         # which feature groups really help? (XGBoost, 1 seed)
```
Add a number at the end for a quick run, e.g. `python scripts/03_train_mlp.py data 1` (1 seed).

## Part 2 - tracking, serving, monitoring
```
python scripts/05_train_final.py      # final model -> MLflow (+ registry, promotion gate) -> artifacts/model
python scripts/06_drift_report.py     # PSI drift report over 4 periods

# API (locally)
uvicorn --app-dir src finguard.api:app --reload      # docs: http://localhost:8000/docs
# or everything in Docker (API + PostgreSQL + Redis + MLflow UI)
docker compose up --build
```
Try it:
```
curl localhost:8000/health
curl localhost:8000/score/<TransactionID>            # ids of the test split are in the data
curl -X POST localhost:8000/investigate -H "Content-Type: application/json" \
     -d '{"question": "Why was transaction 3487632 flagged?"}'
curl localhost:8000/monitoring/summary
```
- Decisions: `FRAUD_REVIEW` = score above the lowest-cost threshold, `FRAUD_BLOCK` = score above the threshold with 80% precision (both chosen on validation).
- Explanations use TreeSHAP built into XGBoost (exact SHAP values). They are computed only for flagged transactions, to keep the API fast.
- The assistant writes the numbers itself; an LLM (optional, set `GROQ_API_KEY`) only phrases the explanation. Without a key it uses a template. Model: `qwen/qwen3-27b` via Groq (override with `LLM_MODEL` env var).
- Promotion gate: a new model becomes `production` in the MLflow registry only if its test PR-AUC is not worse than the current one.

## How it works
- Split by time (70 / 15 / 15), never random.
- Every feature uses only the past. Encoders and scalers are fit on train only.
- Thresholds are chosen on validation, then used once on test.
- Cost assumption: a missed fraud costs its amount, a false alarm costs $5.
- Every aggregate / count uses PAST rows only. The Kaggle winners computed them over train+test together (they saw the future too), so their scores are not comparable with ours.

## Known limitations (honest list)
- The API reads features from a file (replay mode). In production the history features (client aggregates, rolling counts) must be computed upstream and stored in a feature store.
- History features depend on how long the data has been collected (at the start everybody is a "new client"), so the drift report shows large PSI on them. This is a real weakness to improve.
- Docker files were not run on the machine where they were written - check `docker compose up` on yours.

More details: `docs/data_dictionary.md`, `docs/ROADMAP.md`.

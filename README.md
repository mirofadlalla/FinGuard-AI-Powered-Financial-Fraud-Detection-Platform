# FinGuard v2 — Fraud Detection on IEEE-CIS (work in progress)

Real data: Kaggle IEEE-CIS Fraud Detection (590K transactions, 3.5% fraud).
Compares Logistic Regression, XGBoost and a PyTorch residual MLP.
Metrics: PR-AUC, ROC-AUC, precision / recall / F1 and business cost. Accuracy is never used.

## Setup
1. Put `train_transaction.csv` and `train_identity.csv` inside `data/`
2. `pip install -r requirements.txt` (for the GPU use the CUDA command inside that file)

## Run
```
python scripts/01_prepare_data.py     # features + time split  -> data/ieee_ready.npz
python scripts/02_baselines.py        # LogReg + XGBoost (3 seeds)
python scripts/03_train_mlp.py        # PyTorch MLP, 4 imbalance strategies x 5 seeds
python scripts/04_ablation.py         # which feature groups really help? (XGBoost, 1 seed)
```
Add a number at the end for a quick run, e.g. `python scripts/03_train_mlp.py data 1` (1 seed).

## How it works
- Split by time (70 / 15 / 15), never random.
- Every feature uses only the past. Encoders and scalers are fit on train only.
- Thresholds are chosen on validation, then used once on test (3 options: best F1, lowest cost, precision >= 50%).
- Cost assumption: a missed fraud costs its amount, a false alarm costs $5.
- Every aggregate / count uses PAST rows only. The Kaggle winners computed them over train+test together (they saw the future too), so their scores are not comparable with ours.
- Results are mean ± std over several seeds. Results tables go to `artifacts/`.

More details: `docs/data_dictionary.md`, `docs/ROADMAP.md`.

"""Step 3: PyTorch MLP. 4 ways to handle the imbalance, each repeated with several seeds.

Run:  python scripts/03_train_mlp.py               (5 seeds, can take a while)
      python scripts/03_train_mlp.py data 1        (quick run: 1 seed)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finguard.metrics import operating_points, report, summarize, no_model_cost
from finguard.models import FraudMLP
from finguard.losses import make_loss
from finguard.torch_data import make_loader
from finguard.trainer import fit, predict

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
(ROOT / "artifacts").mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

d = np.load(data_dir / "ieee_ready.npz", allow_pickle=True)
X_tr, y_tr = d["X_train"], d["y_train"].astype("float32")
X_va, y_va, amt_va = d["X_val"], d["y_val"].astype("float32"), d["amount_val"]
X_te, y_te, amt_te = d["X_test"], d["y_test"].astype("float32"), d["amount_test"]

pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())
# the neural net cannot handle NaN: replace by 0 (= the mean after scaling)
X_tr, X_va, X_te = [np.nan_to_num(x, nan=0.0) for x in (X_tr, X_va, X_te)]
n_features = X_tr.shape[1]
print(f"features: {n_features} | train rows: {len(y_tr):,}")

# validation and test loaders never change
val_loader = make_loader(X_va, y_va, batch_size=4096, device=device)
test_loader = make_loader(X_te, y_te, batch_size=4096, device=device)

# name -> (loss type, oversample fraud 10x?)
# (comment a line out if you want a faster run)
strategies = {
    "MLP + BCE": ("bce", False),
    "MLP + weighted BCE": ("weighted", False),
    "MLP + focal": ("focal", False),
    "MLP + BCE + oversampling": ("bce", True),
}

runs = []
for seed in range(n_seeds):
    for name, (loss_kind, oversample) in strategies.items():
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader = make_loader(X_tr, y_tr, batch_size=2048, shuffle=True, device=device,
                                   fraud_weight=10 if oversample else None)
        model = FraudMLP(n_features, hidden=256, n_blocks=2, dropout=0.2)
        ckpt = ROOT / "artifacts" / f"{name.replace(' ', '_').replace('+', '')}_seed{seed}.pt"
        model, history = fit(model, train_loader, val_loader, y_va,
                             make_loss(loss_kind, pos_weight, device),
                             device=device, ckpt=str(ckpt), verbose=False)

        p_val = predict(model, val_loader, device)
        p_test = predict(model, test_loader, device)
        for op_name, thr in operating_points(y_va, p_val, amt_va).items():
            row = report(y_te, p_test, amt_te, thr, f"{name} @{op_name}")
            row["seed"] = seed
            runs.append(row)

        best_val = max(h["val_pr_auc"] for h in history)
        print(f"seed {seed} | {name:26s} | epochs {len(history):2d} | best val PR-AUC {best_val:.4f}")

runs = pd.DataFrame(runs)
runs.to_csv(ROOT / "artifacts" / "mlp_all_runs.csv", index=False)
summary = summarize(runs)
summary.to_csv(ROOT / "artifacts" / "mlp_results.csv", index=False)

print(f"\nCost if we use NO model: {no_model_cost(y_te, amt_te):,.0f}\n")
print(summary.to_string(index=False))

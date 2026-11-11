"""Step 7: Graph neural network (relational GraphSAGE) on the transaction graph.

Needs the neighbour table made by 01_prepare_data.py (run it again if you made the data before).

Run:  python scripts/07_train_gnn.py                    (3 seeds)
      python scripts/07_train_gnn.py data 1             (1 seed)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finguard.gnn import GraphSAGEFraud
from finguard.losses import make_loss
from finguard.metrics import no_model_cost, operating_points, report, summarize
from finguard.torch_data import make_loader
from finguard.trainer import fit, predict

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
smoke = len(sys.argv) > 3 and sys.argv[3] == "smoke"
(ROOT / "artifacts").mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device, "| smoke test" if smoke else "")

d = np.load(data_dir / "ieee_ready.npz", allow_pickle=True)
if "neighbors" not in d.files:
    sys.exit("No neighbour table in the data file. Run scripts/01_prepare_data.py again first.")

# all transactions in time order: the neighbour table uses these row numbers
X_all = np.nan_to_num(np.vstack([d["X_train"], d["X_val"], d["X_test"]]), nan=0.0).astype("float32")
neighbors = d["neighbors"]
n_tr, n_va, n_te = len(d["y_train"]), len(d["y_val"]), len(d["y_test"])
y_tr, y_va, y_te = [d[f"y_{s}"].astype("float32") for s in ("train", "val", "test")]
amt_va, amt_te = d["amount_val"], d["amount_test"]

ids_tr = np.arange(n_tr)
ids_va = np.arange(n_tr, n_tr + n_va)
ids_te = np.arange(n_tr + n_va, n_tr + n_va + n_te)
if smoke:
    ids_tr, y_tr = ids_tr[:20000], y_tr[:20000]
print(f"rows: train {len(ids_tr):,} | val {n_va:,} | test {n_te:,} | features {X_all.shape[1]} | "
      f"relations {list(d['relations'])} | neighbours per relation {neighbors.shape[2]}")
share = (neighbors[:, :, 0] >= 0).mean(axis=0)
print("share of transactions with at least 1 earlier neighbour:", dict(zip(map(str, d["relations"]), share.round(3))))

pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
loaders = dict(device=device, x_dtype=torch.long)              # the loader gives node numbers, the model looks up features
val_loader = make_loader(ids_va, y_va, batch_size=4096, **loaders)
test_loader = make_loader(ids_te, y_te, batch_size=4096, **loaders)

runs, val_preds, test_preds = [], [], []
for seed in range(n_seeds):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = make_loader(ids_tr, y_tr, batch_size=2048, shuffle=True, **loaders)
    model = GraphSAGEFraud(X_all, neighbors, hidden=128, n_blocks=2, dropout=0.2)
    model, history = fit(model, train_loader, val_loader, y_va, make_loss("bce", pos_weight, device), device=device,
                         epochs=2 if smoke else 40, ckpt=str(ROOT / "artifacts" / f"gnn_seed{seed}.pt"), verbose=smoke)

    p_val, p_test = predict(model, val_loader, device), predict(model, test_loader, device)
    val_preds.append(p_val)
    test_preds.append(p_test)
    for op_name, thr in operating_points(y_va, p_val, amt_va).items():
        row = report(y_te, p_test, amt_te, thr, f"GNN (GraphSAGE) @{op_name}")
        row["seed"] = seed
        runs.append(row)
    print(f"seed {seed} | epochs {len(history)} | best val PR-AUC {max(h['val_pr_auc'] for h in history):.4f}")

    del model                                                  # free the GPU before the next seed
    if device == "cuda":
        torch.cuda.empty_cache()

runs = pd.DataFrame(runs)
runs.to_csv(ROOT / "artifacts" / "gnn_all_runs.csv", index=False)
summary = summarize(runs)
summary.to_csv(ROOT / "artifacts" / "gnn_results.csv", index=False)
np.savez(ROOT / "artifacts" / "preds_gnn.npz", val=np.mean(val_preds, axis=0), test=np.mean(test_preds, axis=0))

print(f"\nCost if we use NO model: {no_model_cost(y_te, amt_te):,.0f}\n")
print(summary.to_string(index=False))

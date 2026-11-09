"""Step 1: read the csv files, build features, split by time, save to data/ieee_ready.npz

Run:  python scripts/01_prepare_data.py            (reads from the data/ folder)
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finguard.ieee import prepare

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
print("Reading files from:", data_dir)

d = prepare(data_dir)

total = sum(len(d[f"y_{s}"]) for s in ["train", "val", "test"])
print(f"\nrows: {total:,} | features: {len(d['feature_names'])}")
for s in ["train", "val", "test"]:
    y = d[f"y_{s}"]
    print(f"{s:5s} rows={len(y):>8,}  fraud={y.sum():>6,} ({y.mean():.2%})")

np.savez(data_dir / "ieee_ready.npz", **d)
print("\nSaved ->", data_dir / "ieee_ready.npz")

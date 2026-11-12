"""Step 1: read the csv files, build features, split by time, save to data/ieee_ready.npz

Also saves the fitted Preprocessor to data/preprocessor.pkl so that:
  • 05_train_final.py can embed it in the model bundle (no NameError on `prep`).
  • save_preprocessor.py loads the real mean/std and freq_maps instead of a hollow shell.
  • /predict/raw in api.py correctly z-scores raw transactions before scoring.

Run:  python scripts/01_prepare_data.py            (reads from the data/ folder)
"""
import pickle
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from features.ieee import prepare

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
print("Reading files from:", data_dir)
d = prepare(data_dir)

# Extract and persist the fitted preprocessor BEFORE dropping it from `d`.
prep = d.pop("_prep")
prep_path = data_dir / "preprocessor.pkl"
with open(prep_path, "wb") as _f:
    pickle.dump(prep, _f, protocol=4)
print(f"Saved fitted preprocessor -> {prep_path}  "
      f"(freq_maps: {len(prep.freq_maps)}, num_cols: {len(prep.num_cols)})")

total = sum(len(d[f"y_{s}"]) for s in ["train", "val", "test"])
print(f"\nrows: {total:,} | features: {len(d['feature_names'])}")
for s in ["train", "val", "test"]:
    y = d[f"y_{s}"]
    print(f"{s:5s} rows={len(y):>8,}  fraud={y.sum():>6,} ({y.mean():.2%})")

np.savez(data_dir / "ieee_ready.npz", **d)
print("\nSaved ->", data_dir / "ieee_ready.npz")

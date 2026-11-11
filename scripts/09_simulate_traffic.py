"""Step 9: send fake traffic to the running API so you can watch the dashboards move.

It replays test transactions IN TIME ORDER (so any drift in the data shows up), and for some of them
it sends the real outcome back with POST /feedback (like chargebacks arriving later).

Run (API must be running):  python scripts/09_simulate_traffic.py
      python scripts/09_simulate_traffic.py data 2000 0.05      (2000 transactions, 0.05 s between requests)
"""
import sys
import time
from pathlib import Path
import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
n_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
pause = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
API = "http://localhost:8000"

d = np.load(data_dir / "ieee_ready.npz", allow_pickle=True)
ids, labels = d["id_test"][:n_requests], d["y_test"][:n_requests]
rng = np.random.default_rng(0)

counts, errors = {}, 0
with httpx.Client(base_url=API, timeout=30) as client:
    for i, (txn_id, label) in enumerate(zip(ids, labels), 1):
        r = client.get(f"/score/{int(txn_id)}")
        if r.status_code != 200:
            errors += 1
            continue
        decision = r.json()["decision"]
        counts[decision] = counts.get(decision, 0) + 1
        if rng.random() < 0.4:                                  # 40% of the real outcomes come back
            client.post("/feedback", json={"txn_id": int(txn_id), "is_fraud": bool(label)})
        if i % 100 == 0:
            print(f"{i} sent | {counts} | errors {errors}")
        time.sleep(pause)

print("\nDone.", counts)
print("Monitoring summary:", client.get(f"{API}/monitoring/summary").json())

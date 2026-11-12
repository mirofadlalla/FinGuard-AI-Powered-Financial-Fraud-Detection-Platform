"""Trained XGBoost model + replay feature store used by the API."""
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from assistant.explain import describe

APPROVE, REVIEW, BLOCK = "APPROVE", "FRAUD_REVIEW", "FRAUD_BLOCK"
FRAUD_REVIEW, FRAUD_BLOCK = REVIEW, BLOCK


class Model:
    """Loads the saved model + settings, scores rows and explains them."""

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "meta.json").read_text())
        self.booster = xgb.Booster()
        self.booster.load_model(str(model_dir / "model.ubj"))
        self.booster.set_param({"device": "cpu"})
        self.feature_names = self.meta["feature_names"]
        self.version = self.meta["version"]
        self.review_thr = self.meta["thresholds"]["review"]
        self.block_thr = self.meta["thresholds"]["block"]

    def predict(self, X):
        """Fraud probability for each row of X (rows x features, NaN allowed)."""
        return self.booster.predict(xgb.DMatrix(np.asarray(X, dtype="float32"), missing=np.nan))

    def decide(self, prob):
        if prob >= self.block_thr:
            return BLOCK
        if prob >= self.review_thr:
            return REVIEW
        return APPROVE

    def explain(self, x, top_n=5):
        """Top TreeSHAP factors for one row. Positive SHAP = pushes towards fraud."""
        x = np.asarray(x, dtype="float32").reshape(1, -1)
        contrib = self.booster.predict(xgb.DMatrix(x, missing=np.nan), pred_contribs=True)[0][:-1]
        factors = []
        for i in np.argsort(-np.abs(contrib))[:top_n]:
            name = self.feature_names[i]
            factors.append({
                "feature": name,
                "description": describe(name),
                "shap": round(float(contrib[i]), 4),
                "effect": "raises risk" if contrib[i] > 0 else "lowers risk",
                "value_z": None if np.isnan(x[0, i]) else round(float(x[0, i]), 2),
            })
        return factors


class FeatureStore:
    """Lookup of precomputed feature vectors by TransactionID (file replay + live puts).

    In production these vectors would come from an upstream feature store.
    """

    def __init__(self, path, split="all"):
        d = np.load(path, allow_pickle=True, mmap_mode="r")
        splits = ["train", "val", "test"] if split == "all" else [split]
        self._parts = []
        self.index = {}
        for s in splits:
            x_key, id_key = f"X_{s}", f"id_{s}"
            if x_key not in d.files or id_key not in d.files:
                continue
            part_i = len(self._parts)
            self._parts.append(d[x_key])
            for i, t in enumerate(d[id_key]):
                self.index[int(t)] = (part_i, i)
        self.X = self._parts[0] if self._parts else None
        self._live: dict[int, np.ndarray] = {}

    def __len__(self):
        return len(self.index) + sum(1 for k in self._live if k not in self.index)

    def get(self, txn_id):
        tid = int(txn_id)
        live = self._live.get(tid)
        if live is not None:
            return live
        loc = self.index.get(tid)
        if loc is None:
            return None
        part_i, i = loc
        return self._parts[part_i][i]

    def put(self, txn_id, x, overwrite_prepared=False):
        """Keep a scored vector so /score/{id} works after /predict/raw.

        Prepared train/val/test vectors are not overwritten unless asked — those
        include history that a single raw JSON row cannot reconstruct.
        """
        tid = int(txn_id)
        if not overwrite_prepared and tid in self.index and tid not in self._live:
            return False
        self._live[tid] = np.asarray(x, dtype="float32").reshape(-1).copy()
        return True

"""Everything the API needs: the model, the feature store and readable explanations."""
import json
import re
from pathlib import Path
import numpy as np
import xgboost as xgb

APPROVE, REVIEW, BLOCK = "APPROVE", "FRAUD_REVIEW", "FRAUD_BLOCK"


def describe(name):
    """Turn a feature name into a sentence a fraud analyst can read."""
    if name.endswith("_freq"):
        return f"how common the value of {name[:-5]} is"
    if name.startswith("Vblock_"):
        return "a group of Vesta signals is missing for this transaction"
    if name.endswith("_na"):
        return f"{name[:-3]} is missing"
    if "_agg_" in name:
        who = "this client" if name.startswith("uid") else "this card/address"
        kind = "average" if name.endswith("_mean") else "variation"
        col = name.split("_agg_")[1].rsplit("_", 1)[0]
        return f"{kind} of {col} in the past activity of {who}"
    if name.startswith("combo_"):
        return "recent activity of this card + address + email combination (" + name.split("_count_")[1] + ")"
    if name.startswith(("uid_count", "card_count", "card1_count")):
        return "how many transactions this client/card made recently"
    if name.endswith("secs_since_last"):
        return "time since the previous transaction of this client"
    if name.endswith("amt_vs_mean"):
        return "amount compared with what this client usually spends"
    if name.startswith("Dn_"):
        return f"client age signal ({name[3:]} normalized)"
    if re.fullmatch(r"C\d+", name):
        return f"{name}: card-linkage count from Vesta (how the card connects to addresses, emails, devices)"
    if re.fullmatch(r"D\d+", name):
        return f"{name}: time since an earlier event of this client (Vesta)"
    if re.fullmatch(r"M\d+", name):
        return f"{name}: match check between card and billing details"
    if re.fullmatch(r"V\d+", name):
        return f"{name}: Vesta engineered risk signal"
    if name.startswith("na_count"):
        return "how many fields are missing for this transaction"
    if name == "TransactionAmt":
        return "transaction amount"
    return name


class Model:
    """Loads the saved model + settings, scores rows and explains them."""

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "meta.json").read_text())
        self.booster = xgb.Booster()
        self.booster.load_model(str(model_dir / "model.ubj"))
        self.booster.set_param({"device": "cpu"})          # the API runs on CPU
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
        """Top factors for ONE row. Uses TreeSHAP built into XGBoost (exact SHAP values).
        SHAP values are in log-odds: positive = pushes towards fraud, negative = towards legit."""
        x = np.asarray(x, dtype="float32").reshape(1, -1)
        contrib = self.booster.predict(xgb.DMatrix(x, missing=np.nan), pred_contribs=True)[0][:-1]
        factors = []
        for i in np.argsort(-np.abs(contrib))[:top_n]:
            name = self.feature_names[i]
            factors.append({"feature": name, "description": describe(name),
                            "shap": round(float(contrib[i]), 4),
                            "effect": "raises risk" if contrib[i] > 0 else "lowers risk",
                            "value_z": None if np.isnan(x[0, i]) else round(float(x[0, i]), 2)})
        return factors


class FeatureStore:
    """Where the API finds the features of a transaction.

    Here it is a file with the prepared test transactions (replay mode).
    In a real system these features would be computed upstream and read from a feature store.
    """

    def __init__(self, path, split="test"):
        # mmap_mode='r' avoids loading the full file into RAM at startup; arrays are paged in on demand.
        d = np.load(path, allow_pickle=True, mmap_mode="r")
        self.X = d[f"X_{split}"]
        self.index = {int(t): i for i, t in enumerate(d[f"id_{split}"])}

    def __len__(self):
        return len(self.index)

    def get(self, txn_id):
        i = self.index.get(int(txn_id))
        return None if i is None else self.X[i]

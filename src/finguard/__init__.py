"""FinGuard v2 — fraud detection library.

Public surface:

    Model        – loads a trained XGBoost model, scores rows, explains SHAP factors
    FeatureStore – looks up pre-computed feature vectors by TransactionID
    answer       – natural-language explanation for the /investigate endpoint
"""
from finguard.serving import APPROVE, REVIEW, BLOCK, Model, FeatureStore  # noqa: F401
from finguard.assistant import answer  # noqa: F401

__all__ = ["APPROVE", "REVIEW", "BLOCK", "Model", "FeatureStore", "answer"]

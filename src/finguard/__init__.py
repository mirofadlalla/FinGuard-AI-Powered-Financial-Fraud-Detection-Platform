"""FinGuard v2 — fraud detection library.

Public surface (all lazily imported so test-only code does not require xgboost
or prometheus_client to be installed):

    Model        — loads a trained XGBoost model, scores rows, explains SHAP factors
    FeatureStore — looks up pre-computed feature vectors by TransactionID
    answer       — natural-language explanation for the /investigate endpoint
    LiveMonitor  — rolling drift + quality monitor (wired into the API)

Lazy-import pattern: attributes are resolved on first access, not at import time.
This means `import finguard` is always safe; heavy deps are only required when
the attribute is actually used.
"""
from __future__ import annotations

__all__ = ["APPROVE", "REVIEW", "BLOCK", "Model", "FeatureStore", "answer", "LiveMonitor"]

# Constants are plain strings — no import needed.
APPROVE = "APPROVE"
REVIEW  = "FRAUD_REVIEW"
BLOCK   = "FRAUD_BLOCK"


def __getattr__(name: str):
    """Lazily resolve heavy symbols on first access."""
    if name in ("Model", "FeatureStore"):
        from finguard.serving import Model, FeatureStore  # noqa: F401
        globals()["Model"] = Model
        globals()["FeatureStore"] = FeatureStore
        return globals()[name]
    if name == "answer":
        from finguard.assistant import answer  # noqa: F401
        globals()["answer"] = answer
        return answer
    if name == "LiveMonitor":
        from finguard.monitor import LiveMonitor  # noqa: F401
        globals()["LiveMonitor"] = LiveMonitor
        return LiveMonitor
    raise AttributeError(f"module 'finguard' has no attribute {name!r}")

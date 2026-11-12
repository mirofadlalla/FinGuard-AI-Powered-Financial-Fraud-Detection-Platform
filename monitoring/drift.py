"""Drift monitoring with PSI (Population Stability Index).

PSI compares how a column was distributed at training time with how it looks now:
    PSI < 0.10  -> stable
    0.10 - 0.25 -> small shift, keep an eye on it
    PSI > 0.25  -> big shift, investigate / retrain
"""
import numpy as np

EPS = 1e-4   # avoids log(0)


def _proportions(values, edges):
    """Share of values in each bin (the last number is the share of missing values)."""
    values = np.asarray(values, dtype="float64")
    missing = np.isnan(values)
    counts = np.bincount(np.searchsorted(edges, values[~missing], side="right"), minlength=len(edges) + 1)
    props = counts / max(len(values), 1)
    return np.append(props, missing.mean())


def make_reference_column(values, n_bins=10):
    """Bin edges + the 'normal' proportions, learned from the training data."""
    values = np.asarray(values, dtype="float64")
    clean = values[~np.isnan(values)]
    edges = np.unique(np.quantile(clean, np.linspace(0, 1, n_bins + 1)[1:-1])) if len(clean) else np.array([])
    return {"edges": edges.tolist(), "props": _proportions(values, edges).tolist()}


def make_reference(X, feature_names, scores=None):
    """Reference for every feature (and optionally for the model score)."""
    ref = {name: make_reference_column(X[:, j]) for j, name in enumerate(feature_names)}
    if scores is not None:
        ref["__score__"] = make_reference_column(scores)
    return ref


def psi(reference, values):
    """PSI between the reference of one column and new values."""
    expected = np.clip(np.array(reference["props"]), EPS, None)
    actual = np.clip(_proportions(values, np.array(reference["edges"])), EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def feature_psi(reference, X, feature_names):
    """PSI of every feature. Returns a dict {feature: psi}."""
    return {name: psi(reference[name], X[:, j]) for j, name in enumerate(feature_names)}


def level(value):
    return "stable" if value < 0.10 else "small shift" if value < 0.25 else "BIG SHIFT"


calculate_psi = psi
psi_level = level

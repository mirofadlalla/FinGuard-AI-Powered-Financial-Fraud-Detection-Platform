"""Shared test fixtures for FinGuard test suites."""

import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pytest
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database.db import make_engine


@pytest.fixture(scope="module")
def meta():
    """Load production model metadata."""
    meta_path = ROOT / "artifacts" / "model" / "meta.json"
    if not meta_path.exists():
        pytest.skip("artifacts/model/meta.json not found")
    return json.loads(meta_path.read_text())


@pytest.fixture(scope="module")
def preprocessor():
    """Load production preprocessor."""
    prep_path = ROOT / "artifacts" / "model" / "preprocessor.pkl"
    if not prep_path.exists():
        pytest.skip("artifacts/model/preprocessor.pkl not found")
    with open(prep_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def booster():
    """Load production XGBoost booster."""
    model_path = ROOT / "artifacts" / "model" / "model.ubj"
    if not model_path.exists():
        pytest.skip("artifacts/model/model.ubj not found")
    b = xgb.Booster()
    b.load_model(str(model_path))
    b.set_param({"device": "cpu"})
    return b


@pytest.fixture(scope="module")
def test_data():
    """Load memory-mapped test set arrays."""
    npz_path = ROOT / "data" / "ieee_ready.npz"
    if not npz_path.exists():
        pytest.skip("data/ieee_ready.npz not found")
    npz = np.load(npz_path, allow_pickle=True, mmap_mode="r")
    return {
        "X_val":    npz["X_val"].astype("float32"),
        "y_val":    npz["y_val"],
        "amt_val":  npz["amount_val"],
        "X_test":   npz["X_test"].astype("float32"),
        "y_test":   npz["y_test"],
        "amt_test": npz["amount_test"],
        "id_test":  npz["id_test"],
    }


@pytest.fixture
def sqlite_engine(tmp_path):
    """Isolated temporary SQLite engine for database tests."""
    return make_engine(f"sqlite:///{tmp_path}/test.db")

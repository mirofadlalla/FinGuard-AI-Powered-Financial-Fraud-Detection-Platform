"""Process-wide serving state: model, feature store, DB, cache, monitor."""
from __future__ import annotations

import logging
import os
import pickle
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finguard.db import Cache, make_engine
from finguard.monitor import LiveMonitor, set_baselines
from finguard.raw_features import load_identity_table
from finguard.serving import FeatureStore, Model

log = logging.getLogger(__name__)
_reload_lock = threading.Lock()


@dataclass
class AppRuntime:
    model: Model | None = None
    store: FeatureStore | None = None
    identity: Any = None
    db: Any = None
    cache: Cache | None = None
    monitor: LiveMonitor | None = None
    preprocessor: Any = None


runtime = AppRuntime()


def _load_preprocessor(model_dir: Path):
    path = model_dir / "preprocessor.pkl"
    if not path.exists():
        log.warning(
            "preprocessor.pkl not found — /predict/raw unavailable. "
            "Run the data-prep / train-final scripts so it is copied into MODEL_DIR."
        )
        return None
    with open(path, "rb") as f:
        prep = pickle.load(f)
    log.info("Preprocessor loaded (%d freq_maps)", len(prep.freq_maps))
    return prep


def load_runtime(rt: AppRuntime | None = None) -> AppRuntime:
    """Load model, store, identity table, DB, cache and monitor into `rt`."""
    rt = rt or runtime
    model_dir = Path(os.getenv("MODEL_DIR", "artifacts/model"))
    model = Model(model_dir)
    rt.model = model
    feature_file = Path(os.getenv("FEATURE_FILE", "data/ieee_ready.npz"))
    rt.store = FeatureStore(feature_file)
    rt.identity = load_identity_table(feature_file.parent)
    rt.db = make_engine()
    rt.cache = Cache()
    rt.monitor = LiveMonitor(
        model_dir=str(model_dir),
        feature_names=model.feature_names,
        review_thr=model.review_thr,
    )
    set_baselines(model.meta)
    rt.preprocessor = _load_preprocessor(model_dir)
    return rt


def reload_model(rt: AppRuntime | None = None) -> AppRuntime:
    """Swap the in-memory model without restarting the process.

    In-flight /score requests finish with the old model; new requests use the new one
    after this returns. Returns the updated runtime.
    """
    rt = rt or runtime
    if not _reload_lock.acquire(blocking=False):
        raise RuntimeError("A reload is already in progress")
    try:
        model_dir = Path(os.getenv("MODEL_DIR", "artifacts/model"))
        new_model = Model(model_dir)
        new_monitor = LiveMonitor(
            model_dir=str(model_dir),
            feature_names=new_model.feature_names,
            review_thr=new_model.review_thr,
        )
        rt.model = new_model
        rt.monitor = new_monitor
        rt.preprocessor = _load_preprocessor(model_dir)
        set_baselines(new_model.meta)
        log.info("Hot-reloaded model %s", new_model.version)
        return rt
    finally:
        _reload_lock.release()


@asynccontextmanager
async def lifespan(_app):
    load_runtime(runtime)
    log.info("FinGuard started — model %s", runtime.model.version)
    yield
    log.info("FinGuard shutting down")

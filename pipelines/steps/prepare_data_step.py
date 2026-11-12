"""ZenML Step: Data Preparation and Leak-Free Feature Engineering."""

import pickle
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from zenml import step

from features.ieee import prepare


@step
def prepare_data_step(data_dir: str = "data") -> str:
    """Prepares IEEE-CIS dataset: loads raw CSVs, engineers leak-free features,
    splits by time, fits the Preprocessor, and saves ieee_ready.npz & preprocessor.pkl.
    
    Returns the path to the prepared data directory.
    """
    path = Path(data_dir)
    print(f"Executing ZenML Step: Preparing data from {path.resolve()}...")
    
    # Run full leak-free pipeline
    d = prepare(path)
    
    # Extract and save fitted preprocessor
    prep = d.pop("_prep")
    prep_path = path / "preprocessor.pkl"
    with open(prep_path, "wb") as f:
        pickle.dump(prep, f, protocol=4)
    print(f"Saved fitted preprocessor -> {prep_path}")
    
    # Save prepared dataset arrays
    npz_path = path / "ieee_ready.npz"
    np.savez(npz_path, **d)
    print(f"Saved prepared dataset -> {npz_path}")
    
    return str(path)

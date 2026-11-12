"""Features package for FinGuard."""
import sys
import types

from features.ieee import (
    Preprocessor,
    prepare,
    load_raw,
    add_features,
    signed_log,
    pick_v_columns,
    build_neighbor_table,
    recent_neighbors,
    rolling_count,
    add_rolling,
    add_history,
    add_past_aggs,
    C_COLS,
    D_COLS,
    M_COLS,
    V_COLS,
    CARD_COLS,
    ID_NUM_COLS,
    ID_CAT_COLS,
    FREQ_COLS,
    BASE_LOG_COLS,
    PLAIN_COLS,
    DN_COLS,
    RELATIONS,
)

# Compatibility shim for deserializing legacy pickles referencing finguard.ieee
if "finguard" not in sys.modules:
    _fg = types.ModuleType("finguard")
    sys.modules["finguard"] = _fg
if "finguard.ieee" not in sys.modules:
    import features.ieee as _ieee
    sys.modules["finguard.ieee"] = _ieee
    setattr(sys.modules["finguard"], "ieee", _ieee)

__all__ = [
    "Preprocessor",
    "prepare",
    "load_raw",
    "add_features",
    "signed_log",
    "pick_v_columns",
    "build_neighbor_table",
    "recent_neighbors",
    "rolling_count",
    "add_rolling",
    "add_history",
    "add_past_aggs",
    "C_COLS",
    "D_COLS",
    "M_COLS",
    "V_COLS",
    "CARD_COLS",
    "ID_NUM_COLS",
    "ID_CAT_COLS",
    "FREQ_COLS",
    "BASE_LOG_COLS",
    "PLAIN_COLS",
    "DN_COLS",
    "RELATIONS",
]

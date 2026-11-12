"""IEEE-CIS fraud data: load -> build features -> split by time -> encode.

Rules we follow everywhere:
  * a feature only uses information from BEFORE the transaction (no peeking at the future)
  * every encoder / scaler is learned on the TRAIN part only
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import types

# Ensure legacy pickles referencing finguard.ieee unpickle cleanly
if "finguard" not in sys.modules:
    _fg = types.ModuleType("finguard")
    sys.modules["finguard"] = _fg
if "finguard.ieee" not in sys.modules:
    sys.modules["finguard.ieee"] = sys.modules[__name__]
    setattr(sys.modules["finguard"], "ieee", sys.modules[__name__])

#  column lists 
C_COLS = [f"C{i}" for i in range(1, 15)]       # counts (ex: how many addresses are linked to the card)
D_COLS = [f"D{i}" for i in range(1, 16)]       # time deltas (days since some earlier event)
M_COLS = [f"M{i}" for i in range(1, 10)]       # match flags (T / F)
V_COLS = [f"V{i}" for i in range(1, 340)]      # Vesta engineered features (we keep only some of them)
CARD_COLS = [f"card{i}" for i in range(1, 7)]

ID_NUM_COLS = [f"id_{i:02d}" for i in range(1, 12)]    # identity numbers: id_01 ... id_11
ID_CAT_COLS = [f"id_{i}" for i in range(12, 39)]       # identity categories: id_12 ... id_38

TRANSACTION_WANTED = set(
    ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD"] + CARD_COLS
    + ["addr1", "addr2", "dist1", "P_emaildomain", "R_emaildomain"] + C_COLS + D_COLS + M_COLS + V_COLS)
IDENTITY_WANTED = set(["TransactionID", "DeviceType", "DeviceInfo"] + ID_NUM_COLS + ID_CAT_COLS)

# D columns that become "normalized" (D - day). The top solutions picked these ones.
DN_COLS = [f"Dn_D{i}" for i in (2, 4, 10, 11, 15)]

# aggregates over the client: (column, statistic). Taken from the top solutions.
UID_MEAN = ["M9", "M5", "Dn_D2", "Dn_D15", "C13", "C9", "C1", "C11", "TransactionAmt"]
UID_STD = ["TransactionAmt", "Dn_D15", "Dn_D4"]
CARD_MEAN = ["M4", "M1", "M7"]
CARD_STD = ["M8"]


def agg_names(prefix, mean_cols, std_cols):
    return ([f"{prefix}_agg_{c}_mean" for c in mean_cols] + [f"{prefix}_agg_{c}_std" for c in std_cols])


AGG_COLS = agg_names("uid", UID_MEAN, UID_STD) + agg_names("card", CARD_MEAN, CARD_STD)

# combinations of columns (a "who is this" fingerprint) and how busy each one was recently
COMBOS = {"c1_a1_pe": ["card1", "addr1", "P_emaildomain"],
          "c1_a1_re": ["card1", "addr1", "R_emaildomain"],
          "c3_a1_pe": ["card3", "addr1", "P_emaildomain"],
          "c4_a1_pe": ["card4", "addr1", "P_emaildomain"],
          "c4_a1_re": ["card4", "addr1", "R_emaildomain"]}
COMBO_WINDOWS = {"1d": 86400, "7d": 7 * 86400}
COMBO_COLS = [f"combo_{n}_count_{w}" for n in COMBOS for w in COMBO_WINDOWS]

# ID-like columns: the number itself means nothing (card1 = 9500 is not "bigger" than 9000),
# so we replace each value by how often it appears in the train set.
FREQ_COLS = (["ProductCD"] + CARD_COLS + ["addr1", "addr2", "P_emaildomain", "R_emaildomain",
             "DeviceType", "DeviceInfo", "card_key"] + ID_CAT_COLS)

# numeric columns with a long tail -> we squash them with a log
BASE_LOG_COLS = (["TransactionAmt", "dist1",
                  "card_amt_vs_mean", "card_secs_since_last",
                  "uid_amt_vs_mean", "uid_secs_since_last",
                  "uid_count_1h", "uid_count_1d", "uid_count_7d", "card1_count_1d"]
                 + COMBO_COLS + AGG_COLS + C_COLS + D_COLS)
# numeric columns we keep as they are (they only get scaled later)
PLAIN_COLS = (["hour_sin", "hour_cos", "dow", "amt_decimal", "email_same", "has_identity",
               "na_count_D", "na_count_V", "na_count_id"] + DN_COLS + M_COLS)


#  loading 
def _read_csv(path, wanted, float_cols):
    """Read only the columns we want. Also turns 'id-01' into 'id_01' (test files use dashes)."""
    df = pd.read_csv(path, usecols=lambda c: c.replace("-", "_") in wanted,
                     dtype={c: "float32" for c in float_cols})
    df.columns = [c.replace("-", "_") for c in df.columns]
    return df


def load_raw(data_dir):
    """Read both csv files and join them on TransactionID."""
    data_dir = Path(data_dir)
    float_cols = V_COLS + C_COLS + D_COLS + ID_NUM_COLS + ["TransactionAmt", "dist1"]
    tx = _read_csv(data_dir / "train_transaction.csv", TRANSACTION_WANTED, float_cols)
    ident = _read_csv(data_dir / "train_identity.csv", IDENTITY_WANTED, float_cols)

    # only ~24% of transactions have identity info, so we remember who has it
    tx["has_identity"] = tx["TransactionID"].isin(ident["TransactionID"]).astype("float32")
    df = tx.merge(ident, on="TransactionID", how="left")

    n_v = sum(c in df.columns for c in V_COLS)
    n_id = sum(c in df.columns for c in ID_NUM_COLS + ID_CAT_COLS)
    print(f"Loaded {len(df):,} rows | V columns found: {n_v} | identity columns found: {n_id}")
    return df


#  feature engineering (past only) 
def add_history(df, key, prefix):
    """Past behaviour of each group (ex: each card). Only uses rows BEFORE the current one."""
    codes = pd.factorize(df[key])[0]                      # one integer per group
    amount = df["TransactionAmt"].astype("float64")

    count_before = df.groupby(codes).cumcount()           # how many transactions before this one
    past_sum = amount.groupby(codes).cumsum() - amount    # sum of past amounts
    past_mean = past_sum / count_before.replace(0, np.nan)

    df[f"{prefix}_count_before"] = count_before
    # is this amount bigger than what this group usually spends?
    df[f"{prefix}_amt_vs_mean"] = (amount / past_mean).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    # seconds since the previous transaction of this group (-1 = first time we see it)
    df[f"{prefix}_secs_since_last"] = df["TransactionDT"].groupby(codes).diff().fillna(-1)


def add_past_aggs(df, key, prefix, mean_cols, std_cols):
    """Mean / std of a column over the PAST rows of the same group (NaN values are skipped).

    Running sums trick: mean = sum / n, std = sqrt(sum of squares / n - mean^2).
    We subtract the current row so it never sees itself.
    """
    codes = pd.factorize(df[key])[0]
    for col in sorted(set(mean_cols) | set(std_cols)):
        x = df[col].astype("float64")
        has_value = x.notna().astype("float64")
        x0 = x.fillna(0.0)

        n = (has_value.groupby(codes).cumsum() - has_value).replace(0, np.nan)
        mean = (x0.groupby(codes).cumsum() - x0) / n
        if col in mean_cols:
            df[f"{prefix}_agg_{col}_mean"] = mean
        if col in std_cols:
            sq = x0 ** 2
            var = ((sq.groupby(codes).cumsum() - sq) / n - mean ** 2).clip(lower=0)
            df[f"{prefix}_agg_{col}_std"] = np.sqrt(var)


def rolling_count(codes, t, seconds):
    """For each row: how many rows of the SAME group happened in the last `seconds` (not counting itself).

    Trick: sort by (group, time), then glue group and time into one number
    so one searchsorted call works for all groups at once.

    Raises OverflowError if any code value is so large that codes * BIG would
    overflow int64.  The safe limit is 9_200_000_000_000_000_000 // BIG
    (conservative: leaves headroom for the time offset).  Callers should
    factorize codes first (pd.factorize gives 0-based integers) so this never
    fires in practice.
    """
    BIG = 10 ** 9                                         # bigger than any time value
    # Conservative int64 ceiling that matches what the test asserts.
    # 9_200_000_000 * BIG = 9.2e18, safely inside int64 max (9.22e18).
    _MAX_CODE = 9_200_000_000_000_000_000 // BIG          # == 9_200_000_000
    max_code = int(codes.max()) if len(codes) else 0
    if max_code > _MAX_CODE:
        raise OverflowError(
            f"code value {max_code} > {_MAX_CODE}: multiplying by {BIG} would overflow int64. "
            "Factorize the codes first (e.g. pd.factorize gives 0-based integers)."
        )
    order = np.lexsort((t, codes))
    glued = codes[order].astype("int64") * BIG + t[order]
    first_in_window = np.searchsorted(glued, glued - seconds, side="left")
    first_same_time = np.searchsorted(glued, glued, side="left")
    out = np.empty(len(t))
    out[order] = first_same_time - first_in_window
    return out


def add_rolling(df, key, prefix, windows):
    codes = pd.factorize(df[key])[0]
    t = df["TransactionDT"].values.astype("int64")
    for name, seconds in windows.items():
        df[f"{prefix}_count_{name}"] = rolling_count(codes, t, seconds)


def add_features(df):
    """Create the new columns. Rows must be in time order."""
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    day = df["TransactionDT"] // 86400            # TransactionDT = seconds from some start date

    # --- time (hour as a circle: 23h and 0h are close) ---
    hour = (df["TransactionDT"] / 3600) % 24
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow"] = day % 7

    # --- amount and simple flags ---
    df["amt_decimal"] = df["TransactionAmt"] % 1          # ex: 59.00 vs 59.37
    df["email_same"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype("float32")
    mapping = {"T": 1.0, "F": 0.0, "M0": 0.0, "M1": 1.0, "M2": 2.0}
    for col in M_COLS:
        df[col] = df[col].map(mapping).astype("float32")

    # --- normalized D columns: (D - today) is almost the same number for every transaction of a client ---
    day_float = df["TransactionDT"] / 86400
    for name in DN_COLS:
        d = name.replace("Dn_", "")
        df[name] = (df[d] - day_float).astype("float32")

    # --- how many values are missing in each row (missing itself is a signal) ---
    v_present = [c for c in V_COLS if c in df.columns]
    id_present = [c for c in ID_NUM_COLS + ID_CAT_COLS + ["DeviceType", "DeviceInfo"] if c in df.columns]
    df["na_count_D"] = df[D_COLS].isna().sum(axis=1)
    df["na_count_V"] = df[v_present].isna().sum(axis=1)
    df["na_count_id"] = df[id_present].isna().sum(axis=1)

    # --- "client id" ---
    # card + address = probably the same person
    df["card_key"] = df["card1"].astype(str) + "_" + df["addr1"].astype(str)
    # D1 = days since the card's first transaction, so (day - D1) = the day the card started.
    # card + address + start day = a much better client id (the "magic" of the 1st place solution)
    start_day = (day - df["D1"]).round()
    df["uid"] = df["card_key"] + "_" + start_day.astype(str)

    # --- behaviour history of the client (all use the past only) ---
    add_history(df, "card_key", "card")
    add_history(df, "uid", "uid")
    add_rolling(df, "uid", "uid", {"1h": 3600, "1d": 86400, "7d": 7 * 86400})
    add_rolling(df, "card1", "card1", {"1d": 86400})

    # --- aggregates over the client (past rows only) ---
    add_past_aggs(df, "uid", "uid", UID_MEAN, UID_STD)
    add_past_aggs(df, "card_key", "card", CARD_MEAN, CARD_STD)

    # --- combinations: how busy was this card+address+email in the last day / week ---
    for name, cols in COMBOS.items():
        key = df[cols[0]].astype(str)
        for c in cols[1:]:
            key = key + "_" + df[c].astype(str)
        df["_combo_key"] = key
        add_rolling(df, "_combo_key", f"combo_{name}", COMBO_WINDOWS)
    df = df.drop(columns="_combo_key")
    return df


#  graph (used by the GNN) 
# Each transaction is a node. It is connected to the LAST k EARLIER transactions that share
# an entity with it (same client, same card+address, same device, same email+address).
# Edges only point from the past to the present, so the graph never leaks the future.
RELATIONS = ["uid", "card_key", "DeviceInfo", "email_addr"]
N_NEIGHBORS = 10
MAX_GROUP_SIZE = 1000      # an entity with more transactions than this is "generic" (ex: device = "Windows")


def recent_neighbors(codes, k):
    """For each row: row numbers of the previous k rows with the same code (-1 = none).
    Rows must already be in time order. codes < 0 means 'no entity'."""
    n = len(codes)
    order = np.argsort(codes, kind="stable")            # stable: keeps the time order inside each group
    sorted_codes = codes[order]
    pos = np.arange(n)
    is_first = np.r_[True, sorted_codes[1:] != sorted_codes[:-1]]
    group_start = np.maximum.accumulate(np.where(is_first, pos, 0))
    rank = pos - group_start                            # how many rows of the same group came before

    out = np.full((n, k), -1, dtype="int32")
    for j in range(1, k + 1):                           # j-th previous row of the group
        ok = rank >= j
        out[order[ok], j - 1] = order[pos[ok] - j]
    out[codes < 0] = -1
    return out


def build_neighbor_table(df, k=N_NEIGHBORS, max_group_size=MAX_GROUP_SIZE):
    """Array (rows, relations, k) with the neighbour row numbers. df must be in time order."""
    has_card = df["card1"].notna() & df["addr1"].notna()
    keys = {"uid": df["uid"].where(df["D1"].notna() & has_card),
            "card_key": df["card_key"].where(has_card),
            "DeviceInfo": df["DeviceInfo"],
            "email_addr": (df["P_emaildomain"].astype(str) + "_" + df["addr1"].astype(str))
                          .where(df["P_emaildomain"].notna() & df["addr1"].notna())}
    table = np.full((len(df), len(RELATIONS), k), -1, dtype="int32")
    for r, name in enumerate(RELATIONS):
        codes, _ = pd.factorize(keys[name])             # missing values -> -1
        sizes = np.bincount(codes[codes >= 0], minlength=codes.max() + 1)
        codes = np.where((codes >= 0) & (sizes[np.maximum(codes, 0)] <= max_group_size), codes, -1)
        table[:, r, :] = recent_neighbors(codes, k)
    return table


#  preprocessing 
def pick_v_columns(train, corr_limit=0.75, sample_size=100_000):
    """Reduce the V columns (no labels used), like the top solutions did:
    1) put columns with the same number of missing values in one group
    2) inside a group, columns that are strongly correlated are 'the same information'
    3) keep one per cluster: the one with the most different values
    """
    cols = [c for c in V_COLS if c in train.columns]
    if not cols:
        return []
    sample = train[cols].sample(min(sample_size, len(train)), random_state=0)
    n_missing = train[cols].isna().sum()

    chosen = []
    for _, group in n_missing.groupby(n_missing):
        names = list(group.index)
        corr = sample[names].corr().abs().fillna(0).values

        # connected clusters of correlated columns (simple union-find)
        parent = list(range(len(names)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if corr[i, j] > corr_limit:
                    parent[find(i)] = find(j)
        clusters = {}
        for i, name in enumerate(names):
            clusters.setdefault(find(i), []).append(name)

        for members in clusters.values():
            best = max(members, key=lambda c: sample[c].nunique())
            if sample[best].nunique() > 1:                 # skip constant columns
                chosen.append(best)
    print(f"V columns: {len(cols)} -> keeping {len(chosen)}")
    return chosen


def signed_log(x):
    """log that also works for negative numbers (some columns use -1)."""
    return np.sign(x) * np.log1p(np.abs(x))


class Preprocessor:
    """Learns everything from the TRAIN data only, then applies it to any data.

    Missing values are KEPT as NaN: XGBoost knows how to use them.
    For Logistic Regression / the neural net the scripts replace NaN by 0 (= the mean after scaling).
    """

    def fit(self, train):
        self.v_cols = pick_v_columns(train)
        self.freq_cols = [c for c in FREQ_COLS if c in train.columns]
        id_nums = [c for c in ID_NUM_COLS if c in train.columns]

        log_cols = [c for c in BASE_LOG_COLS if c in train.columns] + id_nums + self.v_cols
        self.log_cols = set(log_cols)
        self.num_cols = log_cols + [c for c in PLAIN_COLS if c in train.columns]

        self.freq_maps = {c: train[c].value_counts(normalize=True) for c in self.freq_cols}
        # "is missing" flag for columns that are missing a lot
        self.na_cols = [c for c in self.num_cols
                        if c not in self.v_cols and train[c].isna().mean() > 0.05]

        # V columns come in blocks that are missing together. The block pattern helps to
        # recognise clients, so we add ONE flag per block.
        v_all = [c for c in V_COLS if c in train.columns]
        n_missing = train[v_all].isna().sum()
        self.v_block_reps = [g.index[0] for n, g in n_missing.groupby(n_missing) if n > 0 and len(g) >= 2]

        raw = self._build(train)
        self.mean = raw.mean()
        self.std = raw.std().replace(0, 1).fillna(1)
        self.feature_names = list(raw.columns)
        return self

    def _build(self, df):
        cols = {}
        for c in self.freq_cols:
            cols[c + "_freq"] = df[c].map(self.freq_maps[c]).fillna(0)
        for c in self.na_cols:
            cols[c + "_na"] = df[c].isna()
        for rep in self.v_block_reps:
            cols[f"Vblock_{rep}_na"] = df[rep].isna()
        for c in self.num_cols:
            cols[c] = signed_log(df[c]) if c in self.log_cols else df[c]
        return pd.DataFrame(cols, index=df.index).astype("float32")

    def transform(self, df):
        x = (self._build(df) - self.mean) / self.std
        return x.clip(-5, 5).astype("float32")             # cut crazy outliers (helps the neural net)


def prepare(data_dir, train_frac=0.70, val_frac=0.15):
    """Full pipeline: read -> features -> time split -> encode. Returns a dict of arrays."""
    df = add_features(load_raw(data_dir))
    neighbors = build_neighbor_table(df)               # for the GNN (row numbers of the whole sorted data)

    # split by time (rows are sorted by TransactionDT)
    n = len(df)
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    parts = {"train": df.iloc[:a], "val": df.iloc[a:b], "test": df.iloc[b:]}

    prep = Preprocessor().fit(parts["train"])

    # splits are consecutive in time, so row numbers in `neighbors` match vstack(train, val, test)
    out = {"feature_names": np.array(prep.feature_names), "neighbors": neighbors, "relations": np.array(RELATIONS)}
    for name, part in parts.items():
        out[f"X_{name}"] = prep.transform(part).values
        out[f"id_{name}"] = part["TransactionID"].values.astype("int64")    # needed later by the API
        out[f"y_{name}"] = part["isFraud"].values.astype("int8")
        out[f"amount_{name}"] = part["TransactionAmt"].values.astype("float32")   # for the cost metric
    # Return the fitted preprocessor so callers can persist it alongside the arrays.
    # Saving it here (from the actual fit) is the only way to recover the real mean/std
    # and freq_maps for correct serving via /predict/raw.
    out["_prep"] = prep
    return out

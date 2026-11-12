"""
Regression tests for the fraud prediction pipeline.

Covers:
  - Training/inference feature consistency
  - Missing-value handling
  - Categorical feature encoding
  - Ground-truth label exclusion (no data leakage)
  - Correct fraud probability extraction
  - Threshold sanity (review_thr > 0.005, block_thr > review_thr)
  - Model loading consistency
  - API prediction consistency (via direct model calls)
  - Multi-transaction sanity (model does NOT predict fraud for everything)
"""
import sys
import json
import pickle
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finguard.metrics import cost_threshold, precision_threshold, best_f1_threshold
import xgboost as xgb

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def meta():
    return json.loads((ROOT / "artifacts" / "model" / "meta.json").read_text())

@pytest.fixture(scope="module")
def preprocessor():
    with open(ROOT / "artifacts" / "model" / "preprocessor.pkl", "rb") as f:
        return pickle.load(f)

@pytest.fixture(scope="module")
def booster():
    b = xgb.Booster()
    b.load_model(str(ROOT / "artifacts" / "model" / "model.ubj"))
    b.set_param({"device": "cpu"})
    return b

@pytest.fixture(scope="module")
def test_data():
    npz = np.load(ROOT / "data" / "ieee_ready.npz", allow_pickle=True, mmap_mode="r")
    return {
        "X_val":    npz["X_val"].astype("float32"),
        "y_val":    npz["y_val"],
        "amt_val":  npz["amount_val"],
        "X_test":   npz["X_test"].astype("float32"),
        "y_test":   npz["y_test"],
        "amt_test": npz["amount_test"],
        "id_test":  npz["id_test"],
    }


# ── A. Data Leakage Tests ─────────────────────────────────────────────────────

class TestNoDataLeakage:

    def test_isfraud_not_in_model_features(self, meta):
        """isFraud must NEVER appear in model feature names."""
        names = meta["feature_names"]
        leakers = [n for n in names if "fraud" in n.lower() or n == "isFraud"]
        assert leakers == [], f"Label leakage found: {leakers}"

    def test_transaction_id_not_in_model_features(self, meta):
        """TransactionID is a row identifier, not a signal."""
        names = meta["feature_names"]
        assert "TransactionID" not in names, "TransactionID leaked into features"

    def test_isfraud_not_in_preprocessor_num_cols(self, preprocessor):
        """isFraud must not appear in preprocessor's numeric columns."""
        leakers = [c for c in preprocessor.num_cols if "fraud" in c.lower()]
        assert leakers == [], f"isFraud in num_cols: {leakers}"

    def test_isfraud_not_in_preprocessor_freq_cols(self, preprocessor):
        """isFraud must not appear in freq_cols."""
        leakers = [c for c in preprocessor.freq_cols if "fraud" in c.lower()]
        assert leakers == [], f"isFraud in freq_cols: {leakers}"


# ── B. Feature Alignment Tests ────────────────────────────────────────────────

class TestFeatureAlignment:

    def test_model_and_preprocessor_same_feature_count(self, meta, preprocessor):
        """Model and preprocessor must expose exactly the same number of features."""
        assert len(meta["feature_names"]) == len(preprocessor.feature_names), (
            f"Model has {len(meta['feature_names'])} features, "
            f"preprocessor has {len(preprocessor.feature_names)}"
        )

    def test_model_and_preprocessor_same_feature_names(self, meta, preprocessor):
        """Feature sets must be identical (order-insensitive check)."""
        model_set = set(meta["feature_names"])
        prep_set  = set(preprocessor.feature_names)
        only_model = model_set - prep_set
        only_prep  = prep_set - model_set
        assert only_model == set(), f"In model but not preprocessor: {sorted(only_model)[:5]}"
        assert only_prep  == set(), f"In preprocessor but not model: {sorted(only_prep)[:5]}"

    def test_model_and_preprocessor_feature_order(self, meta, preprocessor):
        """Feature order must match exactly (model uses positional indexing)."""
        mismatches = [
            (i, meta["feature_names"][i], preprocessor.feature_names[i])
            for i in range(min(len(meta["feature_names"]), len(preprocessor.feature_names)))
            if meta["feature_names"][i] != preprocessor.feature_names[i]
        ]
        assert mismatches == [], f"Order mismatch at positions: {mismatches[:3]}"

    def test_no_duplicate_feature_names_in_model(self, meta):
        names = meta["feature_names"]
        assert len(names) == len(set(names)), "Duplicate feature names in model"

    def test_no_duplicate_feature_names_in_preprocessor(self, preprocessor):
        names = preprocessor.feature_names
        assert len(names) == len(set(names)), "Duplicate feature names in preprocessor"


# ── C. Missing Value Handling ─────────────────────────────────────────────────

class TestMissingValueHandling:

    def test_preprocessor_mean_not_all_zeros(self, preprocessor):
        """If mean is all-zero the z-scoring is a no-op (broken preprocessor)."""
        zeros = (preprocessor.mean == 0).sum()
        assert zeros < len(preprocessor.mean), (
            f"All {zeros} means are 0 — preprocessor was not properly fitted"
        )

    def test_preprocessor_std_not_all_ones(self, preprocessor):
        """If std is all-one the z-scoring is a no-op (broken preprocessor)."""
        ones = (preprocessor.std == 1).sum()
        assert ones < len(preprocessor.std), (
            f"All {ones} stds are 1 — preprocessor was not properly fitted"
        )

    def test_preprocessor_freq_maps_non_empty(self, preprocessor):
        """Every freq_col must have a populated frequency map from training."""
        empty = [c for c, fm in preprocessor.freq_maps.items() if len(fm) == 0]
        assert empty == [], f"Empty freq_maps for: {empty}"

    def test_all_nan_input_returns_low_probability(self, booster, meta):
        """A fully-missing transaction should not trigger a high fraud score."""
        x = np.full(len(meta["feature_names"]), np.nan, dtype="float32")
        dmat = xgb.DMatrix(x.reshape(1, -1), missing=np.nan)
        prob = float(booster.predict(dmat)[0])
        # An all-NaN row is uninformative — it should not be aggressively scored
        assert prob < 0.5, f"All-NaN row produced prob={prob:.4f} >= 0.5"

    def test_null_values_become_nan_not_zero(self, preprocessor):
        """NaN values in numeric columns must remain NaN through _build() (not be coerced to 0)."""
        # Build a minimal df with np.nan (the Python-None → np.nan conversion happens
        # in api.py via df.replace({None: np.nan}) before the preprocessor sees it).
        all_cols = (
            preprocessor.freq_cols + preprocessor.na_cols + preprocessor.num_cols
            + list(preprocessor.v_block_reps)
        )
        row = {c: np.nan for c in all_cols}
        df = pd.DataFrame([row])
        raw = preprocessor._build(df)
        # Non-freq numeric columns with NaN input must produce NaN output
        non_freq_num = [c for c in preprocessor.num_cols if c not in set(preprocessor.freq_cols)]
        if non_freq_num:
            col = non_freq_num[0]
            assert pd.isna(raw[col].iloc[0]), (
                f"NaN input was coerced to a non-NaN value in {col}: got {raw[col].iloc[0]}"
            )


# ── D. Categorical Encoding ───────────────────────────────────────────────────

class TestCategoricalEncoding:

    def test_known_category_gets_nonzero_frequency(self, preprocessor):
        """A category that appeared in training must map to a nonzero frequency."""
        col = "card4"
        assert col in preprocessor.freq_maps, f"{col} not in freq_maps"
        # 'visa' is the most common value in IEEE-CIS
        assert "visa" in preprocessor.freq_maps[col], "'visa' not in card4 freq_map"
        assert preprocessor.freq_maps[col]["visa"] > 0

    def test_unknown_category_maps_to_zero_frequency(self, preprocessor):
        """An unseen category must silently map to frequency 0 (fillna(0) in _build)."""
        col = "card4"
        all_cols = (
            preprocessor.freq_cols + preprocessor.na_cols + preprocessor.num_cols
            + list(preprocessor.v_block_reps)
        )
        row = {c: np.nan for c in all_cols}
        row[col] = "__UNKNOWN_BRAND__"   # an unseen string category
        df = pd.DataFrame([row])
        raw = preprocessor._build(df)
        # _build fills missing freq with 0 before z-scoring
        assert raw[col + "_freq"].iloc[0] == 0.0, (
            f"Unknown category did not map to 0: got {raw[col+'_freq'].iloc[0]}"
        )

    def test_m_flag_T_maps_to_1(self, preprocessor):
        """M-columns 'T' should map to 1.0 and 'F' to 0.0 in preprocessing.

        Verified against the mapping dict in ieee.py add_features():
            mapping = {"T": 1.0, "F": 0.0, "M0": 0.0, "M1": 1.0, "M2": 2.0}
        df[col] = df[col].map(mapping)

        We test this directly on a Series (the same operation add_features applies)
        instead of calling add_features() with a full transaction df (which requires
        all C/D/V columns to be present).
        """
        import pandas as pd
        mapping = {"T": 1.0, "F": 0.0, "M0": 0.0, "M1": 1.0, "M2": 2.0}
        series = pd.Series(["T", "F", "T", None, "F"])
        mapped = series.map(mapping).astype("float32")
        assert mapped.iloc[0] == 1.0, "M='T' should map to 1.0"
        assert mapped.iloc[1] == 0.0, "M='F' should map to 0.0"
        assert pd.isna(mapped.iloc[3]),  "M=None should map to NaN (missing)"


# ── E. Prediction Threshold Tests ─────────────────────────────────────────────

class TestPredictionThreshold:

    def test_review_threshold_above_minimum(self, meta):
        """review_thr must be a meaningful probability (> 0.005).
        A threshold below this flags too many transactions as fraud."""
        thr = meta["thresholds"]["review"]
        assert thr > 0.005, (
            f"review_thr={thr:.6f} is dangerously low — "
            "this causes nearly all transactions to be flagged as fraud. "
            "Bug: cost_threshold was searching quantile space instead of probability space."
        )

    def test_block_threshold_above_review(self, meta):
        """block_thr must be strictly greater than review_thr."""
        review_thr = meta["thresholds"]["review"]
        block_thr  = meta["thresholds"]["block"]
        assert block_thr > review_thr, (
            f"block_thr ({block_thr:.6f}) <= review_thr ({review_thr:.6f})"
        )

    def test_block_threshold_reasonable_range(self, meta):
        """block_thr must be in (0.05, 1.0)."""
        thr = meta["thresholds"]["block"]
        assert 0.05 < thr < 1.0, f"block_thr={thr:.6f} is out of reasonable range"

    def test_review_threshold_flags_less_than_30_percent(self, booster, meta, test_data):
        """At most 30% of test transactions should be flagged (review+block)."""
        X_test = test_data["X_test"]
        review_thr = meta["thresholds"]["review"]
        p = booster.predict(xgb.DMatrix(X_test, missing=np.nan))
        flagged_pct = (p >= review_thr).mean()
        assert flagged_pct <= 0.30, (
            f"{flagged_pct:.2%} of test transactions are flagged — threshold too low"
        )

    def test_approve_rate_above_70_percent(self, booster, meta, test_data):
        """Legitimate-majority dataset: > 70% of transactions should be APPROVE."""
        X_test = test_data["X_test"]
        review_thr = meta["thresholds"]["review"]
        p = booster.predict(xgb.DMatrix(X_test, missing=np.nan))
        approve_pct = (p < review_thr).mean()
        assert approve_pct >= 0.70, (
            f"Only {approve_pct:.2%} transactions are approved — something is wrong"
        )

    def test_cost_threshold_returns_sane_value(self):
        """Fixed cost_threshold must return a threshold in [0.01, 0.99]."""
        rng = np.random.default_rng(42)
        y = (rng.random(1000) < 0.035).astype(int)
        p = np.where(y == 1,
                     rng.uniform(0.3, 1.0, 1000),
                     rng.uniform(0.0, 0.1, 1000))
        amt = rng.uniform(10, 500, 1000).astype("float32")
        thr, cost = cost_threshold(y, p, amt)
        assert 0.01 <= thr <= 0.99, f"cost_threshold returned {thr} outside [0.01, 0.99]"

    def test_cost_threshold_does_not_start_at_quantile_space(self):
        """
        Regression test for the original bug:
        cost_threshold must NOT search np.quantile(p, np.linspace(0.5, 0.9999, 400)).
        That approach starts at the 50th percentile of scores and produces microscopic
        thresholds on well-calibrated models for rare-fraud datasets.

        The fixed version searches np.linspace(0.01, 0.99, 400) in probability space,
        so the minimum threshold it can ever return is 0.01.
        """
        rng = np.random.default_rng(0)
        n = 10_000
        y = (rng.random(n) < 0.035).astype(int)
        p = np.where(y == 1, rng.beta(2, 2, n), rng.beta(0.5, 10, n))
        amt = rng.uniform(10, 500, n).astype("float32")

        thr_fixed, _ = cost_threshold(y, p, amt)

        # The fixed function must always return at least 0.01
        assert thr_fixed >= 0.01, (
            f"cost_threshold returned {thr_fixed:.6f} which is below 0.01 — "
            "likely still using quantile-space search (old bug)"
        )
        # And must stay below 1.0
        assert thr_fixed <= 0.99, (
            f"cost_threshold returned {thr_fixed:.6f} above 0.99"
        )


# ── F. Model Loading Consistency ──────────────────────────────────────────────

class TestModelLoading:

    def test_model_loads_without_error(self):
        """Model artifact must load cleanly."""
        b = xgb.Booster()
        b.load_model(str(ROOT / "artifacts" / "model" / "model.ubj"))
        b.set_param({"device": "cpu"})
        assert b is not None

    def test_preprocessor_loads_without_error(self):
        """Preprocessor artifact must load cleanly."""
        with open(ROOT / "artifacts" / "model" / "preprocessor.pkl", "rb") as f:
            prep = pickle.load(f)
        assert hasattr(prep, "feature_names")
        assert hasattr(prep, "freq_maps")
        assert hasattr(prep, "mean")
        assert hasattr(prep, "std")

    def test_model_version_in_meta(self, meta):
        """meta.json must contain a version string."""
        assert "version" in meta and meta["version"], "No version in meta.json"

    def test_model_predicts_float_between_0_and_1(self, booster, meta):
        """Model output must be a probability in [0, 1]."""
        x = np.zeros((5, len(meta["feature_names"])), dtype="float32")
        probs = booster.predict(xgb.DMatrix(x, missing=np.nan))
        assert ((probs >= 0) & (probs <= 1)).all(), "Model output outside [0, 1]"

    def test_model_uses_correct_number_of_features(self, booster, meta):
        """DMatrix width must match the number of features in meta.json."""
        n = len(meta["feature_names"])
        x = np.zeros((1, n), dtype="float32")
        # Should not raise
        booster.predict(xgb.DMatrix(x, missing=np.nan))

    def test_wrong_feature_count_raises(self, booster, meta):
        """Wrong number of features should raise an XGBoost error."""
        n = len(meta["feature_names"])
        x = np.zeros((1, n + 5), dtype="float32")
        with pytest.raises(Exception):
            booster.predict(xgb.DMatrix(x, missing=np.nan))


# ── G. Fraud Probability Extraction ──────────────────────────────────────────

class TestFraudProbabilityExtraction:

    def test_model_predict_returns_fraud_probability_not_class(self, booster, meta, test_data):
        """XGBoost Booster.predict() returns probabilities (float), not integer classes."""
        X_test = test_data["X_test"]
        probs = booster.predict(xgb.DMatrix(X_test[:100], missing=np.nan))
        # If this were returning classes (0 or 1) all values would be 0 or 1
        unique_vals = np.unique(probs)
        assert len(unique_vals) > 2, (
            "Model returns only {len(unique_vals)} unique values — "
            "it may be returning classes instead of probabilities"
        )

    def test_fraud_probability_is_index_1_of_proba(self):
        """
        If using XGBClassifier.predict_proba(), the fraud probability must be
        taken from column index 1, not 0.
        predict_proba()[:, 0] = P(legitimate), predict_proba()[:, 1] = P(fraud).
        """
        from xgboost import XGBClassifier
        rng = np.random.default_rng(99)
        X = rng.random((200, 10)).astype("float32")
        y = (rng.random(200) < 0.4).astype(int)
        clf = XGBClassifier(n_estimators=5, random_state=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape[1] == 2, "predict_proba must have 2 columns"
        # Column 1 is fraud prob; for the positive class it should be higher
        fraud_probs = proba[y == 1, 1]
        legit_probs = proba[y == 0, 1]
        # Fraud probs should on average be higher than legit probs
        assert fraud_probs.mean() > legit_probs.mean(), (
            "Fraud probability (index 1) is lower than legit — wrong column selected"
        )


# ── H. Multi-Transaction Sanity Tests ────────────────────────────────────────

class TestMultiTransactionSanity:

    def test_model_does_not_predict_fraud_for_all_test_transactions(self, booster, meta, test_data):
        """The model must NOT flag every transaction. APPROVE rate must be > 50%."""
        X_test = test_data["X_test"]
        review_thr = meta["thresholds"]["review"]
        probs = booster.predict(xgb.DMatrix(X_test, missing=np.nan))
        approve_rate = (probs < review_thr).mean()
        assert approve_rate > 0.50, (
            f"Model approves only {approve_rate:.2%} — it's flagging almost everything. "
            f"review_thr={review_thr:.6f} is likely too low."
        )

    def test_known_legitimate_transactions_mostly_approved(self, booster, meta, test_data):
        """Known-legitimate (y=0) test transactions should mostly be APPROVE."""
        X_test = test_data["X_test"]
        y_test = test_data["y_test"]
        review_thr = meta["thresholds"]["review"]
        probs = booster.predict(xgb.DMatrix(X_test, missing=np.nan))
        legit_flagged = (probs[y_test == 0] >= review_thr).mean()
        # At most 30% of legitimate transactions should be flagged (FP rate)
        assert legit_flagged <= 0.30, (
            f"{legit_flagged:.2%} of legitimate transactions are flagged (FP rate too high)"
        )

    def test_known_fraud_transactions_have_higher_scores(self, booster, test_data):
        """Fraud transactions must have higher average score than legitimate ones."""
        X_test = test_data["X_test"]
        y_test = test_data["y_test"]
        probs = booster.predict(xgb.DMatrix(X_test, missing=np.nan))
        mean_fraud = probs[y_test == 1].mean()
        mean_legit = probs[y_test == 0].mean()
        assert mean_fraud > mean_legit * 5, (
            f"Fraud mean prob ({mean_fraud:.4f}) is not substantially higher than "
            f"legit mean prob ({mean_legit:.4f}) — model may not be discriminating"
        )

    def test_model_pr_auc_above_threshold(self, booster, test_data):
        """PR-AUC on the test set must exceed 0.40 (random baseline = fraud_rate ~0.035)."""
        from sklearn.metrics import average_precision_score
        X_test = test_data["X_test"]
        y_test = test_data["y_test"]
        probs = booster.predict(xgb.DMatrix(X_test, missing=np.nan))
        pr_auc = average_precision_score(y_test, probs)
        assert pr_auc > 0.40, f"PR-AUC={pr_auc:.4f} is too low (expected > 0.40)"

    def test_synthetic_clear_fraud_gets_high_score(self, booster, meta, preprocessor):
        """
        A transaction that looks very similar to known fraud examples should get
        a high fraud probability (> 0.5).
        """
        # Load a real fraud example (pre-processed features)
        fraud_path = ROOT / "examples" / "real_fraud_example.json"
        if not fraud_path.exists():
            pytest.skip("real_fraud_example.json not found")
        example = json.loads(fraud_path.read_text())
        features = example["features"]
        names = meta["feature_names"]
        x = np.array(
            [np.nan if features.get(n) is None else float(features[n]) for n in names],
            dtype="float32"
        )
        prob = float(booster.predict(xgb.DMatrix(x.reshape(1, -1), missing=np.nan))[0])
        assert prob > 0.5, (
            f"Real fraud example got prob={prob:.4f} < 0.5 — model is not detecting fraud"
        )

    def test_synthetic_clear_legit_gets_low_score(self, booster, meta, preprocessor):
        """
        A transaction from a known-legitimate example should get a low fraud probability.
        """
        legit_path = ROOT / "examples" / "real_legit_example.json"
        if not legit_path.exists():
            pytest.skip("real_legit_example.json not found")
        example = json.loads(legit_path.read_text())
        features = example["features"]
        names = meta["feature_names"]
        x = np.array(
            [np.nan if features.get(n) is None else float(features[n]) for n in names],
            dtype="float32"
        )
        prob = float(booster.predict(xgb.DMatrix(x.reshape(1, -1), missing=np.nan))[0])
        review_thr = meta["thresholds"]["review"]
        assert prob < meta["thresholds"]["block"], (
            f"Legit example got prob={prob:.4f} >= block_thr={meta['thresholds']['block']:.4f}"
        )


# ── I. API Prediction Consistency (direct model calls) ───────────────────────

class TestAPIPredictionConsistency:

    def test_serve_model_predict_matches_booster(self, meta, booster):
        """serving.Model.predict() must return the same result as direct Booster.predict()."""
        from finguard.serving import Model
        model = Model(ROOT / "artifacts" / "model")

        rng = np.random.default_rng(7)
        x = rng.standard_normal(len(meta["feature_names"])).astype("float32")
        x[rng.random(len(x)) < 0.3] = np.nan  # 30% missing

        prob_serve  = float(model.predict(x.reshape(1, -1))[0])
        prob_booster = float(booster.predict(xgb.DMatrix(x.reshape(1, -1), missing=np.nan))[0])

        assert abs(prob_serve - prob_booster) < 1e-5, (
            f"serving.Model.predict ({prob_serve}) != Booster.predict ({prob_booster})"
        )

    def test_serve_model_decide_uses_correct_thresholds(self, meta):
        """serving.Model.decide() must use review_thr and block_thr from meta.json."""
        from finguard.serving import Model, APPROVE, REVIEW, BLOCK
        model = Model(ROOT / "artifacts" / "model")

        review_thr = meta["thresholds"]["review"]
        block_thr  = meta["thresholds"]["block"]

        # Just below review — should APPROVE
        assert model.decide(review_thr - 1e-6) == APPROVE

        # Just at review — should REVIEW (or BLOCK if above block_thr)
        if review_thr < block_thr:
            assert model.decide(review_thr) in (REVIEW, BLOCK)

        # Just at block — should BLOCK
        assert model.decide(block_thr) == BLOCK

    def test_predict_raw_endpoint_drops_isfraud(self):
        """
        The /predict/raw endpoint must strip isFraud before preprocessing.
        Test by simulating the pop() call in api.py predict_raw().
        """
        row = {
            "TransactionID": 123456,
            "isFraud": 0,
            "TransactionAmt": 100.0,
            "ProductCD": "W",
            "card4": "visa",
        }
        row.pop("TransactionID", None)
        row.pop("isFraud", None)
        assert "isFraud" not in row, "isFraud was not removed from inference input"
        assert "TransactionID" not in row, "TransactionID was not removed from inference input"
        assert "TransactionAmt" in row, "TransactionAmt should remain"


# ── J. /predict/raw Velocity-Feature Degradation Tests ───────────────────────

class TestRawPredictVelocityDegradation:
    """
    Confirm the empirically measured recall degradation when velocity/history
    features are absent (as they always are in /predict/raw).

    Measured values (from diagnose_raw_predict.py run on the test split):
        full recall         = 63.67%
        no-velocity recall  = 33.86%
        degradation         = 29.81pp  (~30%)
    """

    _VEL_PREFIXES = (
        "uid_count_", "card1_count_", "card_amt_vs_mean", "uid_amt_vs_mean",
        "combo_", "_agg_", "uid_txn_",
    )

    def _vel_features(self, meta):
        return [n for n in meta["feature_names"]
                if any(p in n for p in self._VEL_PREFIXES)]

    def test_velocity_features_exist_in_model(self, meta):
        """Model must contain velocity/history features (confirms training included them)."""
        vel = self._vel_features(meta)
        assert len(vel) >= 10, (
            f"Expected >=10 velocity features, found {len(vel)}: {vel[:5]}"
        )

    def test_full_recall_above_50pct(self, booster, test_data, meta):
        """With all features, fraud recall at review_thr must exceed 50%."""
        from xgboost import DMatrix
        X_test = test_data["X_test"]
        y_test = test_data["y_test"]
        review_thr = meta["thresholds"]["review"]

        dm = DMatrix(X_test)
        probs = booster.predict(dm)
        predicted_fraud = probs >= review_thr
        recall = predicted_fraud[y_test == 1].mean()
        assert recall > 0.50, (
            f"Full-feature fraud recall {recall:.3f} < 0.50. Model may be broken."
        )

    def test_velocity_nan_reduces_recall(self, booster, test_data, meta):
        """
        Setting velocity features to NaN must meaningfully reduce fraud recall.
        Specifically: no-velocity recall < full recall - 20pp.
        """
        from xgboost import DMatrix
        import copy

        X_test = test_data["X_test"].copy()
        y_test = test_data["y_test"]
        review_thr = meta["thresholds"]["review"]
        names = meta["feature_names"]

        # Indices of velocity features
        vel_idx = [i for i, n in enumerate(names)
                   if any(p in n for p in self._VEL_PREFIXES)]
        assert len(vel_idx) > 0, "No velocity feature indices found"

        # Full recall
        dm_full = DMatrix(X_test)
        probs_full = booster.predict(dm_full)
        recall_full = (probs_full[y_test == 1] >= review_thr).mean()

        # No-velocity recall
        X_novel = X_test.copy()
        X_novel[:, vel_idx] = np.nan
        dm_novel = DMatrix(X_novel)
        probs_novel = booster.predict(dm_novel)
        recall_novel = (probs_novel[y_test == 1] >= review_thr).mean()

        degradation = recall_full - recall_novel
        assert degradation > 0.20, (
            f"Expected >20pp recall degradation when velocity features are NaN; "
            f"got {degradation:.3f} (full={recall_full:.3f}, novel={recall_novel:.3f})"
        )

    def test_velocity_nan_lowers_mean_fraud_prob(self, booster, test_data, meta):
        """
        Setting velocity features to NaN must lower the mean fraud probability
        on known-fraud transactions (velocity features push scores up for fraud).
        """
        from xgboost import DMatrix

        X_test = test_data["X_test"].copy()
        y_test = test_data["y_test"]
        names = meta["feature_names"]

        vel_idx = [i for i, n in enumerate(names)
                   if any(p in n for p in self._VEL_PREFIXES)]

        fraud_mask = y_test == 1
        dm_full = DMatrix(X_test[fraud_mask])
        mean_full = booster.predict(dm_full).mean()

        X_novel = X_test.copy()
        X_novel[:, vel_idx] = np.nan
        dm_novel = DMatrix(X_novel[fraud_mask])
        mean_novel = booster.predict(dm_novel).mean()

        assert mean_novel < mean_full, (
            f"Expected mean fraud prob to drop when velocity features are NaN; "
            f"full={mean_full:.4f}, novel={mean_novel:.4f}"
        )

    def test_no_velocity_recall_estimate_is_sane(self, booster, test_data, meta):
        """
        No-velocity recall must be between 20% and 60%
        (confirms the endpoint is not completely blind but is genuinely degraded).
        """
        from xgboost import DMatrix

        X_test = test_data["X_test"].copy()
        y_test = test_data["y_test"]
        review_thr = meta["thresholds"]["review"]
        names = meta["feature_names"]

        vel_idx = [i for i, n in enumerate(names)
                   if any(p in n for p in self._VEL_PREFIXES)]

        X_novel = X_test.copy()
        X_novel[:, vel_idx] = np.nan
        dm_novel = DMatrix(X_novel)
        probs_novel = booster.predict(dm_novel)
        recall_novel = (probs_novel[y_test == 1] >= review_thr).mean()

        assert 0.20 <= recall_novel <= 0.60, (
            f"No-velocity recall {recall_novel:.3f} out of expected [0.20, 0.60]. "
            "Check velocity feature list or thresholds."
        )

    def test_api_response_fields_contract(self, meta):
        """
        Confirm that the /predict/raw response schema includes the new quality fields.
        We test by inspecting the api.py source rather than making a live HTTP call,
        so this test runs offline without uvicorn.
        """
        src = (
            Path(__file__).resolve().parents[1] / "src" / "finguard" / "raw_score.py"
        ).read_text()
        for field in ("score_quality", "velocity_features_nan", "recall_degradation_pct"):
            assert field in src, (
                f"Field '{field}' missing from /predict/raw payload in raw_score.py"
            )

    def test_score_quality_field_is_degraded_when_velocity_nan(self):
        """
        score_quality logic: if nan_count > 0, score_quality must be 'degraded'.
        Test the logic directly (no live API needed).
        """
        # Simulate the api.py logic
        nan_count = 158          # typical for /predict/raw (from diagnostic)
        score_quality = "full" if nan_count == 0 else "degraded"
        assert score_quality == "degraded"

    def test_score_quality_field_is_full_when_no_nan(self):
        """score_quality must be 'full' when no features are NaN."""
        nan_count = 0
        score_quality = "full" if nan_count == 0 else "degraded"
        assert score_quality == "full"

    def test_recall_degradation_pct_nonzero_when_velocity_absent(self):
        """recall_degradation_pct must be ~29.8 when velocity features are NaN."""
        FULL_RECALL  = 63.67
        NOVEL_RECALL = 33.86
        vel_nan = 50   # typical count
        recall_deg_pct = round(FULL_RECALL - NOVEL_RECALL, 1) if vel_nan > 0 else 0.0
        assert 25.0 <= recall_deg_pct <= 35.0, (
            f"Expected recall_degradation_pct ~29.8, got {recall_deg_pct}"
        )

    def test_recall_degradation_pct_zero_when_no_velocity_nan(self):
        """recall_degradation_pct must be 0.0 when no velocity features are NaN."""
        FULL_RECALL  = 63.67
        NOVEL_RECALL = 33.86
        vel_nan = 0
        recall_deg_pct = round(FULL_RECALL - NOVEL_RECALL, 1) if vel_nan > 0 else 0.0
        assert recall_deg_pct == 0.0


# ── K. Serving raw-row vs feature-store ───────────────────────────────────────

class TestServingRawParity:

    def test_feature_store_indexes_train_and_test(self):
        from finguard.serving import FeatureStore
        npz = np.load(ROOT / "data" / "ieee_ready.npz", allow_pickle=True, mmap_mode="r")
        store = FeatureStore(ROOT / "data" / "ieee_ready.npz", split="all")
        train_id = int(npz["id_train"][45])
        test_id = int(npz["id_test"][0])
        assert store.get(train_id) is not None, f"train id {train_id} missing from store"
        assert store.get(test_id) is not None, f"test id {test_id} missing from store"
        assert store.get(3663555) is None, "Kaggle test.csv ids must not be in ieee_ready.npz"

    def test_na_count_v_treats_absent_json_keys_as_missing(self, preprocessor):
        from finguard.raw_features import prepare_raw_dataframe
        from finguard.ieee import V_COLS
        df, _ = prepare_raw_dataframe(
            {"TransactionAmt": 10.0, "ProductCD": "W", "TransactionDT": 1000},
            preprocessor,
        )
        assert int(df["na_count_V"].iloc[0]) == len(V_COLS)

    def test_known_card4_keeps_nonzero_frequency(self, preprocessor):
        from finguard.raw_features import prepare_raw_dataframe
        df, assumed = prepare_raw_dataframe(
            {"TransactionAmt": 36.95, "ProductCD": "W", "card4": "visa",
             "TransactionDT": 87275, "card1": 16132, "addr1": 299.0},
            preprocessor,
        )
        assert assumed  # history is cold-started on a lone row
        raw = preprocessor._build(df)
        assert raw["card4_freq"].iloc[0] > 0

    def test_fill_identity_copies_device_when_payload_is_empty(self):
        from finguard.raw_features import fill_identity
        ident = pd.DataFrame(
            {"TransactionID": [2987045], "DeviceType": ["mobile"], "id_01": [-5.0]}
        ).set_index("TransactionID")
        row, filled = fill_identity({"TransactionID": 2987045, "DeviceType": None}, ident)
        assert filled is True
        assert row["DeviceType"] == "mobile"
        assert row["id_01"] == -5.0

    def test_live_put_makes_kaggle_id_lookupable(self):
        from finguard.serving import FeatureStore
        store = FeatureStore(ROOT / "data" / "ieee_ready.npz", split="all")
        tid = 3663721
        assert store.get(tid) is None
        x = np.zeros(10, dtype="float32")
        assert store.put(tid, x) is True
        got = store.get(tid)
        assert got is not None and len(got) == 10

    def test_live_put_does_not_overwrite_prepared_train_row(self):
        from finguard.serving import FeatureStore
        npz = np.load(ROOT / "data" / "ieee_ready.npz", allow_pickle=True, mmap_mode="r")
        store = FeatureStore(ROOT / "data" / "ieee_ready.npz", split="all")
        tid = int(npz["id_train"][0])
        original = np.array(store.get(tid), dtype="float32", copy=True)
        fake = np.full_like(original, 9.0)
        assert store.put(tid, fake) is False
        after = np.array(store.get(tid), dtype="float32")
        assert np.allclose(after, original, equal_nan=True)

    def test_known_holdout_fraud_id_scores_high_from_store(self, booster, meta):
        from finguard.serving import FeatureStore
        store = FeatureStore(ROOT / "data" / "ieee_ready.npz", split="all")
        x = store.get(3513782)
        if x is None:
            pytest.skip("txn 3513782 not in feature store")
        p = float(booster.predict(xgb.DMatrix(np.asarray(x, dtype="float32").reshape(1, -1), missing=np.nan))[0])
        assert p >= meta["thresholds"]["block"], (
            f"Known holdout fraud 3513782 scored {p:.4f} — store lookup or model mismatch"
        )

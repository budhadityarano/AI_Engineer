"""
pytest examples for ML model testing.
Run: pytest test_model.py -v
"""
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Fixtures ──────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def classification_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture(scope="module")
def regression_data():
    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture(scope="module")
def trained_classifier(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, X_test, y_test


@pytest.fixture(scope="module")
def trained_regressor(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, X_test, y_test


# ── Model Output Shape Tests ────────────────────────────────────
class TestModelOutputShape:
    def test_predict_returns_1d_array(self, trained_classifier):
        model, X_test, _ = trained_classifier
        preds = model.predict(X_test)
        assert preds.ndim == 1, f"Expected 1D, got {preds.ndim}D"

    def test_predict_correct_length(self, trained_classifier):
        model, X_test, y_test = trained_classifier
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)

    def test_predict_proba_shape(self, trained_classifier):
        model, X_test, y_test = trained_classifier
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(y_test), 2)

    def test_proba_sums_to_one(self, trained_classifier):
        model, X_test, _ = trained_classifier
        proba = model.predict_proba(X_test)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ── Performance Threshold Tests ─────────────────────────────────
class TestModelPerformance:
    MIN_ACCURACY = 0.80
    MIN_F1 = 0.75
    MAX_RMSE_RATIO = 0.20  # RMSE must be < 20% of target std

    def test_accuracy_above_threshold(self, trained_classifier):
        model, X_test, y_test = trained_classifier
        acc = accuracy_score(y_test, model.predict(X_test))
        assert acc >= self.MIN_ACCURACY, f"Accuracy {acc:.3f} < threshold {self.MIN_ACCURACY}"

    def test_f1_above_threshold(self, trained_classifier):
        model, X_test, y_test = trained_classifier
        f1 = f1_score(y_test, model.predict(X_test))
        assert f1 >= self.MIN_F1, f"F1 {f1:.3f} < threshold {self.MIN_F1}"

    def test_regression_rmse(self, regression_data, trained_regressor):
        model, X_test, y_test = trained_regressor
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        target_std = y_test.std()
        assert rmse / target_std < self.MAX_RMSE_RATIO, \
            f"RMSE ratio {rmse/target_std:.3f} exceeds {self.MAX_RMSE_RATIO}"

    def test_no_nan_in_predictions(self, trained_classifier):
        model, X_test, _ = trained_classifier
        preds = model.predict(X_test)
        assert not np.isnan(preds).any(), "Predictions contain NaN"


# ── Robustness Tests ──────────────────────────────────────────────
class TestModelRobustness:
    def test_handles_single_sample(self, trained_classifier):
        model, X_test, _ = trained_classifier
        single = X_test[[0]]
        pred = model.predict(single)
        assert len(pred) == 1

    def test_handles_all_same_feature_value(self, trained_classifier):
        model, X_test, _ = trained_classifier
        constant_input = np.zeros_like(X_test)
        try:
            preds = model.predict(constant_input)
            assert len(preds) == len(X_test)
        except Exception as e:
            pytest.fail(f"Model failed on constant input: {e}")

    def test_prediction_is_deterministic(self, trained_classifier):
        model, X_test, _ = trained_classifier
        preds1 = model.predict(X_test)
        preds2 = model.predict(X_test)
        np.testing.assert_array_equal(preds1, preds2)

    def test_feature_importance_sums_to_one(self, trained_classifier):
        model, _, _ = trained_classifier
        importances = model.feature_importances_
        np.testing.assert_allclose(importances.sum(), 1.0, atol=1e-6)


# ── Data Validation Tests ─────────────────────────────────────────
class TestDataValidation:
    def test_no_leakage_between_splits(self, classification_data):
        X_train, X_test, _, _ = classification_data
        # Check sets don't overlap (for datasets with unique rows)
        train_set = set(map(tuple, X_train.round(8)))
        test_set = set(map(tuple, X_test.round(8)))
        overlap = train_set & test_set
        assert len(overlap) == 0, f"Data leakage: {len(overlap)} overlapping rows"

    def test_class_balance_reasonable(self, classification_data):
        _, _, y_train, _ = classification_data
        counts = np.bincount(y_train)
        ratio = counts.min() / counts.max()
        assert ratio > 0.3, f"Severe class imbalance: ratio={ratio:.3f}"

    @pytest.mark.parametrize("n_samples,n_features", [
        (10, 20), (100, 20), (1000, 20)
    ])
    def test_model_scales_with_input_size(self, n_samples, n_features):
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == n_samples

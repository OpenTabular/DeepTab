from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pytest

from deeptab import InferenceModel
from deeptab.models import MLPClassifier, MLPRegressor

# ---------------------------------------------------------------------------
# Shared constants / data helpers
# ---------------------------------------------------------------------------

RANDOM_STATE = 0
FIT_KWARGS: dict[str, Any] = {"max_epochs": 2, "batch_size": 64}
N = 150
N_FEATURES = 5
FEATURE_NAMES = [f"f{i}" for i in range(N_FEATURES)]


def _make_clf_data():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N, N_FEATURES))
    y = rng.integers(0, 2, size=N)
    return pd.DataFrame(X, columns=FEATURE_NAMES), y  # type: ignore[call-overload]


def _make_reg_data():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N, N_FEATURES))
    y = rng.standard_normal(N)
    return pd.DataFrame(X, columns=FEATURE_NAMES), y  # type: ignore[call-overload]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_clf():
    X, y = _make_clf_data()
    clf = MLPClassifier()
    clf.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)
    return clf


@pytest.fixture(scope="module")
def fitted_reg():
    X, y = _make_reg_data()
    reg = MLPRegressor()
    reg.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)
    return reg


@pytest.fixture(scope="module")
def clf_model(fitted_clf):
    return InferenceModel.from_estimator(fitted_clf)


@pytest.fixture(scope="module")
def reg_model(fitted_reg):
    return InferenceModel.from_estimator(fitted_reg)


@pytest.fixture(scope="module")
def X_clf():
    X, _ = _make_clf_data()
    return X


@pytest.fixture(scope="module")
def X_reg():
    X, _ = _make_reg_data()
    return X


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_from_estimator_wraps_fitted(self, fitted_clf):
        model = InferenceModel.from_estimator(fitted_clf)
        assert isinstance(model, InferenceModel)

    def test_from_estimator_raises_on_unfitted(self):
        clf = MLPClassifier()
        with pytest.raises(ValueError, match="unfitted"):
            InferenceModel.from_estimator(clf)

    def test_from_path_round_trip(self, fitted_clf, X_clf):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.deeptab")
            fitted_clf.save(path)
            model = InferenceModel.from_path(path)
        assert isinstance(model, InferenceModel)
        preds = model.predict(X_clf)
        assert preds.shape[0] == len(X_clf)

    def test_from_path_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            InferenceModel.from_path("/nonexistent/path/model.deeptab")


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_task_classification(self, clf_model):
        assert clf_model.task == "classification"

    def test_task_regression(self, reg_model):
        assert reg_model.task == "regression"

    def test_feature_names_returns_list(self, clf_model):
        names = clf_model.feature_names
        assert names == FEATURE_NAMES

    def test_n_features_correct(self, clf_model):
        assert clf_model.n_features == N_FEATURES

    def test_classes_populated_for_classifier(self, clf_model):
        assert clf_model.classes_ is not None
        assert len(clf_model.classes_) == 2

    def test_classes_available_for_regression(self, reg_model):
        # May be None or not present; either is fine
        _ = reg_model.classes_

    def test_task_info_is_dict(self, clf_model):
        info = clf_model.task_info
        assert isinstance(info, dict)

    def test_feature_schema_is_dict(self, clf_model):
        schema = clf_model.feature_schema
        assert isinstance(schema, dict)


# ---------------------------------------------------------------------------
# validate_input
# ---------------------------------------------------------------------------


class TestValidateInput:
    def test_exact_match_returns_dataframe(self, clf_model, X_clf):
        out = clf_model.validate_input(X_clf)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == FEATURE_NAMES

    def test_reorders_columns(self, clf_model, X_clf):
        shuffled = X_clf[FEATURE_NAMES[::-1]]
        out = clf_model.validate_input(shuffled)
        assert list(out.columns) == FEATURE_NAMES

    def test_missing_column_raises(self, clf_model, X_clf):
        X_bad = X_clf.drop(columns=["f0"])
        with pytest.raises(ValueError, match="missing"):
            clf_model.validate_input(X_bad)

    def test_extra_column_raises_by_default(self, clf_model, X_clf):
        X_extra = X_clf.copy()
        X_extra["extra_col"] = 0.0
        with pytest.raises(ValueError, match="unexpected"):
            clf_model.validate_input(X_extra)

    def test_extra_column_dropped_with_warning(self, clf_model, X_clf):
        X_extra = X_clf.copy()
        X_extra["extra_col"] = 0.0
        with pytest.warns(UserWarning, match="not seen during training"):
            out = clf_model.validate_input(X_extra, allow_extra_columns=True)
        assert "extra_col" not in out.columns
        assert list(out.columns) == FEATURE_NAMES

    def test_array_input_accepted(self, clf_model, X_clf):
        # When passed as a numpy array there are no named columns, so
        # only the count check applies (names can't be verified).
        arr = X_clf.values
        # Without named columns the count check should pass silently
        # (the DataFrame will have integer columns 0..N_FEATURES-1)
        # If feature_names is set, integer column names won't match the
        # stored string names; validate_input should raise on missing cols.
        with pytest.raises(ValueError):
            clf_model.validate_input(arr)


# ---------------------------------------------------------------------------
# Prediction — classification
# ---------------------------------------------------------------------------


class TestPredictClassifier:
    @pytest.mark.smoke
    def test_predict_shape(self, clf_model, X_clf):
        preds = clf_model.predict(X_clf)
        assert preds.shape == (N,)

    def test_predict_proba_shape(self, clf_model, X_clf):
        proba = clf_model.predict_proba(X_clf)
        assert proba.shape == (N, 2)

    def test_predict_proba_sums_to_one(self, clf_model, X_clf):
        proba = clf_model.predict_proba(X_clf)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(N), atol=1e-5)

    def test_predict_validates_input(self, clf_model, X_clf):
        X_bad = X_clf.drop(columns=["f0"])
        with pytest.raises(ValueError, match="missing"):
            clf_model.predict(X_bad)

    def test_predict_proba_validates_input(self, clf_model, X_clf):
        X_bad = X_clf.drop(columns=["f1"])
        with pytest.raises(ValueError, match="missing"):
            clf_model.predict_proba(X_bad)


# ---------------------------------------------------------------------------
# Prediction — regression
# ---------------------------------------------------------------------------


class TestPredictRegressor:
    @pytest.mark.smoke
    def test_predict_shape(self, reg_model, X_reg):
        preds = reg_model.predict(X_reg)
        assert preds.shape == (N,)

    def test_predict_proba_raises_type_error(self, reg_model, X_reg):
        with pytest.raises(TypeError, match="classification"):
            reg_model.predict_proba(X_reg)

    def test_predict_params_raises_type_error(self, reg_model, X_reg):
        with pytest.raises(TypeError, match="distributional"):
            reg_model.predict_params(X_reg)


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


class TestInspection:
    def test_describe_contains_inference_task(self, clf_model):
        info = clf_model.describe()
        assert "inference_task" in info
        assert info["inference_task"] == "classification"

    def test_runtime_info_is_dict(self, clf_model):
        info = clf_model.runtime_info()
        assert isinstance(info, dict)

    def test_parameter_table_returns_dataframe(self, clf_model):
        df = clf_model.parameter_table()
        assert isinstance(df, pd.DataFrame)
        assert "num_params" in df.columns

    def test_repr_contains_key_info(self, clf_model):
        r = repr(clf_model)
        assert "InferenceModel" in r
        assert "classification" in r
        assert "MLPClassifier" in r
        assert str(N_FEATURES) in r

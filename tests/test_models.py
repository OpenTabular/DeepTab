"""
End-to-end behavioural tests for the sklearn-compatible model API.

Tests cover fit → predict → evaluate for all 15 stable models across all
three task variants (Classifier, Regressor, LSS).  A small synthetic dataset
keeps CI fast.
"""

import platform
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from deeptab.models import (
    ENODELSS,
    MLPLSS,
    NDTFLSS,
    NODELSS,
    SAINTLSS,
    AutoIntClassifier,
    AutoIntLSS,
    AutoIntRegressor,
    ENODEClassifier,
    ENODERegressor,
    FTTransformerClassifier,
    FTTransformerLSS,
    FTTransformerRegressor,
    MambaTabClassifier,
    MambaTabLSS,
    MambaTabRegressor,
    MambAttentionClassifier,
    MambAttentionLSS,
    MambAttentionRegressor,
    MambularClassifier,
    MambularLSS,
    MambularRegressor,
    MLPClassifier,
    MLPRegressor,
    NDTFClassifier,
    NDTFRegressor,
    NODEClassifier,
    NODERegressor,
    ResNetClassifier,
    ResNetLSS,
    ResNetRegressor,
    SAINTClassifier,
    SAINTRegressor,
    TabMClassifier,
    TabMLSS,
    TabMRegressor,
    TabRClassifier,
    TabRLSS,
    TabRRegressor,
    TabTransformerClassifier,
    TabTransformerLSS,
    TabTransformerRegressor,
    TabulaRNNClassifier,
    TabulaRNNLSS,
    TabulaRNNRegressor,
)

_macos_arm64 = platform.system() == "Darwin" and platform.machine() == "arm64"
_skip_tabr = pytest.mark.skipif(
    _macos_arm64,
    reason="faiss-cpu from PyPI segfaults on macOS arm64; install via conda for TabR support",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SAMPLES = 200
N_FEATURES = 6
N_CLASSES = 3
RANDOM_STATE = 0
FIT_KWARGS: dict[str, Any] = {"max_epochs": 2, "batch_size": 64}


@pytest.fixture(scope="module")
def classification_data():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y_cont = X @ rng.standard_normal(N_FEATURES) + rng.standard_normal(N_SAMPLES)
    y = pd.qcut(y_cont, q=N_CLASSES, labels=False)
    df = pd.DataFrame({f"f{i}": X[:, i] for i in range(N_FEATURES)})
    return train_test_split(df, y, test_size=0.2, random_state=RANDOM_STATE)


@pytest.fixture(scope="module")
def binary_classification_data():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y_cont = X @ rng.standard_normal(N_FEATURES) + rng.standard_normal(N_SAMPLES)
    y = np.where(y_cont > np.median(y_cont), 1, 0)
    df = pd.DataFrame({f"f{i}": X[:, i] for i in range(N_FEATURES)})
    return train_test_split(df, y, test_size=0.2, random_state=RANDOM_STATE)


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y = X @ rng.standard_normal(N_FEATURES) + rng.standard_normal(N_SAMPLES)
    df = pd.DataFrame({f"f{i}": X[:, i] for i in range(N_FEATURES)})
    return train_test_split(df, y, test_size=0.2, random_state=RANDOM_STATE)


@pytest.fixture(scope="module")
def classification_data_with_cat():
    """Fixture with one categorical column — required by TabTransformer."""
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y_cont = X @ rng.standard_normal(N_FEATURES) + rng.standard_normal(N_SAMPLES)
    y = pd.qcut(y_cont, q=N_CLASSES, labels=False)
    df = pd.DataFrame({f"f{i}": X[:, i] for i in range(N_FEATURES)})
    df["cat_col"] = rng.choice(["A", "B", "C"], size=N_SAMPLES)
    return train_test_split(df, y, test_size=0.2, random_state=RANDOM_STATE)


@pytest.fixture(scope="module")
def regression_data_with_cat():
    """Fixture with one categorical column — required by TabTransformer."""
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y = X @ rng.standard_normal(N_FEATURES) + rng.standard_normal(N_SAMPLES)
    df = pd.DataFrame({f"f{i}": X[:, i] for i in range(N_FEATURES)})
    df["cat_col"] = rng.choice(["A", "B", "C"], size=N_SAMPLES)
    return train_test_split(df, y, test_size=0.2, random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------

CLASSIFIERS = [
    MLPClassifier,
    ResNetClassifier,
    FTTransformerClassifier,
    MambularClassifier,
    TabMClassifier,
    pytest.param(TabRClassifier, marks=_skip_tabr),
    NODEClassifier,
    NDTFClassifier,
    SAINTClassifier,
    AutoIntClassifier,
    MambaTabClassifier,
    MambAttentionClassifier,
    TabulaRNNClassifier,
    ENODEClassifier,
]


@pytest.mark.parametrize("cls", CLASSIFIERS)
def test_classifier_fit_predict_shape(cls, classification_data):
    X_train, X_test, y_train, _y_test = classification_data
    model = cls()
    model.fit(X_train, y_train, **FIT_KWARGS)

    assert model.n_features_in_ == X_train.shape[1]
    np.testing.assert_array_equal(model.feature_names_in_, np.asarray(X_train.columns, dtype=object))
    np.testing.assert_array_equal(model.classes_, np.unique(y_train))

    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),), f"{cls.__name__}.predict returned unexpected shape"
    assert set(preds).issubset(set(range(N_CLASSES))), f"{cls.__name__}.predict returned out-of-range labels"


@pytest.mark.parametrize("cls", CLASSIFIERS)
def test_classifier_predict_proba_shape(cls, classification_data):
    X_train, X_test, y_train, _y_test = classification_data
    model = cls()
    model.fit(X_train, y_train, **FIT_KWARGS)

    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), N_CLASSES), f"{cls.__name__}.predict_proba returned unexpected shape"
    np.testing.assert_allclose(
        proba.sum(axis=1),
        np.ones(len(X_test)),
        atol=1e-5,
        err_msg=f"{cls.__name__}.predict_proba rows do not sum to 1",
    )


@pytest.mark.parametrize("cls", CLASSIFIERS)
def test_classifier_evaluate_returns_dict(cls, classification_data):
    X_train, X_test, y_train, y_test = classification_data
    model = cls()
    model.fit(X_train, y_train, **FIT_KWARGS)

    metrics = model.evaluate(X_test, y_test)
    assert isinstance(metrics, dict), f"{cls.__name__}.evaluate should return a dict"
    assert len(metrics) > 0, f"{cls.__name__}.evaluate returned an empty dict"


@pytest.mark.smoke
def test_classifier_binary_predict_proba_and_score(binary_classification_data):
    X_train, X_test, y_train, y_test = binary_classification_data
    model = MLPClassifier()
    model.fit(X_train, y_train, **FIT_KWARGS)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    score = model.score(X_test, y_test)

    assert set(preds).issubset({0, 1})
    assert proba.shape == (len(X_test), 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X_test)), atol=1e-5)
    assert 0.0 <= score <= 1.0


@pytest.mark.smoke
def test_predict_validates_feature_names(classification_data):
    X_train, X_test, y_train, _y_test = classification_data
    model = MLPClassifier()
    model.fit(X_train, y_train, **FIT_KWARGS)

    from deeptab.core.exceptions import ColumnNameError

    with pytest.raises(ColumnNameError):
        model.predict(X_test[X_test.columns[::-1]])


# ---------------------------------------------------------------------------
# Regressor tests
# ---------------------------------------------------------------------------

REGRESSORS = [
    MLPRegressor,
    ResNetRegressor,
    FTTransformerRegressor,
    MambularRegressor,
    TabMRegressor,
    pytest.param(TabRRegressor, marks=_skip_tabr),
    NODERegressor,
    NDTFRegressor,
    SAINTRegressor,
    AutoIntRegressor,
    MambaTabRegressor,
    MambAttentionRegressor,
    TabulaRNNRegressor,
    ENODERegressor,
]


@pytest.mark.parametrize("cls", REGRESSORS)
def test_regressor_fit_predict_shape(cls, regression_data):
    X_train, X_test, y_train, _y_test = regression_data
    model = cls()
    model.fit(X_train, y_train, **FIT_KWARGS)

    assert model.n_features_in_ == X_train.shape[1]
    np.testing.assert_array_equal(model.feature_names_in_, np.asarray(X_train.columns, dtype=object))

    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),), f"{cls.__name__}.predict returned unexpected shape"
    assert np.isfinite(preds).all(), f"{cls.__name__}.predict returned non-finite values"


@pytest.mark.parametrize("cls", REGRESSORS)
def test_regressor_evaluate_returns_dict(cls, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    model = cls()
    model.fit(X_train, y_train, **FIT_KWARGS)

    metrics = model.evaluate(X_test, y_test)
    assert isinstance(metrics, dict), f"{cls.__name__}.evaluate should return a dict"
    assert len(metrics) > 0, f"{cls.__name__}.evaluate returned an empty dict"


@pytest.mark.smoke
def test_regressor_score_returns_r2(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    model = MLPRegressor()
    model.fit(X_train, y_train, **FIT_KWARGS)

    score = model.score(X_test, y_test)
    assert isinstance(score, float), "score() should return a float"
    assert score <= 1.0, "R² score should be at most 1.0"


@pytest.mark.parametrize("cls", CLASSIFIERS)
def test_classifier_score_returns_float_in_unit_interval(cls, classification_data):
    """score() returns a float in [0, 1] for every classifier."""
    X_train, X_test, y_train, y_test = classification_data
    model = cls()
    model.fit(X_train, y_train, **FIT_KWARGS)

    score = model.score(X_test, y_test)
    assert isinstance(score, float), f"{cls.__name__}.score() should return a float"
    assert 0.0 <= score <= 1.0, f"{cls.__name__}.score()={score} is outside [0, 1]"


@pytest.mark.parametrize("cls", REGRESSORS)
def test_regressor_score_returns_r2_all(cls, regression_data):
    """score() returns an R² float ≤ 1.0 for every regressor."""
    X_train, X_test, y_train, y_test = regression_data
    model = cls()
    model.fit(X_train, y_train, **FIT_KWARGS)

    score = model.score(X_test, y_test)
    assert isinstance(score, float), f"{cls.__name__}.score() should return a float"
    assert score <= 1.0, f"{cls.__name__}.score()={score} exceeds 1.0"


# ---------------------------------------------------------------------------
# LSS (distributional regression) tests
# ---------------------------------------------------------------------------

LSS_MODELS = [
    MLPLSS,
    ResNetLSS,
    FTTransformerLSS,
    MambularLSS,
    TabMLSS,
    pytest.param(TabRLSS, marks=_skip_tabr),
    NODELSS,
    NDTFLSS,
    SAINTLSS,
    AutoIntLSS,
    MambaTabLSS,
    MambAttentionLSS,
    TabulaRNNLSS,
    ENODELSS,
]


@pytest.mark.parametrize("cls", LSS_MODELS)
def test_lss_fit_predict_shape(cls, regression_data):
    X_train, X_test, y_train, _y_test = regression_data
    model = cls()
    model.fit(X_train, y_train, family="normal", **FIT_KWARGS)

    assert model.n_features_in_ == X_train.shape[1]
    np.testing.assert_array_equal(model.feature_names_in_, np.asarray(X_train.columns, dtype=object))

    preds = model.predict(X_test)
    # predict returns the location parameter for the normal family
    assert preds.shape[0] == len(X_test), f"{cls.__name__}.predict returned unexpected first dimension"
    assert np.isfinite(preds).all(), f"{cls.__name__}.predict returned non-finite values"


@pytest.mark.parametrize("cls", LSS_MODELS)
def test_lss_evaluate_returns_dict(cls, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    model = cls()
    model.fit(X_train, y_train, family="normal", **FIT_KWARGS)

    metrics = model.evaluate(X_test, y_test)
    assert isinstance(metrics, dict), f"{cls.__name__}.evaluate should return a dict"
    assert len(metrics) > 0, f"{cls.__name__}.evaluate returned an empty dict"


# ---------------------------------------------------------------------------
# Config serialisation round-trip (Requirement 5)
# ---------------------------------------------------------------------------

ALL_ESTIMATOR_CLASSES = (
    CLASSIFIERS + REGRESSORS + LSS_MODELS + [TabTransformerClassifier, TabTransformerRegressor, TabTransformerLSS]
)


@pytest.mark.parametrize("cls", ALL_ESTIMATOR_CLASSES)
def test_config_serialisation_roundtrip(cls):
    """get_params() → construct new instance → config values survive."""
    model = cls()
    params = model.get_params()

    # Constructing a second instance with the same params must not raise.
    model2 = cls(**params)

    # All config kwargs must round-trip exactly.
    for key, value in model._config_kwargs.items():
        assert getattr(model2.config, key, object()) == value, (
            f"{cls.__name__}: config.{key}={value!r} did not survive get_params round-trip"
        )


# ---------------------------------------------------------------------------
# TabTransformer — requires at least one categorical feature
# ---------------------------------------------------------------------------

TAB_TRANSFORMER_MODELS = [
    (TabTransformerClassifier, "classification"),
    (TabTransformerRegressor, "regression"),
    (TabTransformerLSS, "lss"),
]


@pytest.mark.parametrize("cls,task", TAB_TRANSFORMER_MODELS)
def test_tabtransformer_fit_predict(cls, task, classification_data_with_cat, regression_data_with_cat):
    if task == "classification":
        X_train, X_test, y_train, _y_test = classification_data_with_cat
    else:
        X_train, X_test, y_train, _y_test = regression_data_with_cat

    model = cls()
    if task == "lss":
        model.fit(X_train, y_train, family="normal", **FIT_KWARGS)
    else:
        model.fit(X_train, y_train, **FIT_KWARGS)

    preds = model.predict(X_test)
    assert preds.shape[0] == len(X_test), f"{cls.__name__}.predict returned unexpected shape"
    assert np.isfinite(preds).all(), f"{cls.__name__}.predict returned non-finite values"

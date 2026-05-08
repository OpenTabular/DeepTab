"""
End-to-end behavioural tests for the sklearn-compatible model API.

Tests cover fit → predict → evaluate for a representative subset of models
(MLP, ResNet, FTTransformer, Mambular) across all three task variants
(Classifier, Regressor, LSS).  A small synthetic dataset keeps CI fast.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from deeptab.models import (
    MLPLSS,
    FTTransformerClassifier,
    FTTransformerLSS,
    FTTransformerRegressor,
    MambularClassifier,
    MambularLSS,
    MambularRegressor,
    MLPClassifier,
    MLPRegressor,
    ResNetClassifier,
    ResNetLSS,
    ResNetRegressor,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SAMPLES = 200
N_FEATURES = 6
N_CLASSES = 3
RANDOM_STATE = 0
FIT_KWARGS = {"max_epochs": 2, "batch_size": 64}


@pytest.fixture(scope="module")
def classification_data():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y_cont = X @ rng.standard_normal(N_FEATURES) + rng.standard_normal(N_SAMPLES)
    y = pd.qcut(y_cont, q=N_CLASSES, labels=False)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)])
    return train_test_split(df, y, test_size=0.2, random_state=RANDOM_STATE)


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y = X @ rng.standard_normal(N_FEATURES) + rng.standard_normal(N_SAMPLES)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)])
    return train_test_split(df, y, test_size=0.2, random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------

CLASSIFIERS = [
    MLPClassifier,
    ResNetClassifier,
    FTTransformerClassifier,
    MambularClassifier,
]


@pytest.mark.parametrize("cls", CLASSIFIERS)
def test_classifier_fit_predict_shape(cls, classification_data):
    X_train, X_test, y_train, _y_test = classification_data
    model = cls()
    model.fit(X_train, y_train, **FIT_KWARGS)

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


# ---------------------------------------------------------------------------
# Regressor tests
# ---------------------------------------------------------------------------

REGRESSORS = [
    MLPRegressor,
    ResNetRegressor,
    FTTransformerRegressor,
    MambularRegressor,
]


@pytest.mark.parametrize("cls", REGRESSORS)
def test_regressor_fit_predict_shape(cls, regression_data):
    X_train, X_test, y_train, _y_test = regression_data
    model = cls()
    model.fit(X_train, y_train, **FIT_KWARGS)

    preds = model.predict(X_test)
    assert preds.shape[0] == len(X_test), f"{cls.__name__}.predict returned unexpected shape"
    assert np.isfinite(preds).all(), f"{cls.__name__}.predict returned non-finite values"


@pytest.mark.parametrize("cls", REGRESSORS)
def test_regressor_evaluate_returns_dict(cls, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    model = cls()
    model.fit(X_train, y_train, **FIT_KWARGS)

    metrics = model.evaluate(X_test, y_test)
    assert isinstance(metrics, dict), f"{cls.__name__}.evaluate should return a dict"
    assert len(metrics) > 0, f"{cls.__name__}.evaluate returned an empty dict"


# ---------------------------------------------------------------------------
# LSS (distributional regression) tests
# ---------------------------------------------------------------------------

LSS_MODELS = [
    MLPLSS,
    ResNetLSS,
    FTTransformerLSS,
    MambularLSS,
]


@pytest.mark.parametrize("cls", LSS_MODELS)
def test_lss_fit_predict_shape(cls, regression_data):
    X_train, X_test, y_train, _y_test = regression_data
    model = cls()
    model.fit(X_train, y_train, family="normal", **FIT_KWARGS)

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

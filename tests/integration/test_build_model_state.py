"""build_model() replaces the network with untrained weights, so the estimator
must not keep counting as fitted from an earlier fit()."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from deeptab.core.exceptions import NotFittedError
from deeptab.models import MLPLSS, MLPClassifier, MLPRegressor

RANDOM_STATE = 0

FIT_KWARGS: dict[str, Any] = {"max_epochs": 1, "batch_size": 32}


def _make_reg_data():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((120, 4))
    y = rng.standard_normal(120)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(4)]), y  # type: ignore[call-overload]


def test_regressor_is_unfitted_after_rebuilding(tmp_path):
    X, y = _make_reg_data()
    reg = MLPRegressor().fit(X, y, random_state=RANDOM_STATE, checkpoint_path=str(tmp_path), **FIT_KWARGS)
    assert reg.is_fitted_ is True
    assert reg._best_model_path

    reg.build_model(X, y)
    assert reg.is_fitted_ is False
    assert reg._best_model_path is None
    with pytest.raises(NotFittedError):
        reg.predict(X)

    reg.fit(X, y, rebuild=False, checkpoint_path=str(tmp_path), **FIT_KWARGS)
    assert reg.is_fitted_ is True


def test_lss_is_unfitted_after_rebuilding(tmp_path):
    X, y = _make_reg_data()
    model = MLPLSS().fit(X, y, family="normal", random_state=RANDOM_STATE, checkpoint_path=str(tmp_path), **FIT_KWARGS)
    assert model.is_fitted_ is True

    model.build_model(X, y, family="normal")
    assert model.is_fitted_ is False
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_classifier_is_unfitted_when_rebuilding_fails(tmp_path):
    X, _ = _make_reg_data()
    y = np.arange(len(X)) % 3
    clf = MLPClassifier().fit(X, y, random_state=RANDOM_STATE, checkpoint_path=str(tmp_path), **FIT_KWARGS)
    assert clf.is_fitted_ is True

    two_classes = y % 2
    with pytest.raises(ValueError, match="not seen during fit"):
        clf.build_model(X[:100], two_classes[:100], X_val=X[100:], y_val=y[100:])
    assert clf.is_fitted_ is False
    with pytest.raises(NotFittedError):
        clf.predict(X)

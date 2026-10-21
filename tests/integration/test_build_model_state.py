"""build_model() replaces the network with untrained weights, so the estimator
must not keep counting as fitted from an earlier fit()."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from deeptab.core.exceptions import NotFittedError
from deeptab.models import MLPLSS, MLPRegressor

RANDOM_STATE = 0

FIT_KWARGS: dict[str, Any] = {"max_epochs": 1, "batch_size": 32}


def _make_reg_data():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((120, 4))
    y = rng.standard_normal(120)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(4)]), y  # type: ignore[call-overload]


def test_regressor_is_unfitted_after_rebuilding():
    X, y = _make_reg_data()
    reg = MLPRegressor().fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)
    assert reg.is_fitted_ is True

    reg.build_model(X, y)
    assert reg.is_fitted_ is False
    with pytest.raises(NotFittedError):
        reg.predict(X)

    reg.fit(X, y, rebuild=False, **FIT_KWARGS)
    assert reg.is_fitted_ is True


def test_lss_is_unfitted_after_rebuilding():
    X, y = _make_reg_data()
    model = MLPLSS().fit(X, y, family="normal", random_state=RANDOM_STATE, **FIT_KWARGS)
    assert model.is_fitted_ is True

    model.build_model(X, y, family="normal")
    assert model.is_fitted_ is False
    with pytest.raises(NotFittedError):
        model.predict(X)

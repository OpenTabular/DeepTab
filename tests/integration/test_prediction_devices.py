"""Tests for prediction devices."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from deeptab.models import MLPRegressor

RANDOM_STATE = 0

FIT_KWARGS: dict[str, Any] = {"max_epochs": 1, "batch_size": 32}

N = 120

N_FEATURES = 4


def _make_reg_data():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N, N_FEATURES))
    y = rng.standard_normal(N)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)]), y  # type: ignore[call-overload]


class TestPredictDeviceOverride:
    def test_explicit_cpu_device_matches_default(self):
        X, y = _make_reg_data()
        reg = MLPRegressor()
        reg.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)

        default_preds = reg.predict(X)
        cpu_preds = reg.predict(X, device="cpu")
        np.testing.assert_array_almost_equal(default_preds, cpu_preds, decimal=4)

    def test_invalid_device_string_raises(self):
        X, y = _make_reg_data()
        reg = MLPRegressor()
        reg.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)

        with pytest.raises(RuntimeError, match="Invalid device string"):
            reg.predict(X, device="not-a-real-device")

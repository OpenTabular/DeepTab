"""Tests for regression metrics."""

from __future__ import annotations

import numpy as np
import pytest

from deeptab.metrics import (  # Classification; Distributional; Registry; Base; Regression
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    PinballLoss,
    R2Score,
    RootMeanSquaredError,
)

RNG = np.random.default_rng(42)

N = 100


@pytest.fixture
def reg_data():
    """Regression targets and predictions (1-D)."""
    y_true = RNG.normal(0.0, 1.0, N)
    y_pred = y_true + RNG.normal(0.0, 0.1, N)  # near-perfect
    return y_true, y_pred


@pytest.fixture
def lss_data():
    """LSS predictions as 2-D array: [mean, scale]."""
    y_true = RNG.normal(0.0, 1.0, N)
    means = y_true + RNG.normal(0.0, 0.1, N)
    scales = np.abs(RNG.normal(0.5, 0.1, N)) + 0.1
    y_pred_2d = np.column_stack([means, scales])
    return y_true, y_pred_2d


class TestRegressionMetrics:
    def test_mse_returns_float(self, reg_data):
        y_true, y_pred = reg_data
        assert isinstance(MeanSquaredError()(y_true, y_pred), float)

    def test_rmse_returns_float(self, reg_data):
        y_true, y_pred = reg_data
        assert isinstance(RootMeanSquaredError()(y_true, y_pred), float)

    def test_mae_returns_float(self, reg_data):
        y_true, y_pred = reg_data
        assert isinstance(MeanAbsoluteError()(y_true, y_pred), float)

    def test_r2_returns_float(self, reg_data):
        y_true, y_pred = reg_data
        assert isinstance(R2Score()(y_true, y_pred), float)

    def test_rmse_geq_mae(self, reg_data):
        """RMSE >= MAE by the QM-AM inequality."""
        y_true, y_pred = reg_data
        assert RootMeanSquaredError()(y_true, y_pred) >= MeanAbsoluteError()(y_true, y_pred)

    def test_mse_is_rmse_squared(self, reg_data):
        y_true, y_pred = reg_data
        mse = MeanSquaredError()(y_true, y_pred)
        rmse = RootMeanSquaredError()(y_true, y_pred)
        assert abs(mse - rmse**2) < 1e-9

    def test_perfect_predictions_give_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert MeanSquaredError()(y, y) == pytest.approx(0.0)
        assert MeanAbsoluteError()(y, y) == pytest.approx(0.0)
        assert RootMeanSquaredError()(y, y) == pytest.approx(0.0)

    def test_perfect_r2(self):
        y = np.array([1.0, 2.0, 3.0])
        assert R2Score()(y, y) == pytest.approx(1.0)

    def test_r2_bounded_above_by_one(self, reg_data):
        y_true, y_pred = reg_data
        assert R2Score()(y_true, y_pred) <= 1.0 + 1e-9

    def test_2d_lss_array_uses_first_column(self, lss_data):
        """Metrics on 2-D parameter arrays must use column 0 as the mean."""
        y_true, y_pred_2d = lss_data
        y_pred_1d = y_pred_2d[:, 0]
        for Metric in [MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError, R2Score]:
            v_2d = Metric()(y_true, y_pred_2d)
            v_1d = Metric()(y_true, y_pred_1d)
            assert v_2d == pytest.approx(v_1d, rel=1e-6), f"{Metric.__name__}: 2-D result {v_2d} != 1-D result {v_1d}"

    def test_mape_nonnegative(self, reg_data):
        y_true, y_pred = reg_data
        assert MeanAbsolutePercentageError()(y_true, y_pred) >= 0.0

    def test_pinball_at_median_approx_half_mae(self, reg_data):
        """Pinball at tau=0.5 equals 0.5 * MAE."""
        y_true, y_pred = reg_data
        pb = PinballLoss(quantile=0.5)(y_true, y_pred)
        mae = MeanAbsoluteError()(y_true, y_pred)
        assert pb == pytest.approx(0.5 * mae, rel=1e-5)

    def test_pinball_invalid_quantile(self):
        with pytest.raises(ValueError):
            PinballLoss(quantile=0.0)
        with pytest.raises(ValueError):
            PinballLoss(quantile=1.5)

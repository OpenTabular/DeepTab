"""Tests for distributional metrics."""

from __future__ import annotations

import numpy as np
import pytest

from deeptab.distributions import Quantile
from deeptab.metrics import (  # Classification; Distributional; Registry; Base; Regression
    CRPS,
    BetaBrierScore,
    CoverageProbability,
    DirichletError,
    GammaDeviance,
    IntervalScore,
    NegativeBinomialDeviance,
    PoissonDeviance,
    SharpnessScore,
    StudentTLoss,
    TweedieDeviance,
    get_default_metrics,
    get_default_metrics_dict,
)
from deeptab.metrics.registry import get_default_lss_metrics


@pytest.mark.parametrize(
    "family,parameters,mean",
    [
        ("gamma", [[6.0, 2.0], [8.0, 4.0]], [3.0, 2.0]),
        ("zip", [[0.25, 4.0], [0.5, 8.0]], [3.0, 4.0]),
    ],
)
def test_default_lss_rmse_uses_derived_distribution_mean(family, parameters, mean):
    metrics = get_default_metrics_dict("lss", family=family)
    targets = np.array([1.0, 5.0])
    expected = np.sqrt(np.mean((targets - np.asarray(mean)) ** 2))
    assert metrics["rmse"](targets, np.asarray(parameters)) == pytest.approx(expected)


def test_default_tweedie_metric_uses_fitted_power():
    from deeptab.distributions import get_distribution

    metrics = get_default_metrics_dict("lss", family=get_distribution("tweedie", p=1.7))
    deviance = metrics["tweedie_deviance"]
    assert isinstance(deviance, TweedieDeviance)
    assert deviance.p == 1.7


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


@pytest.fixture
def count_data():
    """Count targets (non-negative integers) and predicted means."""
    y_true = RNG.poisson(lam=3.0, size=N).astype(float)
    y_pred = np.clip(y_true + RNG.normal(0.0, 0.5, N), 0.01, None)
    return y_true, y_pred


@pytest.fixture
def proportion_data():
    """Proportion targets in (0, 1) and predicted means."""
    y_true = np.clip(RNG.beta(2.0, 5.0, N), 1e-4, 1 - 1e-4)
    y_pred = np.clip(y_true + RNG.normal(0.0, 0.05, N), 1e-4, 1 - 1e-4)
    return y_true, y_pred


class TestDistributionalMetrics:
    def test_crps_nonnegative(self, lss_data):
        y_true, y_pred = lss_data
        assert CRPS(family="normal")(y_true, y_pred) >= 0.0

    def test_crps_returns_float(self, lss_data):
        y_true, y_pred = lss_data
        assert isinstance(CRPS(family="normal")(y_true, y_pred), float)

    def test_crps_lower_for_better_predictions(self):
        """A near-perfect predictor should have lower CRPS than a bad one."""
        rng = np.random.default_rng(0)
        y_true = rng.normal(0, 1, 200)
        good = np.column_stack([y_true + rng.normal(0, 0.05, 200), np.ones(200) * 0.1])
        bad = np.column_stack([rng.normal(0, 1, 200), np.ones(200) * 2.0])
        assert CRPS(family="normal")(y_true, good) < CRPS(family="normal")(y_true, bad)

    def test_poisson_deviance_nonneg(self, count_data):
        y_true, y_pred = count_data
        assert PoissonDeviance()(y_true, y_pred) >= 0.0

    def test_poisson_deviance_zero_for_perfect(self, count_data):
        """Deviance is 0 when predictions equal targets exactly."""
        y_true, _ = count_data
        y_pred = np.clip(y_true, 1e-9, None)
        assert PoissonDeviance()(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)

    def test_gamma_deviance_zero_for_perfect(self):
        """Gamma deviance is 0 when predictions equal targets exactly."""
        y = np.abs(RNG.normal(1.0, 0.5, N)) + 0.1
        assert GammaDeviance()(y, y) == pytest.approx(0.0, abs=1e-6)

    def test_gamma_deviance_returns_float(self):
        y_true = np.abs(RNG.normal(1.0, 0.5, N)) + 0.1
        y_pred = np.abs(y_true + RNG.normal(0, 0.1, N)) + 0.1
        assert isinstance(GammaDeviance()(y_true, y_pred), float)

    def test_gamma_deviance_matches_sklearn(self):
        """Regression test: the log term's sign must match the deviance definition."""
        from sklearn.metrics import mean_gamma_deviance

        y_true = np.abs(RNG.normal(1.0, 0.5, N)) + 0.1
        y_pred = np.abs(y_true + RNG.normal(0, 0.3, N)) + 0.1
        assert GammaDeviance()(y_true, y_pred) == pytest.approx(mean_gamma_deviance(y_true, y_pred), rel=1e-6)

    def test_gamma_deviance_nonnegative_for_overprediction(self):
        """Regression test: deviance must not reward extreme over-prediction."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.full(3, 1000.0)
        assert GammaDeviance()(y_true, y_pred) > 0.0

    def test_tweedie_deviance_nonneg(self, reg_data):
        y_true = np.abs(reg_data[0]) + 0.1
        y_pred = np.abs(reg_data[1]) + 0.1
        assert TweedieDeviance(p=1.5)(y_true, y_pred) >= 0.0

    def test_tweedie_deviance_invalid_p(self):
        with pytest.raises(ValueError):
            TweedieDeviance(p=0.5)
        with pytest.raises(ValueError):
            TweedieDeviance(p=2.5)

    def test_nb_deviance_returns_float(self, count_data):
        y_true, y_pred = count_data
        result = NegativeBinomialDeviance()(y_true, y_pred)
        assert isinstance(result, float)

    def test_nb_deviance_no_alpha_arg_required(self, count_data):
        """Must not require alpha as a positional argument (was the P0 bug)."""
        y_true, y_pred = count_data
        # Should not raise TypeError
        NegativeBinomialDeviance()(y_true, y_pred)

    def test_beta_brier_nonneg(self, proportion_data):
        y_true, y_pred = proportion_data
        assert BetaBrierScore()(y_true, y_pred) >= 0.0

    def test_beta_brier_zero_for_perfect(self, proportion_data):
        y_true, _ = proportion_data
        assert BetaBrierScore()(y_true, y_true) == pytest.approx(0.0, abs=1e-9)

    def test_dirichlet_error_nonneg(self):
        rng = np.random.default_rng(1)
        y_true = rng.dirichlet([2, 2, 2], size=50)
        y_pred = rng.dirichlet([2, 2, 2], size=50)
        assert DirichletError()(y_true, y_pred) >= 0.0

    def test_dirichlet_error_zero_for_perfect(self):
        y = np.array([[0.2, 0.5, 0.3], [0.1, 0.7, 0.2]])
        assert DirichletError()(y, y) == pytest.approx(0.0, abs=1e-9)

    def test_student_t_loss_returns_float(self, lss_data):
        y_true, y_pred = lss_data
        assert isinstance(StudentTLoss()(y_true, y_pred), float)

    def test_default_student_t_loss_uses_df_loc_scale_columns(self):
        from scipy.stats import t

        y_true = np.array([0.0, 2.0])
        y_pred = np.array([[8.0, 0.0, 1.0], [10.0, 1.0, 2.0]])
        metric = get_default_lss_metrics("studentt")[0]

        expected = -np.mean(t.logpdf(y_true, df=y_pred[:, 0], loc=y_pred[:, 1], scale=y_pred[:, 2]))
        assert metric(y_true, y_pred) == pytest.approx(expected)

    def test_default_zip_deviance_uses_expected_count(self):
        y_true = np.array([0.5, 2.0])
        y_pred = np.array([[0.5, 1.0], [0.2, 2.5]])
        metric = get_default_lss_metrics("zip")[0]

        expected_mean = (1.0 - y_pred[:, 0]) * y_pred[:, 1]
        expected = PoissonDeviance()(y_true, expected_mean)
        assert metric(y_true, y_pred) == pytest.approx(expected)

    def test_default_gamma_deviance_uses_shape_over_rate(self):
        y_true = np.array([2.0, 3.0])
        y_pred = np.array([[4.0, 2.0], [6.0, 2.0]])
        metric = get_default_lss_metrics("gamma")[0]

        assert metric(y_true, y_pred) == pytest.approx(0.0, abs=1e-12)

    def test_default_beta_brier_uses_alpha_over_concentration(self):
        y_true = np.array([0.25, 0.75])
        y_pred = np.array([[1.0, 3.0], [3.0, 1.0]])
        metric = get_default_lss_metrics("beta")[0]

        assert metric(y_true, y_pred) == pytest.approx(0.0, abs=1e-12)

    def test_default_johnson_su_crps_uses_location_and_scale_columns(self):
        y_true = np.array([1.0, 3.0])
        y_pred = np.array([[0.1, 4.0, 1.0, 0.5], [0.2, 5.0, 2.0, 1.5]])
        metric = get_default_lss_metrics("johnsonsu")[0]
        expected_params = y_pred[:, [2, 3]]

        assert metric(y_true, y_pred) == pytest.approx(CRPS(family="normal")(y_true, expected_params))

    def test_default_mog_crps_uses_weighted_mean_and_scale(self):
        y_true = np.array([3.0, 3.0])
        y_pred = np.array(
            [
                [0.25, 0.75, 0.0, 4.0, 1.0, 2.0],
                [0.6, 0.4, 2.0, 4.0, 0.5, 1.0],
            ]
        )
        metric = get_default_lss_metrics("mog")[0]
        weights = y_pred[:, :2]
        means = y_pred[:, 2:4]
        scales = y_pred[:, 4:]
        mean = np.sum(weights * means, axis=1)
        scale = np.sqrt(np.sum(weights * (scales**2 + means**2), axis=1) - mean**2)

        assert metric(y_true, y_pred) == pytest.approx(CRPS(family="normal")(y_true, np.column_stack([mean, scale])))

    def test_default_lognormal_metrics_only_score_outcome_scale_correctly(self):
        assert [metric.name for metric in get_default_lss_metrics("lognormal")] == ["lognormal_nll"]

    def test_default_quantile_metric_scores_median_column(self):
        y_true = np.array([1.0])
        y_pred = np.array([[100.0, 2.0, 0.0]])
        metric = get_default_lss_metrics("quantile")[0]

        assert metric(y_true, y_pred) == pytest.approx(0.5)

    def test_quantile_metric_uses_configured_quantile_metadata(self):
        y_true = np.array([1.0])
        y_pred = np.array([[0.0, 10.0, 2.0]])
        family = Quantile(quantiles=[0.75, 0.25, 0.5])
        metric = get_default_lss_metrics(family)[0]

        assert metric(y_true, y_pred) == pytest.approx(0.5)

    def test_public_lss_factory_accepts_quantiles_keyword(self):
        y_true = np.array([1.0])
        y_pred = np.array([[0.0, 10.0, 2.0]])
        metric = get_default_metrics("lss", "quantile", quantiles=[0.75, 0.25, 0.5])[0]

        assert metric(y_true, y_pred) == pytest.approx(0.5)
        assert metric.needs_raw is False

    def test_interval_score_returns_float(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.column_stack([y_true - 0.5, y_true + 0.5])
        assert isinstance(IntervalScore(alpha=0.05)(y_true, y_pred), float)

    def test_interval_score_increases_with_miscoverage(self):
        """Interval score is worse when predictions miss the true values."""
        y_true = np.array([5.0, 5.0, 5.0])
        good = np.column_stack([y_true - 1.0, y_true + 1.0])  # covers all
        bad = np.column_stack([y_true + 2.0, y_true + 3.0])  # misses all
        assert IntervalScore()(y_true, good) < IntervalScore()(y_true, bad)

    def test_interval_score_requires_2_columns(self):
        with pytest.raises(ValueError):
            IntervalScore()(np.ones(3), np.ones(3))

    def test_coverage_perfect(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.column_stack([y_true - 0.1, y_true + 0.1])
        assert CoverageProbability()(y_true, y_pred) == pytest.approx(1.0)

    def test_coverage_zero(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.column_stack([y_true + 1.0, y_true + 2.0])  # all miss
        assert CoverageProbability()(y_true, y_pred) == pytest.approx(0.0)

    def test_sharpness_nonneg(self):
        y_true = np.ones(5)
        y_pred = np.column_stack([np.zeros(5), np.ones(5) * 2.0])
        assert SharpnessScore()(y_true, y_pred) == pytest.approx(2.0)

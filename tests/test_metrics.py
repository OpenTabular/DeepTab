"""Tests for the deeptab.metrics public API.

Covers:
- Every metric class: correct return type, value, and attribute contract
- 2-D LSS parameter array handling (first column = predicted mean)
- Registry: correct metrics returned per (task, family) key
- DeepTabMetric ABC: name / higher_is_better / needs_raw attributes
- Edge cases: perfect predictions, constant targets, all-zeros
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

import deeptab.metrics as dm
from deeptab.metrics import (  # Classification; Distributional; Registry; Base; Regression
    AUPRC,
    AUROC,
    CRPS,
    METRIC_REGISTRY,
    Accuracy,
    BetaBrierScore,
    BrierScore,
    CoverageProbability,
    DeepTabMetric,
    DirichletError,
    ExpectedCalibrationError,
    F1Score,
    GammaDeviance,
    IntervalScore,
    LogLoss,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    NegativeBinomialDeviance,
    PinballLoss,
    PoissonDeviance,
    R2Score,
    RootMeanSquaredError,
    SharpnessScore,
    StudentTLoss,
    TweedieDeviance,
    get_default_metrics,
    get_default_metrics_dict,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

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
def clf_data_binary():
    """Binary classification labels and probability scores."""
    y_true = RNG.integers(0, 2, N)
    proba_pos = np.clip(y_true + RNG.normal(0.0, 0.2, N), 0.01, 0.99)
    proba = np.column_stack([1.0 - proba_pos, proba_pos])
    return y_true, proba


@pytest.fixture
def clf_data_multiclass():
    """3-class labels and probability matrix."""
    y_true = RNG.integers(0, 3, N)
    raw = RNG.dirichlet(alpha=[2.0, 2.0, 2.0], size=N)
    # Bias toward the true class
    for i, c in enumerate(y_true):
        raw[i, c] += 1.0
    proba = raw / raw.sum(axis=1, keepdims=True)
    return y_true, proba


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


# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------


class TestDeepTabMetricContract:
    """Every concrete metric must satisfy the ABC attribute contract."""

    ALL_METRICS: ClassVar[list] = [
        MeanSquaredError(),
        RootMeanSquaredError(),
        MeanAbsoluteError(),
        R2Score(),
        MeanAbsolutePercentageError(),
        PinballLoss(0.5),
        Accuracy(),
        F1Score(),
        AUROC(),
        AUPRC(),
        LogLoss(),
        BrierScore(),
        ExpectedCalibrationError(),
        CRPS(),
        BetaBrierScore(),
        CoverageProbability(),
        DirichletError(),
        GammaDeviance(),
        IntervalScore(),
        NegativeBinomialDeviance(),
        PoissonDeviance(),
        SharpnessScore(),
        StudentTLoss(),
        TweedieDeviance(),
    ]

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_is_deepTabMetric(self, metric):
        assert isinstance(metric, DeepTabMetric)

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_has_name(self, metric):
        assert isinstance(metric.name, str) and len(metric.name) > 0

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_higher_is_better_is_bool(self, metric):
        assert isinstance(metric.higher_is_better, bool)

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_needs_raw_is_bool(self, metric):
        assert isinstance(metric.needs_raw, bool)

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_repr_is_string(self, metric):
        assert isinstance(repr(metric), str)

    def test_r2_higher_is_better(self):
        assert R2Score().higher_is_better is True

    def test_accuracy_higher_is_better(self):
        assert Accuracy().higher_is_better is True

    def test_auroc_higher_is_better(self):
        assert AUROC().higher_is_better is True

    def test_mse_lower_is_better(self):
        assert MeanSquaredError().higher_is_better is False

    def test_crps_lower_is_better(self):
        assert CRPS().higher_is_better is False

    def test_nll_needs_raw(self):
        from deeptab.distributions.normal import NormalDistribution
        from deeptab.metrics import NegativeLogLikelihood

        nll = NegativeLogLikelihood(NormalDistribution())
        assert nll.needs_raw is True

    def test_standard_metrics_dont_need_raw(self):
        for m in [RootMeanSquaredError(), CRPS(), Accuracy(), PoissonDeviance()]:
            assert m.needs_raw is False


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


class TestClassificationMetrics:
    def test_accuracy_perfect(self):
        y = np.array([0, 1, 2, 0])
        proba = np.eye(3)[[0, 1, 2, 0]]
        assert Accuracy()(y, proba) == pytest.approx(1.0)

    def test_accuracy_all_wrong(self):
        y = np.array([0, 0, 0])
        proba = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]])
        assert Accuracy()(y, proba) == pytest.approx(0.0)

    def test_accuracy_binary_1d_proba(self):
        y = np.array([0, 1, 1, 0])
        proba = np.array([0.1, 0.9, 0.8, 0.2])
        assert Accuracy()(y, proba) == pytest.approx(1.0)

    def test_auroc_in_unit_interval(self, clf_data_binary):
        y_true, proba = clf_data_binary
        score = AUROC()(y_true, proba)
        assert 0.0 <= score <= 1.0

    def test_auroc_multiclass(self, clf_data_multiclass):
        y_true, proba = clf_data_multiclass
        score = AUROC()(y_true, proba)
        assert 0.0 <= score <= 1.0

    def test_auprc_in_unit_interval(self, clf_data_binary):
        y_true, proba = clf_data_binary
        assert 0.0 <= AUPRC()(y_true, proba) <= 1.0

    def test_logloss_nonnegative(self, clf_data_binary):
        y_true, proba = clf_data_binary
        assert LogLoss()(y_true, proba) >= 0.0

    def test_brier_in_unit_interval(self, clf_data_binary):
        y_true, proba = clf_data_binary
        assert 0.0 <= BrierScore()(y_true, proba) <= 1.0

    def test_ece_in_unit_interval(self, clf_data_binary):
        y_true, proba = clf_data_binary
        assert 0.0 <= ExpectedCalibrationError()(y_true, proba) <= 1.0

    def test_ece_zero_for_perfect_calibration(self):
        """A model that always predicts 100% confidence and is always right → ECE = 0."""
        y_true = np.array([0, 1, 0, 1])
        proba = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        assert ExpectedCalibrationError()(y_true, proba) == pytest.approx(0.0)

    def test_f1_perfect(self):
        y = np.array([0, 1, 0, 1])
        proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
        assert F1Score(average="binary")(y, proba) == pytest.approx(1.0)

    def test_f1_invalid_average(self):
        with pytest.raises(ValueError):
            F1Score(average="micro")


# ---------------------------------------------------------------------------
# Distributional metrics
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_regression_returns_list(self):
        metrics = get_default_metrics("regression")
        assert isinstance(metrics, list) and len(metrics) > 0

    def test_classification_returns_list(self):
        metrics = get_default_metrics("classification")
        assert isinstance(metrics, list) and len(metrics) > 0

    @pytest.mark.parametrize(
        "family",
        [
            "normal",
            "lognormal",
            "studentt",
            "gamma",
            "inversegamma",
            "tweedie",
            "beta",
            "poisson",
            "zip",
            "negativebinom",
            "categorical",
            "dirichlet",
            "johnsonsu",
            "mog",
            "quantile",
        ],
    )
    def test_all_lss_families_have_metrics(self, family):
        metrics = get_default_metrics("lss", family=family)
        assert len(metrics) > 0, f"No default metrics for lss:{family}"

    def test_all_registry_entries_are_deepTabMetric(self):
        for key, metric_list in METRIC_REGISTRY.items():
            for m in metric_list:
                assert isinstance(m, DeepTabMetric), f"METRIC_REGISTRY[{key!r}] contains non-DeepTabMetric: {m!r}"

    def test_get_default_metrics_dict_keys_are_names(self):
        d = get_default_metrics_dict("regression")
        for key, metric in d.items():
            assert key == metric.name

    def test_unknown_task_returns_empty(self):
        assert get_default_metrics("unknown_task") == []

    def test_unknown_family_falls_back_to_task(self):
        # "lss" without a matching family key falls back to empty list
        result = get_default_metrics("lss", family="nonexistent")
        assert isinstance(result, list)

    def test_regression_primary_metric_is_rmse(self):
        metrics = get_default_metrics("regression")
        assert metrics[0].name == "rmse"

    def test_lss_normal_primary_metric_is_crps(self):
        metrics = get_default_metrics("lss", "normal")
        assert metrics[0].name == "crps"

    def test_classification_primary_metric_is_accuracy(self):
        metrics = get_default_metrics("classification")
        assert metrics[0].name == "accuracy"


# ---------------------------------------------------------------------------
# Public __all__ completeness
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_all_exports_importable(self):
        for name in dm.__all__:
            assert hasattr(dm, name), f"'{name}' listed in __all__ but not importable"

    def test_no_abstract_classes_in_all(self):
        import inspect

        for name in dm.__all__:
            obj = getattr(dm, name)
            if inspect.isclass(obj):
                assert not inspect.isabstract(obj) or obj is DeepTabMetric, (
                    f"{name} is abstract and should not be directly instantiable"
                )

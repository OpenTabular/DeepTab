"""Tests for metric contracts."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from deeptab.metrics import (  # Classification; Distributional; Registry; Base; Regression
    AUPRC,
    AUROC,
    CRPS,
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
)

RNG = np.random.default_rng(42)

N = 100


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

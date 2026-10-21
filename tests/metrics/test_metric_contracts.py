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


class TestLikelihoodMetricsAcceptColumnTargets:
    """NegativeLogLikelihood and LogScore must score a column-vector target
    like the matching 1-D target instead of broadcasting it to an (N, N) grid."""

    @pytest.mark.parametrize("metric_name", ["NegativeLogLikelihood", "LogScore"])
    def test_column_and_flat_targets_give_the_family_loss(self, metric_name):
        import torch

        import deeptab.metrics as metrics
        from deeptab.distributions.normal import NormalDistribution

        family = NormalDistribution()
        rng = np.random.default_rng(0)
        raw = rng.normal(size=(20, 2)).astype(np.float32)
        targets = rng.normal(size=20).astype(np.float32)
        metric = getattr(metrics, metric_name)(family)
        expected = float(family.compute_loss(torch.tensor(raw), torch.tensor(targets)))
        sign = -1.0 if metric_name == "LogScore" else 1.0
        assert metric(targets, raw) == pytest.approx(sign * expected)
        assert metric(targets[:, None], raw) == pytest.approx(sign * expected)

    def test_logged_validation_nll_matches_validation_loss(self):
        import pandas as pd

        from deeptab.configs import MLPConfig, TrainerConfig
        from deeptab.distributions.normal import NormalDistribution
        from deeptab.metrics import NegativeLogLikelihood
        from deeptab.models import MLPLSS

        rng = np.random.default_rng(0)
        features = pd.DataFrame({name: rng.normal(size=160) for name in ("a", "b", "c", "d")})
        targets = (2 * features["a"] + rng.normal(scale=0.5, size=160)).to_numpy()
        model = MLPLSS(model_config=MLPConfig(layer_sizes=[16]), trainer_config=TrainerConfig(max_epochs=2, lr=1e-2))
        model.fit(
            features[:120],
            targets[:120],
            family="normal",
            X_val=features[120:],
            y_val=targets[120:],
            val_metrics={"nll": NegativeLogLikelihood(NormalDistribution())},
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        assert model._trainer is not None
        logged = model._trainer.callback_metrics
        assert float(logged["val_nll"]) == pytest.approx(float(logged["val_loss"]), rel=1e-4)

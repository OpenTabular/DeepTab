"""Tests for metric registry."""

from __future__ import annotations

import numpy as np
import pytest

import deeptab.metrics as dm
from deeptab.metrics import (  # Classification; Distributional; Registry; Base; Regression
    METRIC_REGISTRY,
    DeepTabMetric,
    get_default_metrics,
    get_default_metrics_dict,
)

RNG = np.random.default_rng(42)

N = 100


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
            "multinomial",
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

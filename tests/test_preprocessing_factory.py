"""Tests for deeptab.core.preprocessing.build_preprocessor.

Covers:
- output_structure/output_format are always forced to "blocks"/"dense", regardless
  of what a PreprocessingConfig would otherwise resolve to.
- The estimator-resolved task always takes precedence over PreprocessingConfig.task;
  a conflicting config value warns rather than being used.
- None preprocessing_config falls back to PreTab's own defaults (forced kwargs
  still applied).
- ObservabilityConfig.verbosity maps onto Preprocessor(verbose=...), and
  structured_logging=True attaches DeepTab's console handler to PreTab's shared
  "pretab" logger exactly once, regardless of how many times it is rebuilt.
- PreprocessingConfig.preset is forwarded to Preprocessor(preset=...); when a
  preset is set and output_dim is left unset, DeepTab defers width resolution to
  the preset instead of forcing its own default of 7. list_available_representations
  is a thin, read-only pass-through to PreTab's representation registry.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from pretab import Preprocessor

from deeptab.configs import PreprocessingConfig
from deeptab.core.exceptions import ConfigWarning
from deeptab.core.observability import ObservabilityConfig
from deeptab.core.preprocessing import (
    _PRETAB_LOGGER_NAME,
    _PretabConsoleHandler,
    build_preprocessor,
    list_available_representations,
)


class TestBuildPreprocessorForcedOutput:
    def test_defaults_force_blocks_and_dense(self):
        pre = build_preprocessor()
        assert pre.output_structure == "blocks"
        assert pre.output_format == "dense"

    def test_forced_kwargs_override_any_config_value(self):
        # PreprocessingConfig has no output_structure/output_format fields at all,
        # so there is nothing a config could set here; forcing must still happen.
        with pytest.warns(FutureWarning):
            cfg = PreprocessingConfig(numerical_method="bspline", target_aware=False, placement_strategy="uniform")
        pre = build_preprocessor(cfg)
        assert pre.output_structure == "blocks"
        assert pre.output_format == "dense"

    def test_blocks_output_is_dict_keyed_by_prefixed_column_name(self):
        pre = build_preprocessor()
        X = pd.DataFrame({"amount": np.linspace(1, 10, 20), "tier": ["a", "b"] * 10})
        y = np.linspace(0, 1, 20)
        out = pre.fit_transform(X, y)
        assert isinstance(out, dict)
        assert "num_amount" in out
        assert "cat_tier" in out


class TestBuildPreprocessorTaskAndSeed:
    def test_task_argument_is_forwarded_when_config_omits_it(self):
        with pytest.warns(FutureWarning):
            cfg = PreprocessingConfig()
        pre = build_preprocessor(cfg, task="classification")
        assert pre.task == "classification"

    def test_estimator_task_takes_precedence_over_config_task(self):
        with pytest.warns(FutureWarning):
            cfg = PreprocessingConfig(task="regression")
        with pytest.warns(ConfigWarning, match="conflicts with the task resolved"):
            pre = build_preprocessor(cfg, task="classification")
        assert pre.task == "classification"

    def test_matching_config_task_does_not_warn(self):
        import warnings

        with pytest.warns(FutureWarning):
            cfg = PreprocessingConfig(task="classification")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConfigWarning)
            pre = build_preprocessor(cfg, task="classification")
        assert pre.task == "classification"

    def test_random_state_argument_is_forwarded(self):
        with pytest.warns(FutureWarning):
            cfg = PreprocessingConfig()
        pre = build_preprocessor(cfg, random_state=42)
        assert pre.random_state == 42

    def test_none_preprocessing_config_still_builds_a_preprocessor(self):
        pre = build_preprocessor(None, task="regression", random_state=0)
        assert isinstance(pre, Preprocessor)
        assert pre.task == "regression"
        assert pre.random_state == 0


class TestEstimatorResolvedTaskAndSeed:
    """The task/random_state an estimator's own preprocessor resolves to."""

    def test_classifier_resolves_classification_task_and_seed(self):
        from deeptab.models.mlp import MLPClassifier

        clf = MLPClassifier(random_state=42)
        assert clf._preprocessor.task == "classification"
        assert clf._preprocessor.random_state == 42

    def test_regressor_resolves_regression_task_and_seed(self):
        from deeptab.models.mlp import MLPRegressor

        reg = MLPRegressor(random_state=7)
        assert reg._preprocessor.task == "regression"
        assert reg._preprocessor.random_state == 7

    def test_lss_defaults_to_regression_before_fit(self):
        from deeptab.models.mlp import MLPLSS

        lss = MLPLSS(random_state=5)
        assert lss._preprocessor.task == "regression"

    def test_lss_resolves_classification_for_categorical_family_after_fit(self):
        import contextlib
        import warnings

        from deeptab.models.mlp import MLPLSS

        X = pd.DataFrame({"amount": np.linspace(1, 10, 30), "tier": ["a", "b"] * 15})
        y = np.tile([0, 1, 2], 10)
        lss = MLPLSS(random_state=5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Suppresses an unrelated, pre-existing categorical-family training bug;
            # the preprocessor's task is already resolved before that point.
            with contextlib.suppress(Exception):
                lss.fit(X, y, family="categorical", max_epochs=1)
        assert lss._preprocessor.task == "classification"

    def test_estimator_task_overrides_conflicting_config_task_with_warning(self):
        from deeptab.models.mlp import MLPClassifier

        with pytest.warns(FutureWarning):
            cfg = PreprocessingConfig(task="regression")
        with pytest.warns(ConfigWarning, match="conflicts with the task resolved"):
            clf = MLPClassifier(preprocessing_config=cfg)
        assert clf._preprocessor.task == "classification"


class TestObservabilityVerbosityMapping:
    """ObservabilityConfig.verbosity maps onto Preprocessor(verbose=...), and
    structured_logging=True attaches DeepTab's console handler to PreTab's
    shared "pretab" logger.
    """

    @pytest.fixture(autouse=True)
    def _reset_pretab_logger(self):
        pretab_logger = logging.getLogger(_PRETAB_LOGGER_NAME)
        original_handlers = pretab_logger.handlers[:]
        original_level = pretab_logger.level
        original_propagate = pretab_logger.propagate
        yield
        pretab_logger.handlers[:] = original_handlers
        pretab_logger.level = original_level
        pretab_logger.propagate = original_propagate

    def test_no_observability_config_keeps_pretabs_own_default(self):
        pre = build_preprocessor()
        assert pre.verbose == 0

    def test_verbosity_is_forwarded_as_verbose(self):
        obs = ObservabilityConfig(verbosity=2)
        pre = build_preprocessor(observability_config=obs)
        assert pre.verbose == 2

    def test_verbosity_above_three_is_clamped(self):
        obs = ObservabilityConfig(verbosity=7)
        pre = build_preprocessor(observability_config=obs)
        assert pre.verbose == 3

    def test_negative_verbosity_is_clamped_to_zero(self):
        obs = ObservabilityConfig(verbosity=-1)
        pre = build_preprocessor(observability_config=obs)
        assert pre.verbose == 0

    def test_structured_logging_true_attaches_console_handler(self):
        obs = ObservabilityConfig(structured_logging=True, log_to_console=True, verbosity=1)
        build_preprocessor(observability_config=obs)
        handlers = logging.getLogger(_PRETAB_LOGGER_NAME).handlers
        assert any(isinstance(handler, _PretabConsoleHandler) for handler in handlers)

    def test_handler_attachment_is_idempotent(self):
        obs = ObservabilityConfig(structured_logging=True, log_to_console=True, verbosity=1)
        for _ in range(3):
            build_preprocessor(observability_config=obs)
        handlers = logging.getLogger(_PRETAB_LOGGER_NAME).handlers
        assert sum(isinstance(handler, _PretabConsoleHandler) for handler in handlers) == 1

    def test_structured_logging_false_does_not_attach_handler(self):
        obs = ObservabilityConfig(structured_logging=False, verbosity=3)
        build_preprocessor(observability_config=obs)
        handlers = logging.getLogger(_PRETAB_LOGGER_NAME).handlers
        assert not any(isinstance(handler, _PretabConsoleHandler) for handler in handlers)

    def test_log_to_console_false_does_not_attach_handler(self):
        obs = ObservabilityConfig(structured_logging=True, log_to_console=False, verbosity=3)
        build_preprocessor(observability_config=obs)
        handlers = logging.getLogger(_PRETAB_LOGGER_NAME).handlers
        assert not any(isinstance(handler, _PretabConsoleHandler) for handler in handlers)

    def test_estimator_forwards_its_observability_config_verbosity(self):
        from deeptab.models.mlp import MLPClassifier

        obs = ObservabilityConfig(verbosity=2)
        clf = MLPClassifier(observability_config=obs)
        assert clf._preprocessor.verbose == 2


class TestPresets:
    """PreprocessingConfig.preset forwards to Preprocessor(preset=...)."""

    def _fit_data(self):
        X = pd.DataFrame({"a": np.linspace(0, 1, 200), "cat": (["p", "q", "r"] * 67)[:200]})
        y = np.linspace(0, 1, 200)
        return X, y

    def test_expanded_preset_widens_and_one_hot_encodes(self):
        cfg = PreprocessingConfig(preset="expanded")
        pre = build_preprocessor(cfg, task="regression")
        X, y = self._fit_data()
        pre.fit(X, y)
        assert pre.output_dims_["a"] == 10
        assert pre.get_resolved_config()["categorical_method"] == "one-hot"

    def test_standard_preset_resolves_numerical_method_from_task(self):
        cfg_reg = PreprocessingConfig(preset="standard")
        pre_reg = build_preprocessor(cfg_reg, task="regression")
        X, y = self._fit_data()
        pre_reg.fit(X, y)
        assert pre_reg.get_resolved_config()["numerical_method"] == "bspline"

        cfg_clf = PreprocessingConfig(preset="standard")
        pre_clf = build_preprocessor(cfg_clf, task="classification")
        pre_clf.fit(X, (y > 0.5).astype(int))
        assert pre_clf.get_resolved_config()["numerical_method"] == "ple"

    def test_adaptive_preset_sizes_width_per_feature(self):
        cfg = PreprocessingConfig(preset="adaptive")
        pre = build_preprocessor(cfg, task="regression")
        X, y = self._fit_data()
        pre.fit(X, y)
        assert 7 <= pre.output_dims_["a"] <= 15

    def test_explicit_output_dim_overrides_preset_width(self):
        cfg = PreprocessingConfig(preset="expanded", output_dim=6)
        pre = build_preprocessor(cfg, task="regression")
        X, y = self._fit_data()
        pre.fit(X, y)
        assert pre.output_dims_["a"] != 10

    def test_no_preset_behavior_is_unchanged(self):
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = PreprocessingConfig()
        assert any(issubclass(w.category, FutureWarning) for w in caught)
        pre = build_preprocessor(cfg, task="regression")
        assert pre.output_dim == 7


class TestListAvailableRepresentations:
    def test_returns_nonempty_sorted_list(self):
        names = list_available_representations()
        assert names == sorted(names)
        assert len(names) > 0
        assert "ple" in names

    def test_filters_by_feature_kind(self):
        numerical = list_available_representations(feature_kind="numerical")
        categorical = list_available_representations(feature_kind="categorical")
        assert "ple" in numerical
        assert "ple" not in categorical

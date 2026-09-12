"""Tests for deeptab.core.preprocessing.build_preprocessor.

Covers:
- output_structure/output_format are always forced to "blocks"/"dense", regardless
  of what a PreprocessingConfig would otherwise resolve to.
- The estimator-resolved task always takes precedence over PreprocessingConfig.task;
  a conflicting config value warns rather than being used.
- None preprocessing_config falls back to PreTab's own defaults (forced kwargs
  still applied).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pretab import Preprocessor

from deeptab.configs import PreprocessingConfig
from deeptab.core.exceptions import ConfigWarning
from deeptab.core.preprocessing import build_preprocessor


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

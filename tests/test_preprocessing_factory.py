"""Tests for deeptab.core.preprocessing.build_preprocessor.

Covers:
- output_structure/output_format are always forced to "blocks"/"dense", regardless
  of what a PreprocessingConfig would otherwise resolve to.
- task/random_state are only applied when the config did not already resolve them.
- None preprocessing_config falls back to PreTab's own defaults (forced kwargs
  still applied).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pretab import Preprocessor

from deeptab.configs import PreprocessingConfig
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

    def test_explicit_config_task_takes_precedence_over_argument(self):
        with pytest.warns(FutureWarning):
            cfg = PreprocessingConfig(task="regression")
        pre = build_preprocessor(cfg, task="classification")
        assert pre.task == "regression"

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

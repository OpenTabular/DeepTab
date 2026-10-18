"""Tests for observability configuration."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.base import clone

from deeptab.configs import MLPConfig
from deeptab.core.observability import ObservabilityConfig
from deeptab.models import MLPLSS, MLPClassifier, MLPRegressor


@pytest.mark.parametrize("model_cls", [MLPClassifier, MLPRegressor, MLPLSS])
@pytest.mark.parametrize("split_config", [False, True])
def test_observability_config_survives_get_params_set_params_and_clone(model_cls, split_config, tmp_path):
    config_kwargs = {"model_config": MLPConfig(layer_sizes=[16])} if split_config else {}
    obs = ObservabilityConfig(root_dir=str(tmp_path), experiment_name="cloning")
    model = model_cls(observability_config=obs, **config_kwargs)
    if not split_config:
        model.set_params(layer_sizes=[16])
    params = model.get_params()
    assert "_observability_config" not in params
    assert params["observability_config"] is obs

    replacement = ObservabilityConfig(root_dir=str(tmp_path), experiment_name="replacement")
    assert model.set_params(observability_config=replacement) is model
    assert model._observability_config is replacement
    assert model.get_params(deep=False)["observability_config"] is replacement

    cloned = clone(model)
    assert isinstance(cloned, (MLPClassifier, MLPRegressor, MLPLSS))
    assert cloned._observability_config is not None
    assert cloned._observability_config == replacement
    assert cloned._observability_config is not replacement
    assert cloned.config.layer_sizes == [16]
    assert cloned._run_dir is None
    cloned._observability_config.experiment_name = "independent"
    assert model._observability_config.experiment_name == "replacement"


def test_clone_preserves_post_construction_observability():
    model = MLPClassifier()
    obs = ObservabilityConfig()
    model.configure_observability(obs)
    assert model.get_params()["observability_config"] is obs
    cloned = clone(model)
    assert isinstance(cloned, MLPClassifier)
    assert cloned._observability_config == obs


@pytest.mark.parametrize("split_config", [False, True])
def test_fitted_clone_creates_independent_run_artifacts(split_config, tmp_path):
    config_kwargs = {"model_config": MLPConfig(layer_sizes=[16])} if split_config else {}
    model = MLPClassifier(
        observability_config=ObservabilityConfig(root_dir=str(tmp_path), experiment_name="cloning"),
        random_state=42,
        **config_kwargs,
    )
    if not split_config:
        model.set_params(layer_sizes=[16])
    features = np.random.default_rng(42).standard_normal((60, 4))
    targets = np.tile([0, 1], 30)
    fit_kwargs = {
        "max_epochs": 1,
        "batch_size": 16,
        "accelerator": "cpu",
        "logger": False,
        "enable_progress_bar": False,
    }
    model.fit(features, targets, **fit_kwargs)
    cloned = clone(model)
    assert isinstance(cloned, MLPClassifier)
    assert getattr(cloned, "_run_dir", None) is None
    cloned.fit(features, targets, **fit_kwargs)

    model_run_dir = model._run_dir
    cloned_run_dir = cloned._run_dir
    assert isinstance(model_run_dir, str)
    assert isinstance(cloned_run_dir, str)
    assert cloned_run_dir != model_run_dir
    assert Path(model_run_dir, "summary.json").is_file()
    assert Path(cloned_run_dir, "summary.json").is_file()
    assert Path(cloned_run_dir).parent == tmp_path / "runs" / "cloning"


def test_configure_observability_post_construction(monkeypatch):
    """configure_observability() can be called after construction."""
    fake_structlog = MagicMock()
    fake_structlog.wrap_logger.return_value = MagicMock()
    monkeypatch.setitem(sys.modules, "structlog", fake_structlog)

    from deeptab.models import MLPClassifier

    clf = MLPClassifier()
    assert clf._event_logger is None
    clf.configure_observability(ObservabilityConfig(structured_logging=True))
    assert clf._event_logger is not None

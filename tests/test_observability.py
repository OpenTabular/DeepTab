"""Tests for the observability layer (Phase 8).

Covers:
- Default instantiation imports no optional packages.
- ``use_structlog=True`` raises ``ImportError`` when structlog is absent.
- ``experiment_trackers=["mlflow"]`` raises ``ImportError`` when mlflow absent.
- ``experiment_trackers=["tensorboard"]`` raises ``ImportError`` when tensorboard absent.
- Unknown tracker name raises ``ValueError``.
- User-provided logger is appended, not replaced.
- ``configure_observability()`` works post-construction.
- ``_observability_config`` is absent from ``get_params()`` output.
- ``_emit_event`` is a no-op when no logger is configured.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from deeptab.core.observability import ObservabilityConfig, build_lightning_loggers, build_structlog_logger
from deeptab.models._mixins.observability import _ObservabilityMixin

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class _FakeLogger:
    """Minimal fake that records calls to info()."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **kwargs: Any) -> None:
        self.calls.append((event, kwargs))


# ---------------------------------------------------------------------------
# ObservabilityConfig
# ---------------------------------------------------------------------------


def test_observability_config_defaults():
    cfg = ObservabilityConfig()
    assert cfg.root_dir == "deeptab_runs"
    assert cfg.experiment_name == "default"
    assert cfg.verbosity == 1
    assert cfg.structured_logging is False
    assert cfg.log_to_console is True
    assert cfg.log_to_file is False
    assert cfg.experiment_trackers == []
    assert cfg.tensorboard_save_dir == "deeptab_runs/tensorboard"
    assert cfg.tensorboard_name == "deeptab"
    assert cfg.mlflow_experiment_name == "deeptab"
    assert cfg.mlflow_tracking_uri == "sqlite:///deeptab_runs/mlflow/backend/mlflow.db"
    assert cfg.mlflow_artifact_location == "deeptab_runs/mlflow/artifacts"
    assert cfg.mlflow_run_name is None
    assert cfg.mlflow_log_model is True
    assert cfg.logger is None


def test_observability_config_is_dataclass():
    from dataclasses import fields

    names = {f.name for f in fields(ObservabilityConfig)}
    assert names == {
        "root_dir",
        "experiment_name",
        "verbosity",
        "structured_logging",
        "log_to_console",
        "log_to_file",
        "experiment_trackers",
        "tensorboard_save_dir",
        "tensorboard_name",
        "mlflow_experiment_name",
        "mlflow_tracking_uri",
        "mlflow_artifact_location",
        "mlflow_run_name",
        "mlflow_log_model",
        "logger",
    }


# ---------------------------------------------------------------------------
# build_structlog_logger — absent package path
# ---------------------------------------------------------------------------


def test_root_dir_derives_all_paths():
    """Custom root_dir propagates to all three sub-paths."""
    cfg = ObservabilityConfig(root_dir="runs/proj")
    assert cfg.tensorboard_save_dir == "runs/proj/tensorboard"
    assert cfg.mlflow_tracking_uri == "sqlite:///runs/proj/mlflow/backend/mlflow.db"
    assert cfg.mlflow_artifact_location == "runs/proj/mlflow/artifacts"


def test_root_dir_explicit_override_not_clobbered():
    """Explicit sub-path overrides are not replaced by root_dir resolution."""
    cfg = ObservabilityConfig(
        root_dir="runs/proj",
        tensorboard_save_dir="/tb_root",
        mlflow_tracking_uri="http://localhost:5000",
        mlflow_artifact_location="/artifacts/custom",
    )
    assert cfg.tensorboard_save_dir == "/tb_root"
    assert cfg.mlflow_tracking_uri == "http://localhost:5000"
    assert cfg.mlflow_artifact_location == "/artifacts/custom"


def test_build_structlog_logger_raises_when_absent(monkeypatch):
    """ImportError with install hint when structlog is not installed."""
    monkeypatch.setitem(sys.modules, "structlog", None)  # type: ignore[arg-type]
    with pytest.raises(ImportError, match="pip install 'deeptab\\[logs\\]'"):
        build_structlog_logger(ObservabilityConfig(structured_logging=True))


def test_build_structlog_logger_returns_info_compatible_object(monkeypatch, capsys):
    """When structlog is available, return an object with .info() that emits output."""
    fake_structlog = MagicMock()
    monkeypatch.setitem(sys.modules, "structlog", fake_structlog)
    logger = build_structlog_logger(
        ObservabilityConfig(structured_logging=True, log_to_console=True, log_to_file=False, verbosity=3)
    )
    logger.info("test_event", key="value")
    captured = capsys.readouterr()
    assert "test_event" in captured.out
    assert "key=value" in captured.out


# ---------------------------------------------------------------------------
# build_lightning_loggers
# ---------------------------------------------------------------------------


def test_build_lightning_loggers_empty_config():
    cfg = ObservabilityConfig()
    result = build_lightning_loggers(cfg)
    assert result == []


def test_build_lightning_loggers_user_logger_appended():
    user_logger = _FakeLogger()
    cfg = ObservabilityConfig(logger=user_logger)
    result = build_lightning_loggers(cfg)
    assert result == [user_logger]


def test_build_lightning_loggers_unknown_tracker_raises():
    cfg = ObservabilityConfig(experiment_trackers=["wandb"])
    with pytest.raises(ValueError, match=r"Unknown experiment tracker.*'wandb'"):
        build_lightning_loggers(cfg)


def test_build_lightning_loggers_mlflow_absent(monkeypatch):
    """ImportError with install hint when mlflow is not installed."""
    # Simulate mlflow being absent by blocking its import inside Lightning
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__  # type: ignore[attr-defined]

    def _block_mlflow(name, *args, **kwargs):
        if "MLFlowLogger" in name or (len(args) >= 3 and "MLFlowLogger" in str(args[2])):
            raise ImportError("No module named 'mlflow'")
        return real_import(name, *args, **kwargs)

    # Use monkeypatch on the lightning loggers module directly
    mock_module = MagicMock()
    mock_module.MLFlowLogger.side_effect = ImportError("No module named 'mlflow'")

    import lightning.pytorch.loggers as lpl

    original_MLFlowLogger = getattr(lpl, "MLFlowLogger", None)

    # Patch lightning.pytorch.loggers so that importing MLFlowLogger raises
    monkeypatch.setitem(sys.modules, "lightning.pytorch.loggers", None)  # type: ignore[arg-type]
    cfg = ObservabilityConfig(experiment_trackers=["mlflow"])
    with pytest.raises(ImportError, match="pip install 'deeptab\\[mlflow\\]'"):
        build_lightning_loggers(cfg)


def test_build_lightning_loggers_tensorboard_absent(monkeypatch):
    """ImportError with install hint when tensorboard is not installed."""
    monkeypatch.setitem(sys.modules, "lightning.pytorch.loggers", None)  # type: ignore[arg-type]
    cfg = ObservabilityConfig(experiment_trackers=["tensorboard"])
    with pytest.raises(ImportError, match="pip install 'deeptab\\[tensorboard\\]'"):
        build_lightning_loggers(cfg)


def test_build_lightning_loggers_user_logger_does_not_replace(monkeypatch):
    """User-provided logger is appended alongside built-in trackers."""
    user_logger = _FakeLogger()
    # Mock TensorBoardLogger
    fake_tb = MagicMock()
    fake_lpl = MagicMock()
    fake_lpl.TensorBoardLogger.return_value = fake_tb
    monkeypatch.setitem(sys.modules, "lightning.pytorch.loggers", fake_lpl)
    cfg = ObservabilityConfig(experiment_trackers=["tensorboard"], logger=user_logger)
    result = build_lightning_loggers(cfg)
    assert len(result) == 2
    assert result[-1] is user_logger


# ---------------------------------------------------------------------------
# _ObservabilityMixin
# ---------------------------------------------------------------------------


def test_emit_event_noop_by_default():
    """_emit_event does nothing when no logger is attached."""

    class _Estimator(_ObservabilityMixin):
        pass

    est = _Estimator()
    # Should not raise
    est._emit_event("fit_started", n_samples=100)


def test_emit_event_dispatches_to_logger():
    logger = _FakeLogger()

    class _Estimator(_ObservabilityMixin):
        pass

    est = _Estimator()
    est._event_logger = logger
    est._emit_event("fit_started", n_samples=100)
    assert logger.calls == [("fit_started", {"n_samples": 100})]


def test_configure_observability_wires_structlog(monkeypatch, capsys):
    fake_structlog = MagicMock()
    monkeypatch.setitem(sys.modules, "structlog", fake_structlog)

    class _Estimator(_ObservabilityMixin):
        pass

    est = _Estimator()
    assert est._event_logger is None
    est.configure_observability(ObservabilityConfig(structured_logging=True, log_to_console=True, log_to_file=False))
    assert est._event_logger is not None
    est._emit_event("fit.started")
    captured = capsys.readouterr()
    assert "fit.started" in captured.out


def test_configure_observability_no_structlog_no_logger():
    """No-op when structured_logging=False and no tracker — _event_logger stays None."""

    class _Estimator(_ObservabilityMixin):
        pass

    est = _Estimator()
    est.configure_observability(ObservabilityConfig())
    assert est._event_logger is None


# ---------------------------------------------------------------------------
# SklearnBase integration
# ---------------------------------------------------------------------------


def test_observability_config_not_in_get_params():
    """_observability_config is hidden from sklearn get_params/clone."""
    from deeptab.configs import MLPConfig
    from deeptab.models import MLPClassifier

    clf = MLPClassifier()
    clf._observability_config = ObservabilityConfig()
    params = clf.get_params()
    assert "_observability_config" not in params
    assert "observability_config" not in params


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

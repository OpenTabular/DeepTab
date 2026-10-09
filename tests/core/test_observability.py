"""Tests for observability."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from deeptab.core.observability import ObservabilityConfig, build_lightning_loggers, build_structlog_logger


class _FakeLogger:
    """Minimal fake that records calls to info()."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **kwargs: Any) -> None:
        self.calls.append((event, kwargs))


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
    # The guard checks the actual ``mlflow`` package, so simulate its absence.
    monkeypatch.setitem(sys.modules, "mlflow", None)  # type: ignore[arg-type]
    cfg = ObservabilityConfig(experiment_trackers=["mlflow"])
    with pytest.raises(ImportError, match="pip install 'deeptab\\[mlflow\\]'"):
        build_lightning_loggers(cfg)


@pytest.mark.parametrize("uri_kind", ["default", "relative", "absolute", "remote", "memory"])
def test_mlflow_sqlite_parent_is_ready_before_logger_construction(monkeypatch, tmp_path, uri_kind):
    import sqlite3

    monkeypatch.chdir(tmp_path)
    root = tmp_path / "runs"
    database_path = root / "mlflow" / "backend" / "mlflow.db"
    tracking_uri = ""
    if uri_kind == "relative":
        database_path = tmp_path / "custom" / "backend" / "tracking.db"
        tracking_uri = "sqlite:///custom/backend/tracking.db"
    elif uri_kind == "absolute":
        database_path = tmp_path / "absolute" / "backend" / "tracking.db"
        tracking_uri = f"sqlite:///{database_path}"
    elif uri_kind == "remote":
        tracking_uri = "https://tracking.example.com"
    elif uri_kind == "memory":
        tracking_uri = "sqlite:///:memory:"

    def create_logger(**kwargs):
        if uri_kind in {"remote", "memory"}:
            assert not database_path.parent.exists()
        else:
            with sqlite3.connect(database_path) as connection:
                connection.execute("CREATE TABLE runs (id INTEGER)")
        return MagicMock()

    monkeypatch.setattr("deeptab.core.optional_deps.require_mlflow", lambda: None)
    fake_loggers = MagicMock()
    fake_loggers.MLFlowLogger.side_effect = create_logger
    monkeypatch.setitem(sys.modules, "lightning.pytorch.loggers", fake_loggers)
    config = ObservabilityConfig(
        experiment_trackers=["mlflow"],
        root_dir=str(root),
        mlflow_tracking_uri=tracking_uri,
    )
    assert not database_path.parent.exists()
    assert len(build_lightning_loggers(config)) == 1
    assert (root / "mlflow" / "artifacts").is_dir()
    fake_loggers.MLFlowLogger.assert_called_once()
    assert fake_loggers.MLFlowLogger.call_args.kwargs["tracking_uri"] == config.mlflow_tracking_uri


def test_build_lightning_loggers_tensorboard_absent(monkeypatch):
    """ImportError with install hint when tensorboard is not installed."""
    # The guard checks ``torch.utils.tensorboard``, so simulate its absence.
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)  # type: ignore[arg-type]
    cfg = ObservabilityConfig(experiment_trackers=["tensorboard"])
    with pytest.raises(ImportError, match="pip install 'deeptab\\[tensorboard\\]'"):
        build_lightning_loggers(cfg)


def test_build_lightning_loggers_user_logger_does_not_replace(monkeypatch):
    """User-provided logger is appended alongside built-in trackers."""
    user_logger = _FakeLogger()
    # Stub the optional tensorboard import guard so the test does not depend on
    # the 'tensorboard' package being installed in the environment.
    fake_tb_mod = ModuleType("torch.utils.tensorboard")
    fake_tb_mod.SummaryWriter = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", fake_tb_mod)
    # Mock TensorBoardLogger
    fake_tb = MagicMock()
    fake_lpl = MagicMock()
    fake_lpl.TensorBoardLogger.return_value = fake_tb
    monkeypatch.setitem(sys.modules, "lightning.pytorch.loggers", fake_lpl)
    cfg = ObservabilityConfig(experiment_trackers=["tensorboard"], logger=user_logger)
    result = build_lightning_loggers(cfg)
    assert len(result) == 2
    assert result[-1] is user_logger

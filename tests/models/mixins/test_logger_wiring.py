"""Tests for logger wiring."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

from deeptab.core.observability import ObservabilityConfig
from deeptab.models._mixins.observability import _ObservabilityMixin


class _FakeLogger:
    """Minimal fake that records calls to info()."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **kwargs: Any) -> None:
        self.calls.append((event, kwargs))


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

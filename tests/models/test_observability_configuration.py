"""Tests for observability configuration."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

from deeptab.core.observability import ObservabilityConfig


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

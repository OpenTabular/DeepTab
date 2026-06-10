"""Unit tests for each mixin in isolation.

Each mixin is tested through a minimal fake subclass so that no real
PyTorch, Lightning, or sklearn machinery is required.  These tests are
fast (<1 s total) and give precise failure messages when a mixin changes
its contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeptab.models._mixins.observability import _NoOpEventLogger, _ObservabilityMixin, _SupportsInfo

# ---------------------------------------------------------------------------
# Helpers — minimal fake classes
# ---------------------------------------------------------------------------


class _FakeEstimator(_ObservabilityMixin):
    """Minimal class that just inherits the observability mixin."""

    pass


class _RecordingLogger:
    """Capture every call to info() for assertion."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def info(self, event: str, **kwargs) -> None:
        self.calls.append((event, kwargs))

    def events(self) -> list[str]:
        return [e for e, _ in self.calls]

    def kwargs_for(self, event: str) -> dict:
        for e, kw in self.calls:
            if e == event:
                return kw
        raise KeyError(f"Event '{event}' was never emitted.")


# ===========================================================================
# _ObservabilityMixin
# ===========================================================================


class TestObservabilityMixin:
    """_ObservabilityMixin — lifecycle event dispatch."""

    def test_no_logger_by_default(self):
        obj = _FakeEstimator()
        assert obj._event_logger is None

    def test_emit_event_silent_when_no_logger(self):
        """_emit_event must never raise when no logger is attached."""
        obj = _FakeEstimator()
        obj._emit_event("anything", foo=1)  # should not raise

    def test_emit_event_calls_logger_info(self):
        logger = _RecordingLogger()
        obj = _FakeEstimator()
        obj._event_logger = logger
        obj._emit_event("fit_started", n_samples=100)
        assert logger.events() == ["fit_started"]
        assert logger.kwargs_for("fit_started") == {"n_samples": 100}

    def test_emit_event_passes_all_kwargs(self):
        logger = _RecordingLogger()
        obj = _FakeEstimator()
        obj._event_logger = logger
        obj._emit_event("custom", a=1, b="two", c=3.0)
        assert logger.kwargs_for("custom") == {"a": 1, "b": "two", "c": 3.0}

    def test_replacing_logger_takes_effect_immediately(self):
        logger1 = _RecordingLogger()
        logger2 = _RecordingLogger()
        obj = _FakeEstimator()
        obj._event_logger = logger1
        obj._emit_event("first")
        obj._event_logger = logger2
        obj._emit_event("second")
        assert logger1.events() == ["first"]
        assert logger2.events() == ["second"]

    def test_setting_logger_to_none_silences_again(self):
        logger = _RecordingLogger()
        obj = _FakeEstimator()
        obj._event_logger = logger
        obj._emit_event("before")
        obj._event_logger = None
        obj._emit_event("after")  # should not raise or record
        assert logger.events() == ["before"]


class TestNoOpEventLogger:
    """_NoOpEventLogger — must never raise or produce side effects."""

    def test_info_accepts_any_kwargs(self):
        noop = _NoOpEventLogger()
        noop.info("event", a=1, b=[1, 2, 3], c={"nested": True})

    def test_info_returns_none(self):
        noop = _NoOpEventLogger()
        result = noop.info("event")
        assert result is None


# ===========================================================================
# _ObservabilityMixin — full lifecycle event names (Phase 4 inventory)
# ===========================================================================


_EXPECTED_FIT_EVENTS = [
    "fit_started",
    "data_module_created",
    "data_prepared",
    "task_model_created",
    "model_built",
    "training_started",
    "training_completed",
    "fit_completed",
]

_EXPECTED_PREDICT_EVENTS = [
    "predict_started",
    "predict_completed",
]

_EXPECTED_SERIALIZATION_EVENTS_SAVE = ["save_started", "save_completed"]
_EXPECTED_SERIALIZATION_EVENTS_LOAD = ["load_completed"]


class TestEventInventoryViaFastTrainer:
    """Confirm the full Phase 4 event inventory fires on a real fit/predict call.

    Uses a very small dataset and a fast TrainerConfig so the test completes
    quickly.  We only check that the expected event names appear; we do not
    validate kwargs values here (those are checked by the smoke tests in
    test_dependency_inversion.py).
    """

    @pytest.fixture(scope="class")
    def fitted_clf(self):
        from deeptab.configs import TrainerConfig
        from deeptab.models.mlp import MLPClassifier

        clf = MLPClassifier(trainer_config=TrainerConfig(max_epochs=2, patience=2, lr_patience=2))
        logger = _RecordingLogger()
        clf._event_logger = logger

        X = np.random.default_rng(42).standard_normal((60, 4))
        y = np.array([0, 1, 2] * 20)
        clf.fit(X, y)
        return clf, logger, X

    def test_fit_events_fired(self, fitted_clf):
        _, logger, _ = fitted_clf
        fired = set(logger.events())
        for event in _EXPECTED_FIT_EVENTS:
            assert event in fired, f"Expected fit event '{event}' was not emitted."

    def test_fit_started_carries_n_samples(self, fitted_clf):
        _, logger, _ = fitted_clf
        kw = logger.kwargs_for("fit_started")
        assert kw["n_samples"] == 60

    def test_training_started_carries_max_epochs_and_batch_size(self, fitted_clf):
        _, logger, _ = fitted_clf
        kw = logger.kwargs_for("training_started")
        assert "max_epochs" in kw
        assert "batch_size" in kw

    def test_model_built_carries_n_params(self, fitted_clf):
        _, logger, _ = fitted_clf
        kw = logger.kwargs_for("model_built")
        assert "n_params" in kw
        assert isinstance(kw["n_params"], int)
        assert kw["n_params"] > 0

    def test_training_completed_carries_best_val_loss(self, fitted_clf):
        _, logger, _ = fitted_clf
        kw = logger.kwargs_for("training_completed")
        assert "best_val_loss" in kw

    def test_predict_events_fired(self, fitted_clf):
        clf, _, X = fitted_clf
        predict_logger = _RecordingLogger()
        clf._event_logger = predict_logger
        clf.predict(X)
        fired = set(predict_logger.events())
        for event in _EXPECTED_PREDICT_EVENTS:
            assert event in fired, f"Expected predict event '{event}' was not emitted."

    def test_predict_started_carries_n_samples(self, fitted_clf):
        clf, _, X = fitted_clf
        predict_logger = _RecordingLogger()
        clf._event_logger = predict_logger
        clf.predict(X)
        kw = predict_logger.kwargs_for("predict_started")
        assert kw["n_samples"] == len(X)

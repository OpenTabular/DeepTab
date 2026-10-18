"""Tests for observability mixin."""

from __future__ import annotations

from deeptab.models._mixins.observability import _NoOpEventLogger, _ObservabilityMixin


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


_EXPECTED_FIT_EVENTS = [
    "fit.started",
    "data.created",
    "model.created",
    "train.started",
    "train.completed",
    "fit.completed",
]

_EXPECTED_PREDICT_EVENTS = [
    "predict_started",
    "predict_completed",
]

_EXPECTED_SERIALIZATION_EVENTS_SAVE = ["save_started", "save_completed"]

_EXPECTED_SERIALIZATION_EVENTS_LOAD = ["load_completed"]

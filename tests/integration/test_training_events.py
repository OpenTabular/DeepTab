"""Tests for training events."""

from __future__ import annotations

import numpy as np
import pytest


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


class TestEventInventoryViaFastTrainer:
    """Verify lifecycle events and their payloads during real fitting and prediction."""

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
        kw = logger.kwargs_for("fit.started")
        assert kw["n_samples"] == 60

    def test_training_started_carries_max_epochs_and_batch_size(self, fitted_clf):
        _, logger, _ = fitted_clf
        kw = logger.kwargs_for("train.started")
        assert "max_epochs" in kw
        assert "batch_size" in kw

    def test_model_built_carries_n_params(self, fitted_clf):
        _, logger, _ = fitted_clf
        kw = logger.kwargs_for("model.created")
        assert "n_params" in kw
        assert isinstance(kw["n_params"], int)
        assert kw["n_params"] > 0

    def test_training_completed_carries_best_val_loss(self, fitted_clf):
        _, logger, _ = fitted_clf
        kw = logger.kwargs_for("train.completed")
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

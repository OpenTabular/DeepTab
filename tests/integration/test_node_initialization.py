"""Tests for node initialization."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightning.pytorch import Trainer as LightningTrainer

from deeptab.models import NODEClassifier

RANDOM_STATE = 0

FIT_KWARGS: dict[str, Any] = {"max_epochs": 1, "batch_size": 32}

N = 120

N_FEATURES = 4


def _make_clf_data():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N, N_FEATURES))
    y = rng.integers(0, 2, size=N)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)]), y  # type: ignore[call-overload]


class TestDataAwareInitSkipsSanityCheck:
    def test_initialize_never_called_during_sanity_check(self, monkeypatch):
        from deeptab.nn.blocks.node import ODST

        X, y = _make_clf_data()

        sanity_check_active = {"value": False}
        original_run_sanity_check = LightningTrainer._run_sanity_check

        def _tracked_run_sanity_check(self):
            sanity_check_active["value"] = True
            try:
                return original_run_sanity_check(self)
            finally:
                sanity_check_active["value"] = False

        monkeypatch.setattr(LightningTrainer, "_run_sanity_check", _tracked_run_sanity_check)

        calls_during_sanity_check = []
        original_initialize = ODST.initialize

        def _tracked_initialize(self, *args, **kwargs):
            calls_during_sanity_check.append(sanity_check_active["value"])
            return original_initialize(self, *args, **kwargs)

        monkeypatch.setattr(ODST, "initialize", _tracked_initialize)

        clf = NODEClassifier()
        clf.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)

        assert calls_during_sanity_check, "initialize() should have fired at least once (warm-up + first batch)"
        assert not any(calls_during_sanity_check), "initialize() must never fire while the sanity check is running"

    def test_predictions_are_finite(self):
        """End-to-end guard: skipping init during the sanity check must not
        leave thresholds uninitialized (NaN) for real training/prediction."""
        X, y = _make_clf_data()
        clf = NODEClassifier()
        clf.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)
        preds = clf.predict_proba(X)
        assert np.all(np.isfinite(preds))

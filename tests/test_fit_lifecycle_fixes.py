"""Regression tests for GH-452 (state and lifecycle defects).

Covers the sub-bugs not already exercised by ``tests/test_reproducibility.py``
(seeding/random_state precedence) and ``tests/test_profile.py`` (dry-run state
leakage):

* A failed second ``fit()`` call must leave ``is_fitted_`` honestly ``False``
  rather than reporting a stale ``True`` from a previous successful fit.
* ``fit(**trainer_kwargs)`` must accept a caller-supplied ``callbacks=``
  argument (a list or a single ``Callback`` instance) instead of crashing
  with a duplicate-keyword ``TypeError``, and that callback must actually run.
* NODE/ENODE's data-aware threshold initialization must not fire on
  Lightning's eval-mode sanity-check batch (which is drawn from the
  validation split), only on a genuine training-mode batch.
* ``predict(device=...)`` must actually route inference through the
  requested device rather than silently ignoring it, and must reject an
  invalid device string.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from lightning.pytorch import Trainer as LightningTrainer
from lightning.pytorch.callbacks import Callback

from deeptab.core.exceptions import NotFittedError
from deeptab.models import MLPRegressor, NODEClassifier

RANDOM_STATE = 0
FIT_KWARGS: dict[str, Any] = {"max_epochs": 1, "batch_size": 32}
N = 120
N_FEATURES = 4


def _make_reg_data():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N, N_FEATURES))
    y = rng.standard_normal(N)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)]), y  # type: ignore[call-overload]


def _make_clf_data():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N, N_FEATURES))
    y = rng.integers(0, 2, size=N)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)]), y  # type: ignore[call-overload]


# ---------------------------------------------------------------------------
# Bug 3 — a failed fit() must not leave is_fitted_ stuck at True
# ---------------------------------------------------------------------------


class TestFailedFitLeavesIsFittedFalse:
    def test_failed_second_fit_sets_is_fitted_false(self, monkeypatch):
        X, y = _make_reg_data()
        reg = MLPRegressor()
        reg.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)
        assert reg.is_fitted_ is True

        def _raise_mid_fit(self, *args, **kwargs):
            raise RuntimeError("simulated mid-fit crash")

        monkeypatch.setattr(LightningTrainer, "fit", _raise_mid_fit)

        with pytest.raises(RuntimeError, match="simulated mid-fit crash"):
            reg.fit(X, y, random_state=RANDOM_STATE + 1, **FIT_KWARGS)

        assert reg.is_fitted_ is False

    def test_predict_raises_not_fitted_after_failed_refit(self, monkeypatch):
        X, y = _make_reg_data()
        reg = MLPRegressor()
        reg.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)

        def _raise_mid_fit(self, *args, **kwargs):
            raise RuntimeError("simulated mid-fit crash")

        monkeypatch.setattr(LightningTrainer, "fit", _raise_mid_fit)
        with pytest.raises(RuntimeError):
            reg.fit(X, y, random_state=RANDOM_STATE + 1, **FIT_KWARGS)

        with pytest.raises(NotFittedError):
            reg.predict(X)


# ---------------------------------------------------------------------------
# Bug 4 — fit(callbacks=...) must not crash and must actually run
# ---------------------------------------------------------------------------


class _RecordingCallback(Callback):
    def __init__(self):
        self.epoch_ends = 0

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_ends += 1


class TestFitAcceptsCustomCallbacks:
    def test_callback_list_is_accepted_and_runs(self):
        X, y = _make_reg_data()
        reg = MLPRegressor()
        cb = _RecordingCallback()
        reg.fit(X, y, random_state=RANDOM_STATE, callbacks=[cb], **FIT_KWARGS)
        assert cb.epoch_ends > 0, "Custom callback's on_train_epoch_end must have fired"

    def test_single_callback_instance_is_accepted(self):
        X, y = _make_reg_data()
        reg = MLPRegressor()
        cb = _RecordingCallback()
        reg.fit(X, y, random_state=RANDOM_STATE, callbacks=cb, **FIT_KWARGS)
        assert cb.epoch_ends > 0

    def test_builtin_callbacks_still_present(self):
        """A custom callbacks= must augment, not replace, the built-in ones."""
        X, y = _make_reg_data()
        reg = MLPRegressor()
        cb = _RecordingCallback()
        reg.fit(X, y, random_state=RANDOM_STATE, callbacks=[cb], **FIT_KWARGS)
        assert reg._best_model_path, "ModelCheckpoint callback must still have produced a checkpoint"


# ---------------------------------------------------------------------------
# Bug 5 (NODE/ENODE) — data-aware init must not use the sanity-check batch
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Bug 6 — predict(device=...) must actually take effect
# ---------------------------------------------------------------------------


class TestPredictDeviceOverride:
    def test_explicit_cpu_device_matches_default(self):
        X, y = _make_reg_data()
        reg = MLPRegressor()
        reg.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)

        default_preds = reg.predict(X)
        cpu_preds = reg.predict(X, device="cpu")
        np.testing.assert_array_almost_equal(default_preds, cpu_preds, decimal=4)

    def test_invalid_device_string_raises(self):
        X, y = _make_reg_data()
        reg = MLPRegressor()
        reg.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)

        with pytest.raises(RuntimeError, match="Invalid device string"):
            reg.predict(X, device="not-a-real-device")

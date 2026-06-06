"""Tests for InspectionMixin.profile().

Covers:
* successful dry-run (model discarded afterwards)
* successful profile on an already-built model (state preserved)
* all required keys present in the returned dict
* parameter and memory estimates are consistent
* forward-pass timing is positive
* build failure returns builds=False and a non-empty error string
* dry_run=False leaves the model built after the call
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from deeptab.models import MLPClassifier, MLPRegressor

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

RANDOM_STATE = 0
FIT_KWARGS: dict[str, Any] = {"max_epochs": 1, "batch_size": 32}

_REQUIRED_KEYS = {
    "builds",
    "error",
    "device",
    "dtype",
    "total_params",
    "trainable_params",
    "memory_mb",
    "batch_shape",
    "output_shape",
    "loss_fct",
    "forward_ms_median",
    "forward_ms_min",
    "describe",
    "runtime",
}


def _binary_data(n: int = 200, n_features: int = 5):
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 2, size=n)
    df = pd.DataFrame({f"f{i}": X[:, i] for i in range(n_features)})
    return df, y


def _regression_data(n: int = 200, n_features: int = 5):
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((n, n_features))
    y = rng.standard_normal(n)
    df = pd.DataFrame({f"f{i}": X[:, i] for i in range(n_features)})
    return df, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProfileKeys:
    """All required keys are always present in the returned dict."""

    def test_dry_run_has_all_keys(self):
        X, y = _binary_data()
        clf = MLPClassifier()
        result = clf.profile(X, y, dry_run=True, random_state=RANDOM_STATE)
        assert _REQUIRED_KEYS <= result.keys(), f"Missing keys: {_REQUIRED_KEYS - result.keys()}"

    def test_failed_build_has_all_keys(self, monkeypatch):
        """Even when build raises, all keys must be present (with builds=False)."""
        X, y = _binary_data()
        clf = MLPClassifier()
        monkeypatch.setattr(clf, "build_model", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        result = clf.profile(X, y, dry_run=True, random_state=RANDOM_STATE)
        assert _REQUIRED_KEYS <= result.keys()


class TestProfileDryRun:
    """dry_run=True leaves the estimator in its pre-call state."""

    def test_unfitted_estimator_remains_unfitted(self):
        X, y = _binary_data()
        clf = MLPClassifier()
        assert not clf.built

        result = clf.profile(X, y, dry_run=True, random_state=RANDOM_STATE)

        assert result["builds"] is True
        assert not clf.built, "Estimator should remain unbuilt after dry_run=True"
        assert clf.task_model is None

    def test_already_fitted_estimator_state_preserved(self):
        X, y = _binary_data()
        clf = MLPClassifier()
        clf.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)
        assert clf.built

        result = clf.profile(X, y, dry_run=True, random_state=RANDOM_STATE)

        assert result["builds"] is True
        # Model was already built — dry_run must NOT discard the existing state
        assert clf.built
        assert clf.task_model is not None

    def test_dry_run_false_leaves_model_built(self):
        X, y = _binary_data()
        clf = MLPClassifier()
        assert not clf.built

        result = clf.profile(X, y, dry_run=False, random_state=RANDOM_STATE)

        assert result["builds"] is True
        assert clf.built, "dry_run=False should leave the model built"


class TestProfileContent:
    """Returned values are numerically sensible."""

    def test_builds_true_on_success(self):
        X, y = _binary_data()
        result = MLPClassifier().profile(X, y, dry_run=True, random_state=RANDOM_STATE)
        assert result["builds"] is True
        assert result["error"] is None

    def test_params_positive(self):
        X, y = _binary_data()
        result = MLPClassifier().profile(X, y, dry_run=True, random_state=RANDOM_STATE)
        assert result["total_params"] > 0
        assert result["trainable_params"] > 0
        assert result["trainable_params"] <= result["total_params"]

    def test_memory_mb_consistent_with_params(self):
        X, y = _binary_data()
        result = MLPClassifier().profile(X, y, dry_run=True, random_state=RANDOM_STATE)
        # float32 → 4 bytes/param
        expected_min = result["total_params"] * 2 / (1024**2)  # bfloat16 lower bound
        expected_max = result["total_params"] * 8 / (1024**2)  # float64 upper bound
        assert expected_min <= result["memory_mb"] <= expected_max

    def test_dtype_is_string(self):
        X, y = _binary_data()
        result = MLPClassifier().profile(X, y, dry_run=True, random_state=RANDOM_STATE)
        assert isinstance(result["dtype"], str)
        assert "torch." not in result["dtype"]

    def test_device_is_string(self):
        X, y = _binary_data()
        result = MLPClassifier().profile(X, y, dry_run=True, random_state=RANDOM_STATE)
        assert isinstance(result["device"], str)

    def test_forward_timing_positive(self):
        X, y = _binary_data()
        result = MLPClassifier().profile(X, y, dry_run=True, n_forward_passes=3, random_state=RANDOM_STATE)
        assert result["forward_ms_median"] is not None
        assert result["forward_ms_median"] > 0
        assert result["forward_ms_min"] is not None
        assert result["forward_ms_min"] > 0
        assert result["forward_ms_min"] <= result["forward_ms_median"]

    def test_output_shape_is_list(self):
        X, y = _binary_data()
        result = MLPClassifier().profile(X, y, dry_run=True, random_state=RANDOM_STATE)
        assert isinstance(result["output_shape"], list)
        assert len(result["output_shape"]) >= 1

    def test_batch_shape_is_dict(self):
        X, y = _binary_data()
        result = MLPClassifier().profile(X, y, dry_run=True, random_state=RANDOM_STATE)
        assert isinstance(result["batch_shape"], dict)

    def test_loss_fct_name_is_string(self):
        X, y = _binary_data()
        result = MLPClassifier().profile(X, y, dry_run=True, random_state=RANDOM_STATE)
        assert isinstance(result["loss_fct"], str)
        # Binary classification → default BCE loss
        assert "BCE" in result["loss_fct"] or "bce" in result["loss_fct"].lower()

    def test_describe_and_runtime_dicts_populated(self):
        X, y = _binary_data()
        result = MLPClassifier().profile(X, y, dry_run=True, random_state=RANDOM_STATE)
        assert isinstance(result["describe"], dict)
        assert isinstance(result["runtime"], dict)

    def test_regressor_profile(self):
        X, y = _regression_data()
        result = MLPRegressor().profile(X, y, dry_run=True, random_state=RANDOM_STATE)
        assert result["builds"] is True
        assert result["total_params"] > 0


class TestProfileFailure:
    """Graceful failure reporting when build raises."""

    def test_builds_false_on_bad_data(self, monkeypatch):
        X, y = _binary_data()
        clf = MLPClassifier()

        def _raise(*a, **kw):
            raise RuntimeError("intentional build failure")

        monkeypatch.setattr(clf, "build_model", _raise)

        result = clf.profile(X, y, dry_run=True, random_state=RANDOM_STATE)
        assert result["builds"] is False
        assert result["error"] is not None
        assert len(result["error"]) > 0

    def test_estimator_state_unchanged_after_failure(self, monkeypatch):
        X, y = _binary_data()
        clf = MLPClassifier()

        def _raise(*a, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(clf, "build_model", _raise)

        clf.profile(X, y, dry_run=True, random_state=RANDOM_STATE)
        assert not clf.built

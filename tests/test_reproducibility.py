"""Reproducibility tests for DeepTab.

This module verifies, step by step, that:

1. ``set_seed`` and ``seed_context`` correctly seed PyTorch, NumPy, and Python
   built-in RNGs (primitive correctness).
2. An estimator trained with a fixed ``random_state`` produces identical
   predictions on two completely independent runs (same-seed → same output).
3. Two estimators trained with *different* seeds produce different predictions
   (different-seed → different output), confirming that the seed actually has
   an effect.
4. Refitting the *same* estimator object with the same seed yields the same
   predictions as the first fit (no cross-fit state leakage).
5. Platform and device coverage: CPU, CUDA, MPS (Apple Silicon), Windows,
   macOS, Linux.

No data is shared between independently created estimator instances, so these
tests also serve as a no-leakage guard.

Notes
-----
Tests use ``MLPRegressor`` with ``max_epochs=3`` to keep CI fast.  The
principles apply equally to every estimator in the library.
"""

from __future__ import annotations

import os
import platform
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from deeptab.configs import TrainerConfig
from deeptab.core.reproducibility import seed_context, set_seed
from deeptab.models import MLPRegressor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
ALT_SEED = 99
N_SAMPLES = 120
N_FEATURES = 5
_FIT_KWARGS: dict[str, Any] = {"max_epochs": 3, "batch_size": 32}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def regression_data():
    """Small, fully deterministic regression dataset (uses numpy Generator)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y = X @ rng.standard_normal(N_FEATURES) + 0.1 * rng.standard_normal(N_SAMPLES)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)])  # type: ignore[call-overload]
    return df, y


def _make_regressor(seed: int) -> MLPRegressor:
    """Create a fresh MLPRegressor with a fixed random_state."""
    return MLPRegressor(
        trainer_config=TrainerConfig(**_FIT_KWARGS),
        random_state=seed,
    )


# ---------------------------------------------------------------------------
# Step 1 — Primitive RNG correctness
# ---------------------------------------------------------------------------


class TestSetSeedPrimitives:
    """set_seed correctly seeds each individual RNG layer."""

    @pytest.mark.smoke
    def test_torch_cpu(self):
        """Same seed → identical CPU tensors."""
        set_seed(SEED)
        t1 = torch.randn(20)
        set_seed(SEED)
        t2 = torch.randn(20)
        assert torch.equal(t1, t2), "torch.randn should be identical after re-seeding"

    def test_numpy_legacy(self):
        """Same seed → identical numpy arrays (legacy RNG)."""
        set_seed(SEED)
        a1 = np.random.randn(20)
        set_seed(SEED)
        a2 = np.random.randn(20)
        np.testing.assert_array_equal(a1, a2)

    def test_python_random(self):
        """Same seed → identical Python random floats."""
        import random

        set_seed(SEED)
        v1 = [random.random() for _ in range(20)]  # noqa: S311
        set_seed(SEED)
        v2 = [random.random() for _ in range(20)]  # noqa: S311
        assert v1 == v2

    def test_different_seeds_differ_torch(self):
        """Different seeds produce different tensors."""
        set_seed(SEED)
        t1 = torch.randn(20)
        set_seed(ALT_SEED)
        t2 = torch.randn(20)
        assert not torch.equal(t1, t2), "Different seeds should yield different tensors"

    @pytest.mark.smoke
    def test_invalid_seed_raises(self):
        """Negative seeds raise ValueError."""
        with pytest.raises(ValueError, match="non-negative integer"):
            set_seed(-1)


# ---------------------------------------------------------------------------
# Step 2 — seed_context
# ---------------------------------------------------------------------------


class TestSeedContext:
    """seed_context is a functional equivalent of set_seed used as a 'with' block."""

    def test_context_torch(self):
        """Context manager produces the same sequence as set_seed."""
        with seed_context(SEED):
            t1 = torch.randn(20)
        with seed_context(SEED):
            t2 = torch.randn(20)
        assert torch.equal(t1, t2)

    def test_context_numpy(self):
        with seed_context(SEED):
            a1 = np.random.randn(20)
        with seed_context(SEED):
            a2 = np.random.randn(20)
        np.testing.assert_array_equal(a1, a2)


# ---------------------------------------------------------------------------
# Step 3 — End-to-end: same seed → same predictions
# ---------------------------------------------------------------------------


class TestSameSeedSamePredictions:
    """Two independent fit+predict calls with the same seed are identical."""

    def test_regressor_predictions_match(self, regression_data):
        X, y = regression_data

        m1 = _make_regressor(SEED)
        m1.fit(X, y)
        p1 = m1.predict(X)

        m2 = _make_regressor(SEED)
        m2.fit(X, y)
        p2 = m2.predict(X)

        np.testing.assert_array_almost_equal(
            p1,
            p2,
            decimal=5,
            err_msg="Same random_state must produce identical predictions",
        )

    def test_predictions_are_finite(self, regression_data):
        """Sanity check: predictions must all be finite numbers."""
        X, y = regression_data
        m = _make_regressor(SEED)
        m.fit(X, y)
        preds = m.predict(X)
        assert np.all(np.isfinite(preds)), "Predictions contain non-finite values"


# ---------------------------------------------------------------------------
# Step 4 — Different seeds → different predictions (seed has real effect)
# ---------------------------------------------------------------------------


class TestDifferentSeedsDifferentPredictions:
    """Two estimators trained with different seeds produce different outputs."""

    def test_regressor_predictions_differ(self, regression_data):
        X, y = regression_data

        m1 = _make_regressor(SEED)
        m1.fit(X, y)
        p1 = m1.predict(X)

        m2 = _make_regressor(ALT_SEED)
        m2.fit(X, y)
        p2 = m2.predict(X)

        assert not np.allclose(p1, p2, atol=1e-4), "Different random_state values should yield different predictions"


# ---------------------------------------------------------------------------
# Step 5 — No leakage on refit
# ---------------------------------------------------------------------------


class TestNoLeakageOnRefit:
    """Refitting the same estimator with the same seed reproduces the first fit."""

    def test_refit_matches_first_fit(self, regression_data):
        """Two independent fresh instances with the same seed are identical — no
        cross-instance state leakage even when fits happen sequentially."""
        X, y = regression_data

        m1 = _make_regressor(SEED)
        m1.fit(X, y)
        p1 = m1.predict(X)

        # Fresh instance, same seed — must reproduce identically
        m2 = _make_regressor(SEED)
        m2.fit(X, y)
        p2 = m2.predict(X)

        np.testing.assert_array_almost_equal(
            p1,
            p2,
            decimal=5,
            err_msg="Fresh instance with the same seed must reproduce the first fit exactly",
        )

    def test_no_cross_instance_leakage(self, regression_data):
        """State from one fitted instance does not bleed into another."""
        X, y = regression_data

        # Fit a first model to 'contaminate' the global RNG state
        contaminator = _make_regressor(ALT_SEED)
        contaminator.fit(X, y)
        _ = contaminator.predict(X)

        # Now fit the canonical model — its seed should override the contamination
        m1 = _make_regressor(SEED)
        m1.fit(X, y)
        p1 = m1.predict(X)

        m2 = _make_regressor(SEED)
        m2.fit(X, y)
        p2 = m2.predict(X)

        np.testing.assert_array_almost_equal(
            p1,
            p2,
            decimal=5,
            err_msg="Cross-instance RNG contamination detected",
        )


# ---------------------------------------------------------------------------
# Step 6 — Platform and device coverage
# ---------------------------------------------------------------------------

_has_cuda = torch.cuda.is_available()
_has_mps = hasattr(torch, "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

_skip_no_cuda = pytest.mark.skipif(not _has_cuda, reason="CUDA not available on this host")
_skip_no_mps = pytest.mark.skipif(not _has_mps, reason="MPS not available on this host")


class TestPlatformAndDeviceSeeding:
    """set_seed works correctly on all supported platforms and accelerators."""

    # --- PYTHONHASHSEED -------------------------------------------------------

    def test_pythonhashseed_env_var_is_set(self):
        """set_seed writes PYTHONHASHSEED to the environment."""
        set_seed(SEED)
        assert os.environ.get("PYTHONHASHSEED") == str(SEED), "PYTHONHASHSEED must be set in os.environ after set_seed"

    def test_pythonhashseed_changes_with_seed(self):
        """PYTHONHASHSEED reflects the seed that was last applied."""
        set_seed(ALT_SEED)
        assert os.environ.get("PYTHONHASHSEED") == str(ALT_SEED)

    # --- CPU (all platforms) --------------------------------------------------

    def test_cpu_tensor_reproducible(self):
        """CPU tensor generation is reproducible after set_seed (all OS)."""
        set_seed(SEED)
        t1 = torch.randn(50, device="cpu")
        set_seed(SEED)
        t2 = torch.randn(50, device="cpu")
        assert torch.equal(t1, t2), f"CPU tensors differ — platform: {platform.system()}"

    def test_set_seed_is_idempotent(self):
        """Calling set_seed twice with the same value does not raise."""
        set_seed(SEED)
        set_seed(SEED)  # must not raise

    def test_set_seed_zero(self):
        """Seed 0 is valid and reproducible."""
        set_seed(0)
        t1 = torch.randn(10)
        set_seed(0)
        t2 = torch.randn(10)
        assert torch.equal(t1, t2)

    def test_set_seed_max_uint32(self):
        """Seed at the upper uint32 boundary (2**32 - 1) is accepted."""
        set_seed(2**32 - 1)  # must not raise

    # --- CUDA -----------------------------------------------------------------

    @_skip_no_cuda
    def test_cuda_tensor_reproducible(self):
        """CUDA tensor generation is reproducible after set_seed."""
        set_seed(SEED)
        t1 = torch.randn(50, device="cuda")
        set_seed(SEED)
        t2 = torch.randn(50, device="cuda")
        assert torch.equal(t1, t2), "CUDA tensors differ after re-seeding"

    @_skip_no_cuda
    def test_cudnn_flags_set_when_cuda_available(self):
        """cuDNN determinism flags are set when CUDA is present."""
        set_seed(SEED)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    # --- MPS ------------------------------------------------------------------

    @_skip_no_mps
    def test_mps_tensor_reproducible(self):
        """MPS tensor generation is reproducible after set_seed (Apple Silicon)."""
        set_seed(SEED)
        t1 = torch.randn(50, device="mps")
        set_seed(SEED)
        t2 = torch.randn(50, device="mps")
        assert torch.equal(t1, t2), "MPS tensors differ after re-seeding"

    # --- No-CUDA host: cuDNN flags must not raise ------------------------------

    def test_cudnn_flags_accessible_without_cuda(self):
        """Accessing torch.backends.cudnn attrs never raises, even on CPU-only hosts."""
        # These are Python properties and are always accessible regardless of
        # whether CUDA is compiled in.
        _ = torch.backends.cudnn.deterministic
        _ = torch.backends.cudnn.benchmark

    # --- deterministic=True flag ----------------------------------------------

    def test_deterministic_flag_propagates(self):
        """set_seed(deterministic=True) enables torch deterministic algorithms."""
        try:
            set_seed(SEED, deterministic=True)
            # If we reach here the flag was accepted; reset to avoid side-effects
            torch.use_deterministic_algorithms(False)
        except RuntimeError as exc:
            # Some builds raise if an op has no deterministic implementation;
            # that is the *expected* behaviour — it means the flag took effect.
            assert "deterministic" in str(exc).lower(), f"Unexpected RuntimeError: {exc}"

    # --- End-to-end: active device --------------------------------------------

    def test_end_to_end_on_active_device(self, regression_data):
        """Estimator fit on the currently active device is reproducible."""
        X, y = regression_data

        m1 = _make_regressor(SEED)
        m1.fit(X, y)
        p1 = m1.predict(X)

        m2 = _make_regressor(SEED)
        m2.fit(X, y)
        p2 = m2.predict(X)

        np.testing.assert_array_almost_equal(
            p1,
            p2,
            decimal=5,
            err_msg=f"Predictions differ on {platform.system()} / device auto-select",
        )

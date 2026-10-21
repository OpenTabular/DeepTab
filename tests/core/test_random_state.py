"""Tests for random state."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from deeptab.core.reproducibility import seed_context, set_seed

SEED = 42

ALT_SEED = 99

N_SAMPLES = 120

N_FEATURES = 5

_FIT_KWARGS: dict[str, Any] = {"max_epochs": 3, "batch_size": 32}


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

    @pytest.mark.parametrize("seed", [np.int64(SEED), np.uint32(SEED), np.int32(SEED)])
    def test_numpy_integer_seed_matches_python_int(self, seed):
        set_seed(SEED)
        expected = torch.randn(5)
        set_seed(seed)
        torch.testing.assert_close(torch.randn(5), expected)

    @pytest.mark.parametrize("seed", [1.0, "7", np.float64(7.0)])
    def test_non_integer_seed_is_still_rejected(self, seed):
        with pytest.raises(ValueError, match="non-negative integer"):
            set_seed(seed)


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


_has_cuda = torch.cuda.is_available()

_has_mps = hasattr(torch, "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

_skip_no_cuda = pytest.mark.skipif(not _has_cuda, reason="CUDA not available on this host")

_skip_no_mps = pytest.mark.skipif(not _has_mps, reason="MPS not available on this host")


def test_estimator_accepts_numpy_integer_random_state():
    import pandas as pd

    from deeptab.configs import TrainerConfig
    from deeptab.models import MLPRegressor

    features = pd.DataFrame({"a": np.linspace(-1.0, 1.0, 64), "b": np.linspace(1.0, -1.0, 64)})
    targets = features["a"].to_numpy()
    fit_kwargs: dict[str, Any] = {"accelerator": "cpu", "logger": False, "enable_progress_bar": False}
    predictions = []
    for seed in (np.int64(SEED), SEED):
        model = MLPRegressor(trainer_config=TrainerConfig(max_epochs=1, batch_size=16), random_state=seed)
        predictions.append(model.fit(features, targets, **fit_kwargs).predict(features))
    np.testing.assert_array_equal(predictions[0], predictions[1])

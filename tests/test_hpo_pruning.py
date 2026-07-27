"""Regression tests for the HPO driver's pruning and best-params handling.

The objective and pruning arithmetic used ``best * 1.5`` / ``best * 100``,
which invert when the validation loss is negative -- routine for LSS models
whose val_loss is a mean NLL.
"""

from typing import Any, cast

import numpy as np
import pandas as pd

from deeptab.models import MLPRegressor
from deeptab.models._mixins.hpo import round_to_nearest_16


def _pruning_threshold(best):
    """Mirror of the threshold expression in _FitMixin.optimize_hparams."""
    return best + 0.5 * abs(best)


def _failure_penalty(best):
    """Mirror of the failed-trial penalty in _FitMixin.optimize_hparams."""
    return best + 100.0 * abs(best) + 1.0


class TestSignSafeObjective:
    def test_pruning_threshold_is_worse_than_best(self):
        for best in (-12.5, -1.0, -0.001, 0.0, 0.5, 42.0):
            assert _pruning_threshold(best) >= best

    def test_failure_penalty_is_worse_than_best(self):
        """A crashing trial must never look better than the incumbent."""
        for best in (-12.5, -1.0, -0.001, 0.0, 0.5, 42.0):
            assert _failure_penalty(best) > best


class TestHeadLayerSizeLength:
    def test_length_key_matches_the_size_prefix(self):
        """Why the final apply loop needs an explicit length branch."""
        assert "head_layer_size_length".startswith("head_layer_size_")

    def test_length_value_would_round_to_zero(self):
        """Rounding the length as if it were a layer size yields a 0-width layer."""
        assert all(round_to_nearest_16(n) == 0 for n in range(1, 6))


class TestValLossTracking:
    def test_val_losses_excludes_sanity_check(self):
        """The sanity-check validation must not enter val_losses.

        Recording it shifts epoch_val_loss_at(e) to the loss of epoch e-1,
        which skews the HPO pruning baseline.
        """
        rng = np.random.RandomState(0)
        X = pd.DataFrame({"a": rng.randn(60), "b": rng.randn(60)})
        y = rng.randn(60)
        model = MLPRegressor()
        model.fit(X, y, max_epochs=2, batch_size=16, accelerator="cpu")
        assert len(cast(Any, model._task_model).val_losses) == 2

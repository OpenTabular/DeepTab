"""Regression tests for public-API robustness gaps found in the 2.0.0 review."""

import numpy as np
import pandas as pd
import pytest

from deeptab.models import MLPClassifier, MLPRegressor

FIT_KW = {"max_epochs": 1, "batch_size": 16, "accelerator": "cpu"}


def _data(n=40, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n)}), rng.randn(n)


def test_getstate_excludes_lightning_module():
    """__getstate__ nulled a phantom 'task_model' key, so pickling kept the real one."""
    X, y = _data()
    model = MLPRegressor()
    model.fit(X, y, **FIT_KW)

    state = model.__getstate__()
    assert state["_task_model"] is None
    assert state.get("task_model") is None


def test_fit_accepts_plain_lists():
    """The fit.started event read X.columns before ensure_dataframe ran."""
    X = [[1.0, 2.0], [2.0, 1.0], [0.5, 0.1], [1.5, 0.7]] * 10
    y = list(np.random.RandomState(0).randn(40))

    model = MLPRegressor()
    model.fit(X, y, **FIT_KW)
    assert model.predict(X).shape == (40,)


class TestSampleWeightValidation:
    def test_all_zero_sample_weight_raises(self):
        """Previously a cryptic torch.multinomial RuntimeError mid-training."""
        X, _ = _data()
        y = np.random.RandomState(1).choice([0, 1], len(X))

        with pytest.raises(ValueError, match="zero"):
            MLPClassifier().fit(X, y, sample_weight=np.zeros(len(X)), **FIT_KW)

    def test_negative_sample_weight_raises(self):
        X, _ = _data()
        y = np.random.RandomState(1).choice([0, 1], len(X))
        weights = np.ones(len(X))
        weights[0] = -1.0

        with pytest.raises(ValueError, match="non-negative"):
            MLPClassifier().fit(X, y, sample_weight=weights, **FIT_KW)

    def test_valid_sample_weight_still_accepted(self):
        X, _ = _data()
        y = np.random.RandomState(1).choice([0, 1], len(X))
        weights = np.ones(len(X))
        weights[:5] = 3.0

        model = MLPClassifier()
        model.fit(X, y, sample_weight=weights, **FIT_KW)
        assert model.predict(X).shape == (len(X),)

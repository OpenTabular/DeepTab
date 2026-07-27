"""Regression test: Trompt's init_rec prompt parameter must be initialized.

``nn.Parameter(torch.empty(...))`` keeps whatever was in the allocation. When
that memory held garbage the model produced NaN logits -- observed as
intermittent CI failures of the TromptClassifier/TromptLSS model tests.
"""

import numpy as np
import pandas as pd
import torch

from deeptab.models.experimental import TromptRegressor


def _data(n=40, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n)}), rng.randn(n)


def test_init_rec_is_initialized_not_left_as_raw_allocation(monkeypatch):
    """Simulate an allocation full of garbage; the constructor must overwrite it."""
    real_empty = torch.empty

    def poisoned_empty(*args, **kwargs):
        tensor = real_empty(*args, **kwargs)
        if tensor.is_floating_point():
            tensor.fill_(float("nan"))
        return tensor

    monkeypatch.setattr(torch, "empty", poisoned_empty)

    X, y = _data()
    model = TromptRegressor()
    model.build_model(X, y)

    estimator = model._task_model.estimator  # type: ignore[union-attr]
    assert torch.isfinite(estimator.init_rec).all()


def test_trompt_predictions_are_finite():
    X, y = _data()
    model = TromptRegressor()
    model.fit(X, y, max_epochs=1, batch_size=16, accelerator="cpu")
    assert np.isfinite(model.predict(X)).all()

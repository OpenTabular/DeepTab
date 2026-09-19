"""Regression tests for the public encode() API.

encode() never switched the model to eval mode, so embeddings taken after fit()
were dropout-corrupted and differed between calls. The LSS override additionally
unpacked 2 values from the 3-element batches preprocess_new_data yields, so it
always raised ValueError.
"""

import numpy as np
import pandas as pd

from deeptab.configs import FTTransformerConfig
from deeptab.models import FTTransformerLSS, FTTransformerRegressor

QUIET = {
    "accelerator": "cpu",
    "devices": 1,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "logger": False,
}


def _data(n=48, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "g": rng.choice(list("xyz"), n)})
    return X, rng.normal(size=n)


def _config(dropout=0.5):
    return FTTransformerConfig(d_model=16, n_layers=1, n_heads=2, attn_dropout=dropout)


def test_encode_is_deterministic_after_fit():
    """Dropout must be disabled: two calls on the same rows must agree."""
    X, y = _data()
    model = FTTransformerRegressor(model_config=_config())
    model.fit(X, y, max_epochs=1, batch_size=16, **QUIET)

    first = np.asarray(model.encode(X))
    second = np.asarray(model.encode(X))
    np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-6)


def test_encode_restores_training_mode():
    X, y = _data()
    model = FTTransformerRegressor(model_config=_config())
    model.fit(X, y, max_epochs=1, batch_size=16, **QUIET)

    model._task_model.train()  # type: ignore[union-attr]
    model.encode(X)
    assert model._task_model.training is True  # type: ignore[union-attr]


def test_lss_encode_returns_embeddings():
    X, y = _data()
    model = FTTransformerLSS(model_config=_config(dropout=0.0))
    model.fit(X, y, family="normal", max_epochs=1, batch_size=16, **QUIET)

    encoded = np.asarray(model.encode(X))
    assert encoded.shape[0] == len(X)

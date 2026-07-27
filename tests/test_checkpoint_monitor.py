"""Regression test: ModelCheckpoint must track the metric the user monitors.

The best checkpoint's weights are restored at the end of fit(), so a
hardcoded val_loss/min checkpoint silently returned the wrong weights for
anyone monitoring a different metric.
"""

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import deeptab.models._mixins.fit as fit_mod
from deeptab.models import MLPRegressor


def _data(n=40, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n)}), rng.randn(n)


@pytest.mark.parametrize(
    ("monitor", "mode"),
    [("val_loss", "min"), ("train_loss_epoch", "min")],
)
def test_checkpoint_tracks_the_monitored_metric(monkeypatch, monitor, mode):
    recorded = {}

    class RecordingCheckpoint(ModelCheckpoint):
        def __init__(self, *args, **kwargs):
            recorded.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(fit_mod, "ModelCheckpoint", RecordingCheckpoint)

    X, y = _data()
    model = MLPRegressor()
    model.fit(X, y, monitor=monitor, mode=mode, max_epochs=1, batch_size=16, accelerator="cpu")

    assert recorded["monitor"] == monitor
    assert recorded["mode"] == mode


def test_early_stopping_and_checkpoint_agree():
    X, y = _data()
    model = MLPRegressor()
    model.fit(X, y, monitor="train_loss_epoch", mode="min", max_epochs=1, batch_size=16, accelerator="cpu")

    callbacks = cast(Any, model._trainer).callbacks
    early = next(c for c in callbacks if isinstance(c, EarlyStopping))
    ckpt = next(c for c in callbacks if isinstance(c, ModelCheckpoint))
    assert (ckpt.monitor, ckpt.mode) == (early.monitor, early.mode)

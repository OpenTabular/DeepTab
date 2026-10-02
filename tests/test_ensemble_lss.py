"""Regression tests: ensemble LSS models must keep their parameter axis.

TabM and Trompt unconditionally squeezed the trailing axis of their output.
For a distribution family with exactly one parameter (poisson, tweedie, ...)
that axis IS the parameter axis, so the distribution code indexed a 1-D tensor
and raised IndexError.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from deeptab.configs import TabMConfig, TrainerConfig
from deeptab.models import TabMLSS

QUIET = {
    "accelerator": "cpu",
    "devices": 1,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "logger": False,
}


def _count_data(n=48, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = np.abs(rng.normal(size=n)).round()
    return X, y


@pytest.mark.parametrize("family", ["poisson", "normal"])
def test_ensemble_lss_fits_single_and_multi_parameter_families(family):
    X, y = _count_data()
    model = TabMLSS(model_config=TabMConfig(), trainer_config=TrainerConfig(max_epochs=1, batch_size=16))
    model.fit(X, y, family=family, **QUIET)
    assert np.isfinite(np.asarray(model.predict(X), dtype=float)).all()


def test_ensemble_output_keeps_parameter_axis_in_lss_mode():
    from deeptab.architectures.tabm import TabM

    info = ({"a": {"dimension": 4, "preprocessing": None, "categories": None}}, {}, {})
    lss_model = TabM(info, num_classes=1, config=TabMConfig(), lss=True)
    out = lss_model([torch.randn(5, 4)], [], [])
    assert out.ndim == 3, f"LSS ensemble output must keep the parameter axis, got {tuple(out.shape)}"

    plain = TabM(info, num_classes=1, config=TabMConfig(), lss=False)
    assert plain([torch.randn(5, 4)], [], []).ndim == 2

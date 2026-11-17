"""A training set whose size leaves a one-row remainder must still train with BatchNorm."""

import numpy as np
import pytest

from deeptab.configs import MambularConfig, MLPConfig, TrainerConfig
from deeptab.models import MambularRegressor, MLPRegressor

FIT_KWARGS = {"accelerator": "cpu", "logger": False, "enable_progress_bar": False, "enable_model_summary": False}


@pytest.mark.parametrize(
    "model_class, model_config",
    [
        (MLPRegressor, MLPConfig(layer_sizes=[8], batch_norm=True)),
        (
            MambularRegressor,
            MambularConfig(d_model=8, n_layers=1, d_state=4, head_layer_sizes=[8], head_use_batch_norm=True),
        ),
    ],
    ids=["mlp-batch-norm", "mambular-head-batch-norm"],
)
def test_batch_norm_models_fit_with_a_one_row_remainder(model_class, model_config, tmp_path):
    features = np.random.default_rng(0).normal(size=(162, 2))  # 129 training rows with val_size=0.2
    model = model_class(
        model_config=model_config, trainer_config=TrainerConfig(max_epochs=1, batch_size=128), random_state=0
    )
    model.fit(features, features[:, 0], checkpoint_path=str(tmp_path), **FIT_KWARGS)
    assert len(model._data_module.y_train) % 128 == 1
    assert np.isfinite(model.predict(features)).all()

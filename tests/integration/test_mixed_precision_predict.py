"""Predictions after mixed-precision training come back as float32 NumPy arrays.

``predict()`` reuses the fit Trainer, so with ``precision="bf16-mixed"`` the
network outputs are bfloat16, which NumPy cannot represent.
"""

import numpy as np
import pytest

from deeptab.configs import TrainerConfig
from deeptab.models import MLPLSS, MLPClassifier, MLPRegressor

FIT_KWARGS = {
    "accelerator": "cpu",
    "devices": 1,
    "precision": "bf16-mixed",
    "logger": False,
    "enable_progress_bar": False,
    "enable_model_summary": False,
}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(80, 3))
    return features, features[:, 0] * 2


def _fit(model_class, features, targets, tmp_path, **fit_options):
    model = model_class(trainer_config=TrainerConfig(max_epochs=1, batch_size=32), random_state=0)
    return model.fit(features, targets, checkpoint_path=str(tmp_path), **fit_options, **FIT_KWARGS)


def test_regressor_predict_after_bf16_training(data, tmp_path):
    features, targets = data
    predictions = _fit(MLPRegressor, features, targets, tmp_path).predict(features)
    assert predictions.dtype == np.float32
    assert predictions.shape == (len(features),)


def test_classifier_predict_proba_after_bf16_training(data, tmp_path):
    features, targets = data
    model = _fit(MLPClassifier, features, (targets > 0).astype(int), tmp_path)
    probabilities = model.predict_proba(features)
    assert probabilities.dtype == np.float32
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)
    assert set(np.unique(model.predict(features))) <= {0, 1}


def test_lss_predict_after_bf16_training(data, tmp_path):
    features, targets = data
    model = _fit(MLPLSS, features, targets, tmp_path, family="normal")
    assert model.predict(features).dtype == np.float32
    assert model.predict(features, raw=True).dtype == np.float32

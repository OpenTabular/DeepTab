"""Validate pre-provisioned dependencies and portable core fitting."""

import importlib.util
import re

import numpy as np
import pytest

from deeptab.configs import MLPConfig, TrainerConfig
from deeptab.core.optional_deps import require_mlflow, require_structlog, require_tensorboard
from deeptab.models import MLPClassifier


@pytest.mark.parametrize(
    "package,extra,require",
    [
        ("structlog", "logs", require_structlog),
        ("mlflow", "mlflow", require_mlflow),
        ("tensorboard", "tensorboard", require_tensorboard),
    ],
)
def test_declared_dependency_profile(pytestconfig, package, extra, require):
    profile = pytestconfig.getoption("--dependency-profile")
    if profile is None:
        pytest.skip("Select --dependency-profile=core or extras for an isolated installation check")
    available = importlib.util.find_spec(package) is not None
    assert available == (profile == "extras"), f"{package} availability contradicts {profile} profile"
    if profile == "core":
        with pytest.raises(ImportError, match=re.escape(f"pip install 'deeptab[{extra}]'")):
            require()
    else:
        dependency = require()
        assert dependency is not None
        if package == "tensorboard":
            assert callable(dependency)
        else:
            assert dependency.__name__ == package


@pytest.mark.integration
def test_default_core_cpu_fit_without_observability(tmp_path):
    generator = np.random.default_rng(17)
    features = generator.normal(size=(48, 3)).astype(np.float32)
    labels = (features[:, 0] > 0).astype(int)
    model = MLPClassifier(
        model_config=MLPConfig(layer_sizes=[8]),
        trainer_config=TrainerConfig(
            max_epochs=2, batch_size=16, patience=1, lr_patience=1, checkpoint_path=str(tmp_path)
        ),
        random_state=17,
    )
    model.fit(
        features[:32],
        labels[:32],
        X_val=features[32:],
        y_val=labels[32:],
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    predictions = model.predict(features[32:])
    probabilities = model.predict_proba(features[32:])
    assert predictions.shape == labels[32:].shape
    assert set(predictions).issubset(set(model.classes_))
    assert probabilities.shape == (16, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

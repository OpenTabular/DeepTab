import numpy as np
import pandas as pd
import pytest

from deeptab.configs import MLPConfig, TrainerConfig
from deeptab.models import MLPLSS, MLPClassifier


def _classification_data(n_samples=64, n_features=4):
    rng = np.random.default_rng(7)
    X_arr = rng.standard_normal((n_samples, n_features))
    y = (X_arr[:, 0] + X_arr[:, 1] > 0).astype(int)
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(n_features)])
    return X, y


def _regression_data(n_samples=64, n_features=4):
    rng = np.random.default_rng(11)
    X_arr = rng.standard_normal((n_samples, n_features))
    y = X_arr @ rng.standard_normal(n_features) + rng.standard_normal(n_samples) * 0.1
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(n_features)])
    return X, y


def test_inspection_methods_before_fit():
    model = MLPClassifier(
        model_config=MLPConfig(layer_sizes=[16]),
        trainer_config=TrainerConfig(max_epochs=1, batch_size=16, patience=1),
    )

    description = model.describe()
    runtime = model.runtime_info()
    summary = model.summary()

    assert description["estimator"] == "MLPClassifier"
    assert description["built"] is False
    assert description["fitted"] is False
    assert description["parameters"] is None
    assert runtime["built"] is False
    assert runtime["batch_size"] == 16
    assert "MLPClassifier summary" in summary

    with pytest.raises(ValueError, match="built or fitted"):
        model.parameter_table()


def test_inspection_methods_after_classifier_fit():
    X, y = _classification_data()
    model = MLPClassifier(
        model_config=MLPConfig(layer_sizes=[16], dropout=0.0),
        trainer_config=TrainerConfig(max_epochs=1, batch_size=16, patience=1),
        random_state=7,
    )
    model.fit(X, y, enable_progress_bar=False, logger=False, enable_model_summary=False)

    description = model.describe()
    runtime = model.runtime_info()
    table = model.parameter_table()
    trainable_table = model.parameter_table(trainable_only=True)
    summary = model.summary()

    assert description["built"] is True
    assert description["fitted"] is True
    assert description["task"] == "classification"
    assert description["feature_counts"] == {
        "numerical": 4,
        "categorical": 0,
        "embedding": 0,
        "total": 4,
    }
    assert description["parameters"]["total"] == model.get_number_of_params(requires_grad=False)
    assert description["parameters"]["trainable"] == model.get_number_of_params(requires_grad=True)

    assert not table.empty
    assert {"name", "module", "shape", "num_params", "trainable", "dtype", "device"}.issubset(table.columns)
    assert int(table["num_params"].sum()) == model.get_number_of_params(requires_grad=False)
    assert trainable_table["trainable"].all()

    assert runtime["built"] is True
    assert runtime["fitted"] is True
    assert runtime["device"] is not None
    assert runtime["dtype"] == "float32"
    assert runtime["batch_size"] == 16
    assert runtime["optimizer_type"] == "Adam"
    assert "Parameters:" in summary
    assert "Device:" in summary


def test_inspection_methods_after_lss_fit():
    X, y = _regression_data()
    model = MLPLSS(
        model_config=MLPConfig(layer_sizes=[16], dropout=0.0),
        trainer_config=TrainerConfig(max_epochs=1, batch_size=16, patience=1),
        random_state=11,
    )
    model.fit(X, y, family="normal", enable_progress_bar=False, logger=False, enable_model_summary=False)

    description = model.describe()
    runtime = model.runtime_info()

    assert description["task"] == "distributional_regression"
    assert description["family"] == "normal"
    assert description["parameters"]["total"] == model.get_number_of_params(requires_grad=False)
    assert not model.parameter_table().empty
    assert runtime["batch_size"] == 16

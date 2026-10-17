"""Collection rules for workflow categories and dependency profiles."""

import importlib
import importlib.util
import platform
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--dependency-profile",
        choices=("core", "extras", "tabr"),
        default=None,
        help="Verify the externally provisioned core, logs/tracking, or native TabR installation.",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    root = Path(__file__).parent
    categories = {
        root / "integration": "integration",
        root / "contracts": "contract",
        root / "learnability": "learnability",
        root / "lifecycle": "lifecycle",
        root / "metrics" / "oracles": "metric_oracle",
        root / "boundaries": "boundary",
        root / "environments": "environment",
    }
    for item in items:
        for directory, marker in categories.items():
            if directory in item.path.parents:
                item.add_marker(getattr(pytest.mark, marker))


@pytest.fixture
def native_tabr(pytestconfig):
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        pytest.skip("Native FAISS TabR is unsupported on Darwin arm64; run the same cases on Linux/Windows.")
    for package in ("delu", "faiss"):
        if importlib.util.find_spec(package) is None:
            if pytestconfig.getoption("--dependency-profile") == "tabr":
                pytest.fail(f"The tabr dependency profile requires {package}")
            pytest.skip("Native TabR requires pip install 'deeptab[tabr]'")
        importlib.import_module(package)


@pytest.fixture
def easy_problem():
    rng = np.random.default_rng(17)
    signal = np.tile(np.array([-1.0, 0.0, 1.0]), 32) + rng.normal(0, 0.05, 96)
    groups = np.digitize(signal, [-0.5, 0.5])
    features = pd.DataFrame(
        {
            "signal": signal.astype(np.float32),
            "other": rng.normal(size=96).astype(np.float32),
            "category": np.asarray(["left", "middle", "right"])[groups],
        }
    )
    return (
        features.iloc[:63],
        features.iloc[63:],
        groups[:63] * 20 + 10,
        groups[63:] * 20 + 10,
        signal[:63],
        signal[63:],
    )


@pytest.fixture
def stable_model_factory(tmp_path, request):
    import deeptab.configs as configs
    import deeptab.models as models
    from deeptab.configs import PreprocessingConfig, TrainerConfig

    overrides = {
        "d_model": 16,
        "n_layers": 1,
        "num_layers": 1,
        "n_heads": 2,
        "layer_sizes": [32, 16],
        "layer_dim": 8,
        "depth": 3,
        "d_state": 4,
        "d_conv": 4,
        "head_dropout": 0.0,
        "dropout": 0.0,
        "attn_dropout": 0.0,
        "ff_dropout": 0.0,
        "rnn_dropout": 0.0,
        "transformer_dim_feedforward": 32,
        "dim_feedforward": 32,
        "ensemble_size": 3,
        "num_blocks": 1,
        "min_depth": 2,
        "max_depth": 3,
        "n_ensembles": 4,
        "d_main": 16,
        "context_size": 4,
        "encoder_n_blocks": 0,
        "predictor_n_blocks": 1,
        "dropout0": 0.0,
        "dropout1": 0.0,
        "context_dropout": 0.0,
        "candidate_encoding_batch_size": 16,
    }

    def make_model(architecture, task, epochs=40, model_options=None, preprocessing_options=None):
        if architecture == "TabR":
            request.getfixturevalue("native_tabr")
        config_class = getattr(configs, architecture + "Config")
        names = {field.name for field in fields(config_class)}
        options = {name: value for name, value in overrides.items() if name in names}
        options.update(model_options or {})
        config = config_class(**options)
        return getattr(
            models, architecture + {"classification": "Classifier", "regression": "Regressor", "lss": "LSS"}[task]
        )(
            model_config=config,
            preprocessing_config=PreprocessingConfig(
                **{"numerical_method": "standardization", "categorical_method": "int", **(preprocessing_options or {})}
            ),
            trainer_config=TrainerConfig(
                max_epochs=epochs,
                batch_size=16,
                lr=0.01,
                weight_decay=0,
                patience=epochs + 1,
                scheduler_type=None,
                checkpoint_path=str(tmp_path),
            ),
            random_state=17,
        )

    return make_model

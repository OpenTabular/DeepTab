import importlib
import inspect
import os

import pytest
import torch

from deeptab import architectures
from deeptab.core import BaseModel

# Paths for models and configs
MODEL_MODULE_PATH = "deeptab.architectures"
_CONFIG_SEARCH_PATHS = [
    "deeptab.configs.models",
    "deeptab.configs.experimental",
]
EXCLUDED_CLASSES = {"TabR"}

# Discover all models (stable + experimental)
model_classes = []
_arch_root = os.path.dirname(architectures.__file__)
_scan = [(MODEL_MODULE_PATH, _arch_root), (MODEL_MODULE_PATH + ".experimental", _arch_root + "/experimental")]
for _mod_prefix, _dir in _scan:
    for filename in os.listdir(_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = f"{_mod_prefix}.{filename[:-3]}"
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseModel) and obj is not BaseModel and obj.__name__ not in EXCLUDED_CLASSES:
                    model_classes.append(obj)


def get_model_config(model_class):
    """Dynamically load the correct config class for each model."""
    model_name = model_class.__name__  # e.g., "Mambular"
    config_class_name = f"{model_name}Config"  # e.g., "MambularConfig"

    for base_path in _CONFIG_SEARCH_PATHS:
        try:
            config_module = importlib.import_module(f"{base_path}.{model_name.lower()}_config")
            config_class = getattr(config_module, config_class_name)
            return config_class()
        except (ModuleNotFoundError, AttributeError):
            continue

    pytest.fail(f"Could not find or instantiate config {config_class_name} for {model_name}")


@pytest.mark.parametrize("num_classes", [1, 3])
@pytest.mark.parametrize("with_embeddings", [False, True])
def test_tabtransformer_categorical_only_forward_and_backward(num_classes, with_embeddings):
    from deeptab.architectures.tabtransformer import TabTransformer
    from deeptab.configs import TabTransformerConfig

    categorical_info = {"city": {"dimension": 1, "categories": 2, "preprocessing": "int"}}
    embedding_info = {"text": {"dimension": 3}} if with_embeddings else {}
    model = TabTransformer(
        feature_information=({}, categorical_info, embedding_info),
        num_classes=num_classes,
        config=TabTransformerConfig(d_model=8, n_heads=2, n_layers=1),
    )
    categories = [torch.tensor([[0], [1], [0], [1]])]
    embeddings = [torch.ones(4, 3)] if with_embeddings else []
    predictions = model([], categories, embeddings)

    assert predictions.shape == (4, num_classes)
    assert torch.isfinite(predictions).all()
    predictions.sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.smoke
@pytest.mark.parametrize("model_class", model_classes)
def test_model_inherits_base_model(model_class):
    """Test that each model correctly inherits from BaseModel."""
    assert issubclass(model_class, BaseModel), f"{model_class.__name__} should inherit from BaseModel."


@pytest.mark.parametrize("model_class", model_classes)
def test_model_has_forward_method(model_class):
    """Test that each model has a forward method with *data."""
    assert hasattr(model_class, "forward"), f"{model_class.__name__} is missing a forward method."

    sig = inspect.signature(model_class.forward)
    assert any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()), (
        f"{model_class.__name__}.forward should have *data argument."
    )


@pytest.mark.parametrize("model_class", model_classes)
def test_model_takes_config(model_class):
    """Test that each model accepts a config argument."""
    sig = inspect.signature(model_class.__init__)
    assert "config" in sig.parameters, f"{model_class.__name__} should accept a 'config' parameter."


@pytest.mark.parametrize("model_class", model_classes)
def test_model_has_num_classes(model_class):
    """Test that each model accepts a num_classes argument."""
    sig = inspect.signature(model_class.__init__)
    assert "num_classes" in sig.parameters, f"{model_class.__name__} should accept a 'num_classes' parameter."


@pytest.mark.parametrize("model_class", model_classes)
def test_model_calls_super_init(model_class):
    """Test that each model calls super().__init__(config=config, **kwargs)."""
    source = inspect.getsource(model_class.__init__)
    assert "super().__init__(config=config" in source, (
        f"{model_class.__name__} should call super().__init__(config=config, **kwargs)."
    )


@pytest.mark.parametrize("model_class", model_classes)
def test_model_initialization(model_class):
    """Test that each model can be initialized with its correct config."""
    config = get_model_config(model_class)
    feature_info = (
        {
            "A": {
                "preprocessing": "imputer -> check_positive -> box-cox",
                "dimension": 1,
                "categories": None,
            }
        },
        {
            "sibsp": {
                "preprocessing": "imputer -> continuous_ordinal",
                "dimension": 1,
                "categories": 8,
            }
        },
        {},
    )  # Mock feature info

    try:
        model = model_class(feature_information=feature_info, num_classes=3, config=config)
    except Exception as e:
        pytest.fail(f"Failed to initialize {model_class.__name__}: {e}")


@pytest.mark.parametrize("model_class", model_classes)
def test_model_defines_key_attributes(model_class):
    """Test that each model defines expected attributes like returns_ensemble"""
    config = get_model_config(model_class)
    feature_info = (
        {
            "A": {
                "preprocessing": "imputer -> check_positive -> box-cox",
                "dimension": 1,
                "categories": None,
            }
        },
        {
            "sibsp": {
                "preprocessing": "imputer -> continuous_ordinal",
                "dimension": 1,
                "categories": 8,
            }
        },
        {},
    )  # Mock feature info

    try:
        model = model_class(feature_information=feature_info, num_classes=3, config=config)
    except TypeError as e:
        pytest.fail(f"Failed to initialize {model_class.__name__}: {e}")

    expected_attrs = ["returns_ensemble"]
    for attr in expected_attrs:
        assert hasattr(model, attr), f"{model_class.__name__} should define '{attr}'."

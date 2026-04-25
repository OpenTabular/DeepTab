import dataclasses
import importlib
import inspect
import os
import typing

import pytest

from deeptab.configs.base_config import BaseConfig  # Ensure correct path

CONFIG_MODULE_PATH = "deeptab.configs"
config_classes = []

# Discover all config classes in deeptab/configs/
for filename in os.listdir(os.path.dirname(__file__) + "/../deeptab/configs"):
    if filename.endswith(".py") and filename != "base_config.py" and not filename.startswith("__"):
        module_name = f"{CONFIG_MODULE_PATH}.{filename[:-3]}"
        module = importlib.import_module(module_name)

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseConfig) and obj is not BaseConfig:
                config_classes.append(obj)


@pytest.mark.parametrize("config_class", config_classes)
def test_config_inherits_baseconfig(config_class):
    """Test that each config class correctly inherits from BaseConfig."""
    assert issubclass(config_class, BaseConfig), f"{config_class.__name__} should inherit from BaseConfig."


@pytest.mark.parametrize("config_class", config_classes)
def test_config_instantiation(config_class):
    """Test that each config class can be instantiated without errors."""
    try:
        config = config_class()
    except Exception as e:
        pytest.fail(f"Failed to instantiate {config_class.__name__}: {e}")


@pytest.mark.parametrize("config_class", config_classes)
def test_config_has_expected_attributes(config_class):
    """Test that each config has all required attributes from BaseConfig."""
    base_attrs = {field.name for field in dataclasses.fields(BaseConfig)}
    config_attrs = {field.name for field in dataclasses.fields(config_class)}

    missing_attrs = base_attrs - config_attrs
    assert not missing_attrs, f"{config_class.__name__} is missing attributes: {missing_attrs}"


@pytest.mark.parametrize("config_class", config_classes)
def test_config_default_values(config_class):
    """Ensure that each config class has default values assigned correctly."""
    config = config_class()

    for field in dataclasses.fields(config_class):
        attr = field.name
        expected_type = field.type

        assert hasattr(config, attr), f"{config_class.__name__} is missing attribute '{attr}'."

        value = getattr(config, attr)

        # Handle generic types properly
        origin = typing.get_origin(expected_type)

        if origin is typing.Literal:
            # If the field is a Literal, ensure the value is one of the allowed options
            allowed_values = typing.get_args(expected_type)
            assert (
                value in allowed_values
            ), f"{config_class.__name__}.{attr} has incorrect value: expected one of {allowed_values}, got {value}"
        elif origin is typing.Union:
            # For Union types (e.g., Optional[str]), check if value matches any type in the union
            allowed_types = typing.get_args(expected_type)
            assert any(
                isinstance(value, t) for t in allowed_types
            ), f"{config_class.__name__}.{attr} has incorrect type: expected one of {allowed_types}, got {type(value)}"
        elif origin is not None:
            # If it's another generic type (e.g., list[str]), check against the base type
            assert (
                isinstance(value, origin) or value is None
            ), f"{config_class.__name__}.{attr} has incorrect type: expected {expected_type}, got {type(value)}"
        else:
            # Standard type check
            assert (
                isinstance(value, expected_type) or value is None
            ), f"{config_class.__name__}.{attr} has incorrect type: expected {expected_type}, got {type(value)}"


@pytest.mark.parametrize("config_class", config_classes)
def test_config_allows_updates(config_class):
    """Ensure that config values can be updated and remain type-consistent."""
    config = config_class()

    update_values = {
        "lr": 0.01,
        "d_model": 128,
        "embedding_type": "plr",
        "activation": lambda x: x,  # Function update
    }

    for attr, new_value in update_values.items():
        if hasattr(config, attr):
            setattr(config, attr, new_value)
            assert getattr(config, attr) == new_value, f"{config_class.__name__}.{attr} did not update correctly."

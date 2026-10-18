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


@pytest.mark.parametrize("architecture_name", ["TabM", "Trompt"])
@pytest.mark.parametrize("lss", [False, True])
@pytest.mark.parametrize("output_dim", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 5])
def test_ensemble_output_preserves_lss_parameter_axis(architecture_name, lss, output_dim, batch_size):
    model_class = next(model_class for model_class in model_classes if model_class.__name__ == architecture_name)
    config = get_model_config(model_class)
    config.d_model = 8
    if architecture_name == "TabM":
        config.layer_sizes = [8, 4]
        config.ensemble_size = 3
        config.dropout = 0.0
    else:
        config.n_cycles = 3
        config.P = 4
    feature_information = ({f"feature_{index}": {"dimension": 1} for index in range(3)}, {}, {})
    model = model_class(feature_information, num_classes=output_dim, config=config, lss=lss).eval()
    predictions = model([torch.randn(batch_size, 1) for _ in range(3)], [], [])

    expected_shape = (batch_size, 3, output_dim) if lss or output_dim > 1 else (batch_size, 3)
    assert predictions.shape == expected_shape
    assert torch.isfinite(predictions).all()
    predictions.sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("architecture_name", ["MLP", "TabM", "Tangos"])
@pytest.mark.parametrize("use_glu", [False, True])
@pytest.mark.parametrize("batch_norm", [False, True])
def test_mlp_options_preserve_hidden_widths(architecture_name, use_glu, batch_norm):
    model_class = next(model_class for model_class in model_classes if model_class.__name__ == architecture_name)
    config = get_model_config(model_class)
    config.layer_sizes = [7, 5]
    config.use_embeddings = False
    config.use_glu = use_glu
    config.batch_norm = batch_norm
    config.dropout = 0.0
    if architecture_name == "TabM":
        config.ensemble_size = 3
    else:
        config.layer_norm = True
    model = model_class(({"value": {"dimension": 2}}, {}, {}), num_classes=3, config=config)
    predictions = model([torch.randn(4, 2)], [], [])
    expected_shape = (4, 3, 3) if architecture_name == "TabM" else (4, 3)
    assert predictions.shape == expected_shape
    assert torch.isfinite(predictions).all()
    predictions.sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("norm", ["LayerNorm", "RMSNorm", "BatchNorm", "LearnableLayerScaling"])
@pytest.mark.parametrize("average_ensembles", [False, True])
def test_tabm_normalization_uses_hidden_width(norm, average_ensembles):
    from deeptab.architectures.tabm import TabM
    from deeptab.configs import TabMConfig

    model = TabM(
        ({"value": {"dimension": 2}}, {}, {}),
        config=TabMConfig(
            use_embeddings=False,
            layer_sizes=[7, 5],
            d_model=16,
            ensemble_size=3,
            norm=norm,
            average_ensembles=average_ensembles,
            dropout=0.0,
        ),
    )
    predictions = model([torch.randn(4, 2)], [], [])
    assert predictions.shape == ((4, 1) if average_ensembles else (4, 3))
    assert torch.isfinite(predictions).all()
    predictions.sum().backward()


@pytest.mark.parametrize("norm", ["LayerNorm", "RMSNorm", "BatchNorm", "LearnableLayerScaling"])
def test_tabm_normalization_keeps_ensemble_members_independent(norm):
    from deeptab.architectures.tabm import TabM
    from deeptab.configs import TabMConfig
    from deeptab.nn.blocks.common import LinearBatchEnsembleLayer

    torch.manual_seed(0)
    model = TabM(
        ({"value": {"dimension": 2}}, {}, {}),
        config=TabMConfig(use_embeddings=False, layer_sizes=[7, 5], ensemble_size=3, norm=norm, dropout=0.0),
    ).eval()
    inputs = [torch.randn(4, 2)]
    baseline = model(inputs, [], [])
    first_layer = model.layers[0]
    assert isinstance(first_layer, LinearBatchEnsembleLayer)
    assert first_layer.r is not None
    with torch.no_grad():
        first_layer.r[0].copy_(torch.randn_like(first_layer.r[0]))
    perturbed = model(inputs, [], [])

    assert not torch.allclose(perturbed[:, 0], baseline[:, 0])
    torch.testing.assert_close(perturbed[:, 1:], baseline[:, 1:])


@pytest.mark.parametrize("norm", ["InstanceNorm", "GroupNorm"])
def test_tabm_rejects_normalization_across_ensemble_members(norm):
    from deeptab.architectures.tabm import TabM
    from deeptab.configs import TabMConfig
    from deeptab.core.exceptions import InvalidParamError

    with pytest.raises(InvalidParamError, match="ensemble members independent"):
        TabM(({"value": {"dimension": 2}}, {}, {}), config=TabMConfig(use_embeddings=False, norm=norm))


@pytest.mark.parametrize("norm_first", [False, True])
def test_fttransformer_applies_final_normalization_once(norm_first):
    from deeptab.architectures.ft_transformer import FTTransformer
    from deeptab.configs import FTTransformerConfig

    model = FTTransformer(
        ({"value": {"dimension": 2}}, {}, {}),
        config=FTTransformerConfig(d_model=8, n_heads=2, n_layers=1, norm="LayerNorm", norm_first=norm_first),
    )
    assert model.norm_f is not None
    assert model.encoder.norm is None
    calls: list[torch.Size] = []
    hook = model.norm_f.register_forward_hook(lambda module, args, output: calls.append(output.shape))
    model([torch.randn(4, 2)], [], []).sum().backward()
    hook.remove()
    assert calls == [torch.Size([4, 8])]


@pytest.mark.parametrize("architecture_name", ["Mambular", "MambAttention"])
def test_shuffled_embeddings_permutation_is_checkpointed(architecture_name):
    model_class = next(model_class for model_class in model_classes if model_class.__name__ == architecture_name)
    config = get_model_config(model_class)
    config.d_model = 8
    config.n_layers = 1
    config.d_state = 4
    config.n_heads = 2
    config.shuffle_embeddings = True
    feature_info = ({f"value_{index}": {"dimension": 1} for index in range(4)}, {}, {})
    model = model_class(feature_info, config=config)
    assert "perm" in dict(model.named_buffers())
    state = model.state_dict()
    assert "perm" in state
    restored = model_class(feature_info, config=config)
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.perm, model.perm, rtol=0, atol=0)
    model.eval()
    restored.eval()
    features = [torch.randn(4, 1) for _ in range(4)]
    torch.testing.assert_close(restored(features, [], []), model(features, [], []), rtol=0, atol=0)


@pytest.mark.parametrize("model_type", ["mLSTM", "sLSTM"])
def test_tabularnn_recurrent_options_preserve_rows(model_type):
    from deeptab.architectures.tabularnn import TabulaRNN
    from deeptab.configs import TabulaRNNConfig

    config = TabulaRNNConfig(d_model=8, dim_feedforward=12, n_layers=2, model_type=model_type)
    features = ({f"value_{index}": {"dimension": 1} for index in range(3)}, {}, {})
    model = TabulaRNN(features, num_classes=3, config=config).eval()
    values = [torch.randn(4, 1) for _ in range(3)]
    output = model(values, [], [])
    separate = torch.cat([model([value[index : index + 1] for value in values], [], []) for index in range(4)])
    assert output.shape == (4, 3)
    torch.testing.assert_close(output, separate)
    output.square().sum().backward()


def test_trompt_initial_prompts_are_initialized_under_nan_allocation(monkeypatch):
    from deeptab.architectures.experimental.trompt import Trompt
    from deeptab.configs import TromptConfig

    original_empty = torch.empty

    def nan_empty(*args, **kwargs):
        output = original_empty(*args, **kwargs)
        if output.is_floating_point():
            output.fill_(float("nan"))
        return output

    monkeypatch.setattr(torch, "empty", nan_empty)
    config = TromptConfig(d_model=8, P=8, n_cycles=1)
    feature_info = ({"value": {"dimension": 2}}, {}, {})
    torch.manual_seed(42)
    model = Trompt(feature_info, config=config)
    assert torch.isfinite(model.init_rec).all()
    torch.manual_seed(42)
    repeated = Trompt(feature_info, config=config)
    torch.testing.assert_close(model.init_rec, repeated.init_rec, rtol=0, atol=0)
    predictions = model([torch.randn(4, 2)], [], [])
    assert torch.isfinite(predictions).all()
    predictions.sum().backward()
    assert model.init_rec.grad is not None
    assert torch.isfinite(model.init_rec.grad).all()


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

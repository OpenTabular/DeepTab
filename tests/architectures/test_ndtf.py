"""Feature projection equivalence and accelerator training checks for NDTF."""

from copy import deepcopy

import pytest
import torch

from deeptab.architectures.ndtf import NDTF
from deeptab.configs import NDTFConfig


def _make_model(width):
    num_info = {"feature": {"preprocessing": "none", "dimension": width, "categories": None}}
    return NDTF(
        feature_information=(num_info, {}, {}),
        num_classes=3,
        config=NDTFConfig(n_ensembles=4),
        input_dimensions=[width, max(1, width // 2), max(1, width // 3), width],
        depths=[4, 5, 6, 8],
        temperatures=[0.7, 0.9, 1.1, 1.3],
    )


@pytest.mark.parametrize("width", [1, 6, 42])
@pytest.mark.parametrize("batch_size", [1, 64])
def test_feature_projection_matches_convolution_outputs_and_gradients(width, batch_size):
    """Use float64 so reduction-order rounding does not dominate the comparison."""
    torch.manual_seed(42)
    model = _make_model(width).double()
    convolution = deepcopy(model.conv_layer)
    inputs = torch.randn(batch_size, width, dtype=torch.float64, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()

    actual = model._feature_interactions(inputs)
    expected = convolution(reference_inputs.unsqueeze(-1)).squeeze(1)

    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
    torch.testing.assert_close(model.conv_layer.weight.grad, convolution.weight.grad)
    assert model.conv_layer.bias is not None
    assert convolution.bias is not None
    torch.testing.assert_close(model.conv_layer.bias.grad, convolution.bias.grad)


def test_one_dimensional_input_builds_every_tree_on_that_input():
    num_info = {"feature": {"preprocessing": "none", "dimension": 1, "categories": None}}
    model = NDTF(feature_information=(num_info, {}, {}), num_classes=1, config=NDTFConfig(n_ensembles=3))
    assert model.input_dimensions == [1, 1, 1]
    assert model([torch.randn(5, 1)], [], []).shape == (5, 1)


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("with_penalty", [False, True])
def test_forest_backward_has_finite_gradients(device, with_penalty):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is not available")
    torch.manual_seed(42)
    model = _make_model(42).to(device)
    inputs = [torch.randn(64, 42, device=device)]

    if with_penalty:
        predictions, penalty = model.penalty_forward(inputs, [], [])
        loss = predictions.square().mean() + penalty
    else:
        predictions = model(inputs, [], [])
        loss = predictions.square().mean()
    loss.backward()

    assert predictions.shape == (64, 3)
    assert torch.isfinite(loss)
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

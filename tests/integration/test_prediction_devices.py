"""Tests for prediction devices."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from deeptab.architectures.mlp import MLP
from deeptab.models import MLPRegressor

RANDOM_STATE = 0

FIT_KWARGS: dict[str, Any] = {"max_epochs": 1, "batch_size": 32}

N = 120

N_FEATURES = 4


def _make_reg_data():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N, N_FEATURES))
    y = rng.standard_normal(N)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)]), y  # type: ignore[call-overload]


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("shape", [(5, 4), (2, 5, 4)])
@pytest.mark.parametrize("bias", [False, True])
def test_mlp_linear_matches_cpu_reference_without_fused_mps_bias(device, shape, bias, monkeypatch):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is not available")
    torch.manual_seed(42)
    reference = torch.nn.Linear(4, 3, bias=bias).double()
    dtype = torch.float32 if device == "mps" else torch.float64
    layer = deepcopy(reference).to(device=device, dtype=dtype)
    inputs = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    reference_inputs = inputs.detach().cpu().double().requires_grad_()
    expected = reference(reference_inputs)
    original_linear = torch.nn.functional.linear
    fused_bias_calls = []

    def record_linear(inputs, weight, bias=None):
        fused_bias_calls.append(bias is not None)
        return original_linear(inputs, weight, bias)

    monkeypatch.setattr(torch.nn.functional, "linear", record_linear)
    actual = MLP._apply_linear(layer, inputs)
    assert fused_bias_calls == [bias and device != "mps"]

    rtol = 1e-5 if device == "mps" else None
    atol = 1e-6 if device == "mps" else None
    torch.testing.assert_close(actual.detach().cpu().double(), expected.detach(), rtol=rtol, atol=atol)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream.cpu().double())
    assert inputs.grad is not None and reference_inputs.grad is not None
    torch.testing.assert_close(inputs.grad.cpu().double(), reference_inputs.grad, rtol=rtol, atol=atol)
    for parameter, reference_parameter in zip(layer.parameters(), reference.parameters(), strict=True):
        assert parameter.grad is not None and reference_parameter.grad is not None
        torch.testing.assert_close(parameter.grad.cpu().double(), reference_parameter.grad, rtol=rtol, atol=atol)


class TestPredictDeviceOverride:
    def test_explicit_cpu_device_matches_default(self):
        X, y = _make_reg_data()
        reg = MLPRegressor()
        reg.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)

        default_preds = reg.predict(X)
        cpu_preds = reg.predict(X, device="cpu")
        np.testing.assert_array_almost_equal(default_preds, cpu_preds, decimal=4)

    def test_invalid_device_string_raises(self):
        X, y = _make_reg_data()
        reg = MLPRegressor()
        reg.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)

        with pytest.raises(RuntimeError, match="Invalid device string"):
            reg.predict(X, device="not-a-real-device")

"""Regression tests for AutoInt's use_cls and fprenorm config options.

- use_cls=True previously crashed: the tabular head's input size was computed
  from the pre-CLS n_inputs, so appending a CLS token to the sequence made the
  head's expected input width wrong.
- fprenorm was checked via a misspelled `prenorm` attribute lookup, so setting
  fprenorm never actually enabled the final normalization layer.
"""

from __future__ import annotations

import pytest
import torch

from deeptab.architectures.autoint import AutoInt
from deeptab.configs import AutoIntConfig

NUM_INFO = {"f0": {"preprocessing": "", "dimension": 1, "categories": None}}
CAT_INFO = {"c0": {"preprocessing": "", "dimension": 1, "categories": 5}}


def _cat_tensor(batch_size, num_categories=5):
    return torch.randint(0, num_categories, (batch_size,))


class TestAutoIntUseClsAndFprenorm:
    def _build(self, **cfg_kwargs):
        config = AutoIntConfig(d_model=16, n_layers=1, n_heads=2, **cfg_kwargs)
        return AutoInt(feature_information=(NUM_INFO, CAT_INFO, {}), num_classes=3, config=config)

    def test_use_cls_does_not_crash_and_output_shape_is_correct(self):
        model = self._build(use_cls=True)
        model.eval()
        batch_size = 4
        data = ([torch.randn(batch_size, 1)], [_cat_tensor(batch_size)], [])
        out = model(*data)
        assert out.shape == (batch_size, 3)

    def test_use_cls_false_still_works(self):
        model = self._build(use_cls=False)
        batch_size = 4
        data = ([torch.randn(batch_size, 1)], [_cat_tensor(batch_size)], [])
        out = model(*data)
        assert out.shape == (batch_size, 3)

    def test_fprenorm_true_adds_final_norm_and_runs(self):
        model = self._build(fprenorm=True)
        assert model.last_norm is not None
        batch_size = 4
        data = ([torch.randn(batch_size, 1)], [_cat_tensor(batch_size)], [])
        out = model(*data)
        assert out.shape == (batch_size, 3)

    def test_fprenorm_false_has_no_final_norm(self):
        model = self._build(fprenorm=False)
        assert model.last_norm is None


class TestAutoIntInteractionOptions:
    @pytest.mark.parametrize("sharing", ["layerwise", "headwise", "key-value"])
    @pytest.mark.parametrize("use_cls", [False, True])
    @pytest.mark.parametrize("fprenorm", [False, True])
    def test_compression_is_used_and_receives_gradients(self, sharing, use_cls, fprenorm):
        config = AutoIntConfig(
            d_model=8,
            n_layers=2,
            n_heads=2,
            kv_compression=0.5,
            kv_compression_sharing=sharing,
            use_cls=use_cls,
            fprenorm=fprenorm,
            attn_dropout=0.0,
        )
        feature_info = ({f"feature_{index}": {"dimension": 1} for index in range(6)}, {}, {})
        model = AutoInt(feature_info, num_classes=3, config=config)
        lengths = []
        hooks = []
        for layer in model.layers:
            assert isinstance(layer, torch.nn.ModuleDict)
            hooks.append(
                layer["attention"].register_forward_pre_hook(lambda module, args: lengths.append(args[1].shape[1]))
            )
        output = model([torch.randn(4, 1) for _ in range(6)], [], [])
        assert output.shape == (4, 3)
        assert lengths == [3, 3]
        output.square().sum().backward()
        parameters = [parameter for name, parameter in model.named_parameters() if "compression" in name]
        assert parameters
        for parameter in parameters:
            gradient = parameter.grad
            assert gradient is not None
            assert torch.isfinite(gradient).all()
            assert gradient.abs().sum() > 0
        for hook in hooks:
            hook.remove()

    def test_default_attends_over_every_feature_token(self):
        feature_info = ({f"feature_{index}": {"dimension": 1} for index in range(6)}, {}, {})
        model = AutoInt(feature_info, config=AutoIntConfig(d_model=8, n_layers=2, n_heads=2))
        lengths = []
        hooks = []
        for layer in model.layers:
            assert isinstance(layer, torch.nn.ModuleDict)
            hooks.append(
                layer["attention"].register_forward_pre_hook(lambda module, args: lengths.append(args[1].shape[1]))
            )
        model([torch.randn(4, 1) for _ in range(6)], [], [])
        for hook in hooks:
            hook.remove()
        assert lengths == [6, 6]
        assert not [name for name, _ in model.named_parameters() if "compression" in name]

    def test_relu_is_applied_after_interaction(self):
        model = AutoInt((NUM_INFO, CAT_INFO, {}), config=AutoIntConfig(d_model=8, n_layers=1, n_heads=2)).eval()
        head_inputs = []
        hook = model.head.register_forward_pre_hook(lambda module, args: head_inputs.append(args[0]))
        model([torch.randn(4, 1)], [_cat_tensor(4)], [])
        hook.remove()
        assert (head_inputs[0] >= 0).all()

    @pytest.mark.parametrize("ratio", [0.0, -0.5, 1.5])
    def test_invalid_compression_ratio_is_rejected(self, ratio):
        with pytest.raises(ValueError, match="kv_compression"):
            AutoInt(
                (NUM_INFO, CAT_INFO, {}), config=AutoIntConfig(kv_compression=ratio, kv_compression_sharing="layerwise")
            )

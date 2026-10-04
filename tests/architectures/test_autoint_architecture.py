"""Regression tests for AutoInt's use_cls and fprenorm config options.

- use_cls=True previously crashed: the tabular head's input size was computed
  from the pre-CLS n_inputs, so appending a CLS token to the sequence made the
  head's expected input width wrong.
- fprenorm was checked via a misspelled `prenorm` attribute lookup, so setting
  fprenorm never actually enabled the final normalization layer.
"""

from __future__ import annotations

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

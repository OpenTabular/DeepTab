"""Tests for deeptab.core.base_model.BaseModel.

Covers:
- save_model/load_model no longer print unconditionally to stdout; they log
  through the standard logging module instead, so a fit at ObservabilityConfig's
  default (no console handler attached) produces no output.
- The status message is still observable via `caplog` when logging is configured.
- initialize_pooling_layers' hidden_size resolution: it previously read
  config.dim_feedforward directly, which only exists on TabulaRNNConfig and
  crashed every other architecture using the "learned_flatten"/"attention"/
  "gated" pooling methods.
"""

from __future__ import annotations

import logging
import os
import tempfile

import pytest
import torch

from deeptab.configs import FTTransformerConfig, TabulaRNNConfig
from deeptab.core.base_model import BaseModel


class _TinyModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)


class TestSaveModelLoadModelLogging:
    def test_save_model_does_not_print_to_stdout(self, capsys):
        model = _TinyModel()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "model.pt")
            model.save_model(path)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_load_model_does_not_print_to_stdout(self, capsys):
        model = _TinyModel()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "model.pt")
            model.save_model(path)
            capsys.readouterr()  # discard save_model's own capture
            loaded = _TinyModel()
            loaded.load_model(path)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_save_model_logs_status_message(self, caplog):
        model = _TinyModel()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "model.pt")
            with caplog.at_level(logging.INFO, logger="deeptab.core.base_model"):
                model.save_model(path)
        assert any("Model parameters saved to" in record.message for record in caplog.records)

    def test_load_model_logs_status_message(self, caplog):
        model = _TinyModel()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "model.pt")
            model.save_model(path)
            loaded = _TinyModel()
            with caplog.at_level(logging.INFO, logger="deeptab.core.base_model"):
                loaded.load_model(path)
        assert any("Model parameters loaded from" in record.message for record in caplog.records)


NUM_INFO = {"f0": {"preprocessing": "", "dimension": 1, "categories": None}}
CAT_INFO = {"c0": {"preprocessing": "", "dimension": 1, "categories": 5}}


class TestPoolingHiddenSizeFix:
    """initialize_pooling_layers must resolve a valid hidden_size for every
    architecture, not just the one (TabulaRNN) that defines dim_feedforward."""

    @pytest.mark.parametrize("pooling_method", ["learned_flatten", "attention", "gated"])
    def test_ft_transformer_does_not_crash(self, pooling_method):
        from deeptab.architectures.ft_transformer import FTTransformer

        config = FTTransformerConfig(d_model=16, n_layers=1, n_heads=2, pooling_method=pooling_method)
        model = FTTransformer(feature_information=(NUM_INFO, {}, {}), num_classes=2, config=config)
        out = model([torch.randn(4, 1)], [], [])
        assert out.shape == (4, 2)

    def test_tabularnn_n_inputs_is_scalar_and_learned_flatten_works(self):
        from deeptab.architectures.tabularnn import TabulaRNN

        config = TabulaRNNConfig(d_model=16, dim_feedforward=8, n_layers=1, pooling_method="learned_flatten")
        model = TabulaRNN(feature_information=(NUM_INFO, CAT_INFO, {}), num_classes=2, config=config)
        out = model([torch.randn(4, 1)], [torch.randint(0, 5, (4,))], [])
        assert out.shape == (4, 2)

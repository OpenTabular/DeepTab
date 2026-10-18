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


class TestLogParameters:
    @pytest.mark.parametrize("custom_logger", [False, True])
    def test_logs_saved_namespace_hyperparameters(self, caplog, custom_logger):
        model = _TinyModel()
        model.config = FTTransformerConfig(d_model=16)
        model.extra_hparams = {"label": "example", "ignored": "hidden"}
        model.save_hyperparameters(ignore=["ignored"])
        original = vars(model.hparams).copy()
        logger_name = "deeptab.tests.parameters" if custom_logger else "deeptab.core.base_model"
        supplied_logger = logging.getLogger(logger_name) if custom_logger else None

        with caplog.at_level(logging.INFO, logger=logger_name):
            model.log_parameters(logger=supplied_logger)

        assert "Hyperparameters:" in caplog.text
        assert "d_model: 16" in caplog.text
        assert "label: example" in caplog.text
        assert "ignored" not in caplog.text
        assert "Total number of trainable parameters: 6" in caplog.text
        assert vars(model.hparams) == original


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


class TestEncodingAndClsPooling:
    @pytest.mark.parametrize("cls_position", [0, 1])
    def test_pooling_respects_embedding_cls_position(self, cls_position):
        from deeptab.architectures.ft_transformer import FTTransformer

        model = FTTransformer(
            (NUM_INFO, {}, {}),
            config=FTTransformerConfig(
                d_model=8, n_layers=1, n_heads=2, use_cls=True, cls_position=cls_position, pooling_method="cls"
            ),
        )
        sequence = model.embedding_layer([torch.randn(4, 1)], [], [])
        expected_index = 0 if cls_position == 0 else -1
        torch.testing.assert_close(model.pool_sequence(sequence), sequence[:, expected_index])
        torch.testing.assert_close(
            model.pool_sequence(sequence), model.embedding_layer.cls_token.squeeze(1).expand(4, -1)
        )

    @pytest.mark.parametrize("architecture_name", ["FTTransformer", "Mambular", "MambAttention", "TabulaRNN"])
    def test_encode_calls_selected_block_once_and_preserves_gradient_mode(self, architecture_name):
        import deeptab.configs as configs
        from deeptab import architectures

        model_class = getattr(architectures, architecture_name)
        config_class = getattr(configs, f"{architecture_name}Config")
        config = config_class(d_model=8, n_layers=1)
        if hasattr(config, "n_heads"):
            config.n_heads = 2
        if hasattr(config, "d_state"):
            config.d_state = 4
        if hasattr(config, "shuffle_embeddings"):
            config.shuffle_embeddings = True
        feature_info = ({f"feature_{index}": {"dimension": 1} for index in range(3)}, {}, {})
        model = model_class(feature_info, config=config).eval()
        block = next(getattr(model, name) for name in ["mamba", "rnn", "lstm", "encoder"] if hasattr(model, name))
        calls = []
        hook = block.register_forward_hook(lambda module, args, output: calls.append(output))
        data = ([torch.randn(4, 1) for _ in range(3)], [], [])
        without_grad = model.encode(data)
        assert len(calls) == 1
        assert not without_grad.requires_grad
        with_grad = model.encode(data, grad=True)
        assert len(calls) == 2
        assert with_grad.requires_grad
        torch.testing.assert_close(with_grad, without_grad)
        with_grad.square().sum().backward()
        hook.remove()

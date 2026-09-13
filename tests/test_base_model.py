"""Tests for deeptab.core.base_model.BaseModel.save_model / load_model.

Covers:
- save_model/load_model no longer print unconditionally to stdout; they log
  through the standard logging module instead, so a fit at ObservabilityConfig's
  default (no console handler attached) produces no output.
- The status message is still observable via `caplog` when logging is configured.
"""

from __future__ import annotations

import logging
import os
import tempfile

import torch

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

"""Tests for mamba blocks."""

from __future__ import annotations

import torch

from deeptab.nn.blocks.mamba import MambaBlock

B = 4  # batch size

D = 32  # embedding dim (divisible by H=4)

S = 6  # sequence length

E = 4  # ensemble size

H = 4  # attention heads

NF = 4  # number of features


class TestMambaBlock:
    def test_dilation_is_applied_to_both_convolutions(self):
        # The forward convolution previously ignored the dilation argument.
        block = MambaBlock(d_model=8, expand_factor=1, d_conv=3, d_state=8, dt_rank=4, dilation=2, bidirectional=True)
        assert block.conv1d_fwd.dilation == (2,)
        assert block.conv1d_bwd.dilation == (2,)

    def test_forward_with_dilation_and_bidirectional_does_not_crash(self):
        block = MambaBlock(d_model=8, expand_factor=1, d_conv=3, d_state=8, dt_rank=4, dilation=3, bidirectional=True)
        x = torch.randn(2, 10, 8)
        out = block(x)
        assert out.shape == x.shape

    def test_use_pscan_import_error_resets_use_pscan_flag(self):
        # mambapy is not installed in the test environment, so this exercises
        # the real ImportError fallback path, which previously left
        # use_pscan=True while pscan=None, crashing on the next forward call.
        block = MambaBlock(d_model=8, expand_factor=1, d_conv=3, d_state=8, dt_rank=4, use_pscan=True)
        assert block.use_pscan is False
        assert block.pscan is None

    def test_forward_does_not_crash_with_use_pscan_requested(self):
        block = MambaBlock(d_model=8, expand_factor=1, d_conv=3, d_state=8, dt_rank=4, use_pscan=True)
        x = torch.randn(2, 6, 8)
        out = block(x)
        assert out.shape == x.shape

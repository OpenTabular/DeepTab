"""Tests for common blocks."""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch.nn as nn

from deeptab.nn.blocks.common import (
    BatchNorm,
    BlockDiagonal,
    ConvRNN,
    EmbeddingLayer,
    EnsembleConvRNN,
    GroupNorm,
    InstanceNorm,
    LayerNorm,
    LearnableFourierFeatures,
    LearnableFourierMask,
    LearnableLayerScaling,
    LearnableRandomPositionalPerturbation,
    LearnableRandomProjection,
    LinearBatchEnsembleLayer,
    MultiHeadAttentionBatchEnsemble,
    NeuralEmbeddingTree,
    OneHotEncoding,
    Periodic,
    PeriodicEmbeddings,
    PeriodicLinearEncodingLayer,
    PositionalInvariance,
    RMSNorm,
    RNNBatchEnsembleLayer,
    SNLinear,
    mLSTMblock,
    sLSTMblock,
    sparsemax,
    sparsemoid,
)

B = 4  # batch size

D = 32  # embedding dim (divisible by H=4)

S = 6  # sequence length

E = 4  # ensemble size

H = 4  # attention heads

NF = 4  # number of features


class TestSNLinear:
    def test_forward_shape(self):
        lin = SNLinear(n=NF, in_features=8, out_features=16)
        x = torch.randn(B, NF, 8)
        assert lin(x).shape == (B, NF, 16)

    def test_2d_input_raises(self):
        lin = SNLinear(n=NF, in_features=8, out_features=16)
        with pytest.raises(ValueError):
            lin(torch.randn(B, 8))

    def test_feature_mismatch_raises(self):
        lin = SNLinear(n=NF, in_features=8, out_features=16)
        with pytest.raises(ValueError):
            lin(torch.randn(B, NF, 12))


class TestSparsemax:
    @pytest.mark.parametrize("dim", [0, -1])
    def test_forward_and_backward_preserve_parameter_values(self, dim):
        values = nn.Parameter(torch.tensor([[1.1, 1.3, -0.8], [2.0, -0.9, 2.4]], dtype=torch.double))
        original = values.detach().clone()
        output = sparsemax(values, dim=dim)

        torch.testing.assert_close(values, original, rtol=0, atol=0)
        torch.testing.assert_close(output.sum(dim=dim), torch.ones_like(output.sum(dim=dim)))
        torch.testing.assert_close(output, sparsemax(original + 10, dim=dim))
        weights = torch.arange(values.numel(), dtype=values.dtype).reshape_as(values)
        (output * weights).sum().backward()
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()
        torch.testing.assert_close(values, original, rtol=0, atol=0)

    def test_output_shape(self):
        out = sparsemax(torch.randn(B, 10))
        assert out is not None
        assert out.shape == (B, 10)

    def test_non_negative(self):
        out = sparsemax(torch.randn(B, 10))
        assert out is not None
        assert (out >= 0).all()

    def test_sparsemoid_range(self):
        out = sparsemoid(torch.randn(B, 10))
        assert out.shape == (B, 10)
        assert (out >= 0).all() and (out <= 1).all()

    # -----------------------------------------------------------------
    # Regression tests for https://github.com/OpenTabular/DeepTab/issues/453:
    # backward() used a bare supp_size.squeeze(), which also drops any other
    # singleton axis of the input (e.g. NODE depth=1), silently misaligning
    # the gradient broadcast instead of raising an error.
    # -----------------------------------------------------------------
    @pytest.mark.parametrize("shape", [(6, 4, 1), (3, 1, 5), (2, 1, 1, 7)])
    def test_backward_gradcheck_with_singleton_dims(self, shape):
        x = torch.randn(*shape, dtype=torch.double, requires_grad=True)
        assert torch.autograd.gradcheck(lambda inputs: sparsemax(inputs, dim=-1), (x,))

    def test_backward_matches_no_singleton_case(self):
        """The other axes' size shouldn't change the gradient sparsemax computes."""
        torch.manual_seed(0)
        base = torch.randn(4, 5, dtype=torch.double)

        x1 = base.unsqueeze(1).clone().requires_grad_(True)  # shape (4, 1, 5)
        out1 = sparsemax(x1, dim=-1)
        out1.sum().backward()

        x3 = base.unsqueeze(1).repeat(1, 3, 1).clone().requires_grad_(True)  # shape (4, 3, 5)
        out3 = sparsemax(x3, dim=-1)
        out3.sum().backward()

        assert x1.grad is not None
        assert x3.grad is not None
        assert torch.allclose(x1.grad.squeeze(1), x3.grad[:, 0, :])


class TestNormalizationLayers:
    def test_rmsnorm(self):
        assert RMSNorm(D)(torch.randn(B, D)).shape == (B, D)

    def test_layernorm(self):
        assert LayerNorm(D)(torch.randn(B, D)).shape == (B, D)

    def test_batchnorm_train(self):
        norm = BatchNorm(D)
        norm.train()
        assert norm(torch.randn(B, D)).shape == (B, D)

    def test_batchnorm_eval(self):
        norm = BatchNorm(D)
        norm.eval()
        assert norm(torch.randn(B, D)).shape == (B, D)

    def test_instancenorm(self):
        inputs = torch.randn(B, D, 3, 5, requires_grad=True)
        actual = InstanceNorm(D)(inputs)
        expected = torch.nn.functional.instance_norm(inputs)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        assert inputs.grad is not None

    def test_groupnorm(self):
        # D=32 divisible by num_groups=4
        assert GroupNorm(num_groups=4, d_model=D)(torch.randn(B, D, 4, 4)).shape == (B, D, 4, 4)

    def test_learnable_layer_scaling(self):
        assert LearnableLayerScaling(D)(torch.randn(B, D)).shape == (B, D)

    @pytest.mark.parametrize("norm_class", [BatchNorm, InstanceNorm, GroupNorm])
    @pytest.mark.parametrize("shape", [(4, 8), (4, 6, 8)])
    def test_feature_last_train_eval_and_backward(self, norm_class, shape):
        norm = norm_class(2, 8) if norm_class is GroupNorm else norm_class(8)
        for training in [True, False]:
            norm.train(training)
            inputs = torch.randn(*shape, requires_grad=True)
            output = norm(inputs)
            assert output.shape == inputs.shape
            assert torch.isfinite(output).all()
            output.square().sum().backward()
            assert inputs.grad is not None
            assert torch.isfinite(inputs.grad).all()

    def test_batchnorm_running_statistics_use_all_non_feature_axes(self):
        norm = BatchNorm(8, momentum=0.5)
        inputs = torch.randn(4, 6, 8, requires_grad=True)
        output = norm(inputs)
        mean = inputs.mean(dim=(0, 1))
        variance = inputs.var(dim=(0, 1), unbiased=False)
        torch.testing.assert_close(output, (inputs - mean) / torch.sqrt(variance + norm.eps))
        torch.testing.assert_close(norm.running_mean, 0.5 * mean.detach())
        torch.testing.assert_close(norm.running_var, 0.5 + 0.5 * variance.detach())
        assert not norm.running_mean.requires_grad
        assert not norm.running_var.requires_grad
        output.square().sum().backward()
        norm(torch.randn(4, 6, 8, requires_grad=True)).square().sum().backward()

    def test_sequence_norms_match_feature_last_references(self):
        inputs = torch.randn(4, 6, 8)
        expected_instance = (inputs - inputs.mean(dim=1, keepdim=True)) / torch.sqrt(
            inputs.var(dim=1, unbiased=False, keepdim=True) + 1e-5
        )
        torch.testing.assert_close(InstanceNorm(8)(inputs), expected_instance)
        expected_group = torch.nn.functional.group_norm(inputs.transpose(1, 2), 2).transpose(1, 2)
        torch.testing.assert_close(GroupNorm(2, 8)(inputs), expected_group)


class TestBlockDiagonal:
    def test_forward_shape(self):
        block = BlockDiagonal(in_features=8, out_features=16, num_blocks=4)
        assert block(torch.randn(B, 8)).shape == (B, 16)

    def test_indivisible_raises(self):
        with pytest.raises(ValueError):
            BlockDiagonal(in_features=8, out_features=10, num_blocks=3)


class TestLearnableFourier:
    def test_lff_shape(self):
        lff = LearnableFourierFeatures(num_features=NF, d_model=D)
        assert lff(torch.randn(B, NF, D)).shape == (B, NF, D)

    def test_lfm_shape(self):
        mask = LearnableFourierMask(NF)
        torch.testing.assert_close(mask.mask, torch.tensor([1.0, 1.0, 0.0, 0.0]))
        inputs = torch.randn(B, NF, D, requires_grad=True)
        output = mask(inputs)
        assert output.shape == inputs.shape
        output.square().sum().backward()
        assert mask.mask.grad is not None

    def test_lrpp_shape(self):
        lrpp = LearnableRandomPositionalPerturbation(num_features=NF, d_model=D)
        assert lrpp(torch.randn(B, NF, D)).shape == (B, NF, D)

    def test_lrp_shape(self):
        lrp = LearnableRandomProjection(d_model=D, projection_dim=16)
        assert lrp(torch.randn(B, NF, D)).shape == (B, NF, 16)


class TestPositionalInvariance:
    def _cfg(self, **kw):
        base = {"d_model": D, "keep_ratio": 0.5, "projection_dim": 16, "d_conv": 3, "conv_bias": True}
        base.update(kw)
        return SimpleNamespace(**base)

    def test_lfm(self):
        layer = PositionalInvariance(self._cfg(), "lfm", seq_len=NF)
        assert layer(torch.randn(B, NF, D)).shape == (B, NF, D)

    def test_lff(self):
        pi = PositionalInvariance(self._cfg(), "lff", seq_len=NF)
        assert pi(torch.randn(B, NF, D)).shape == (B, NF, D)

    def test_lprp(self):
        pi = PositionalInvariance(self._cfg(), "lprp", seq_len=NF)
        assert pi(torch.randn(B, NF, D)).shape == (B, NF, D)

    def test_lrp(self):
        pi = PositionalInvariance(self._cfg(), "lrp", seq_len=NF)
        assert pi(torch.randn(B, NF, D)).shape == (B, NF, 16)

    def test_conv(self):
        in_ch = 8
        pi = PositionalInvariance(self._cfg(), "conv", seq_len=S, in_channels=in_ch)
        out = pi(torch.randn(B, in_ch, S))
        assert out.shape == (B, in_ch, S)

    @pytest.mark.parametrize("kind", ["lfm", "lff", "lprp", "lrp", "conv"])
    @pytest.mark.parametrize("sequence_length", [3, 6])
    def test_rectangular_forward_and_backward(self, kind, sequence_length):
        layer = PositionalInvariance(self._cfg(), kind, seq_len=sequence_length)
        inputs = torch.randn(B, sequence_length, D, requires_grad=True)
        output = layer(inputs)
        width = 16 if kind == "lrp" else D
        assert output.shape == (B, sequence_length, width)
        assert torch.isfinite(output).all()
        output.square().sum().backward()
        assert inputs.grad is not None
        assert torch.isfinite(inputs.grad).all()
        assert all(parameter.grad is not None for parameter in layer.parameters())

    def test_invalid_type_raises(self):
        # The error message reads config.invariance_type, so the attribute must exist.
        cfg = self._cfg(invariance_type="unknown_type")
        with pytest.raises(ValueError):
            PositionalInvariance(cfg, "unknown_type", seq_len=S)


class TestPeriodic:
    def test_periodic_shape(self):
        p = Periodic(n_features=NF, k=8, sigma=0.01)
        assert p(torch.randn(B, NF)).shape == (B, NF, 16)  # 2*k

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError):
            Periodic(n_features=NF, k=8, sigma=0.0)

    def test_embeddings_standard(self):
        pe = PeriodicEmbeddings(n_features=NF, d_embedding=16, n_frequencies=8, activation=True, lite=False)
        assert pe(torch.randn(B, NF)).shape == (B, NF, 16)

    def test_embeddings_lite(self):
        pe = PeriodicEmbeddings(n_features=NF, d_embedding=16, n_frequencies=8, activation=True, lite=True)
        assert pe(torch.randn(B, NF)).shape == (B, NF, 16)

    def test_embeddings_no_activation(self):
        pe = PeriodicEmbeddings(n_features=NF, d_embedding=16, n_frequencies=8, activation=False, lite=False)
        assert pe(torch.randn(B, NF)).shape == (B, NF, 16)

    def test_embeddings_lite_no_activation_raises(self):
        with pytest.raises(ValueError):
            PeriodicEmbeddings(n_features=NF, d_embedding=16, activation=False, lite=True)


class TestNeuralEmbeddingTree:
    def test_forward_shape(self):
        # output_dim must be a power of 2
        tree = NeuralEmbeddingTree(input_dim=8, output_dim=8)
        assert tree(torch.randn(B, 8)).shape == (B, 8)

    def test_with_temperature(self):
        tree = NeuralEmbeddingTree(input_dim=8, output_dim=4, temperature=1.0)
        assert tree(torch.randn(B, 8)).shape == (B, 4)


class TestPeriodicLinearEncoding:
    def test_learnable_bins(self):
        enc = PeriodicLinearEncodingLayer(bins=10, learn_bins=True)
        x = torch.linspace(0.0, 1.0, B).unsqueeze(1)
        assert enc(x).shape == (B, 10)

    def test_fixed_bins(self):
        enc = PeriodicLinearEncodingLayer(bins=8, learn_bins=False)
        x = torch.linspace(0.0, 1.0, B).unsqueeze(1)
        assert enc(x).shape == (B, 8)


def _num_info(n):
    return {f"f{i}": {"dimension": 1, "preprocessing": ""} for i in range(n)}


def _cat_info(n, cats=5):
    return {f"c{i}": {"dimension": 1, "categories": cats} for i in range(n)}


def _emb_cfg(embedding_type="linear", **kw):
    cfg = SimpleNamespace(
        d_model=16,
        embedding_activation=nn.Identity(),
        layer_norm_after_embedding=False,
        embedding_projection=True,
        use_cls=False,
        cls_position=0,
        embedding_dropout=None,
        embedding_type=embedding_type,
        embedding_bias=False,
        n_frequencies=8,
        frequency_init_scale=0.01,
        plr_lite=False,
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


class TestEmbeddingLayer:
    def test_num_and_cat(self):
        layer = EmbeddingLayer(_num_info(2), _cat_info(1), {}, _emb_cfg())
        out = layer([torch.randn(B, 1), torch.randn(B, 1)], [torch.randint(0, 5, (B,))], [])
        assert out.shape == (B, 3, 16)

    def test_num_only(self):
        layer = EmbeddingLayer(_num_info(3), {}, {}, _emb_cfg())
        out = layer([torch.randn(B, 1)] * 3, [], [])
        assert out.shape == (B, 3, 16)

    def test_cat_only(self):
        layer = EmbeddingLayer({}, _cat_info(2), {}, _emb_cfg())
        out = layer([], [torch.randint(0, 5, (B,))] * 2, [])
        assert out.shape == (B, 2, 16)

    def test_layer_norm_after_embedding(self):
        layer = EmbeddingLayer(_num_info(2), {}, {}, _emb_cfg(layer_norm_after_embedding=True))
        out = layer([torch.randn(B, 1)] * 2, [], [])
        assert out.shape == (B, 2, 16)

    def test_use_cls_prepend(self):
        layer = EmbeddingLayer(_num_info(2), {}, {}, _emb_cfg(use_cls=True, cls_position=0))
        out = layer([torch.randn(B, 1)] * 2, [], [])
        assert out.shape == (B, 3, 16)  # 2 features + CLS

    def test_use_cls_append(self):
        layer = EmbeddingLayer(_num_info(2), {}, {}, _emb_cfg(use_cls=True, cls_position=1))
        out = layer([torch.randn(B, 1)] * 2, [], [])
        assert out.shape == (B, 3, 16)

    def test_plr_embedding(self):
        layer = EmbeddingLayer(_num_info(3), {}, {}, _emb_cfg(embedding_type="plr"))
        out = layer([torch.randn(B, 1)] * 3, [], [])
        assert out.shape == (B, 3, 16)

    def test_ndt_embedding(self):
        # d_model=16 is a power of 2, required by NeuralEmbeddingTree
        layer = EmbeddingLayer({"f0": {"dimension": 1, "preprocessing": ""}}, {}, {}, _emb_cfg(embedding_type="ndt"))
        out = layer([torch.randn(B, 1)], [], [])
        assert out.shape[0] == B

    def test_invalid_embedding_type_raises(self):
        with pytest.raises(ValueError):
            EmbeddingLayer(_num_info(2), {}, {}, _emb_cfg(embedding_type="invalid"))

    def test_embedding_dropout(self):
        layer = EmbeddingLayer(_num_info(2), {}, {}, _emb_cfg(embedding_dropout=0.1))
        layer.train()
        out = layer([torch.randn(B, 1)] * 2, [], [])
        assert out.shape == (B, 2, 16)

    def test_emb_features(self):
        emb_info = {"e0": {"dimension": 8, "preprocessing": ""}}
        layer = EmbeddingLayer({}, {}, emb_info, _emb_cfg())
        out = layer([], [], [torch.randn(B, 8)])
        assert out.shape == (B, 1, 16)

    def test_plr_incompatible_preprocessing_raises(self):
        num_info = {"f0": {"dimension": 1, "preprocessing": "one-hot"}}
        layer = EmbeddingLayer(num_info, {}, {}, _emb_cfg(embedding_type="plr"))
        with pytest.raises(ValueError):
            layer([torch.randn(B, 1)], [], [])


class TestEmbeddingLayerCatEncoding:
    """cat_encoding was declared on the config but never consumed by EmbeddingLayer."""

    @staticmethod
    def _cat_embedding_seq(layer, index=0):
        return cast(nn.Sequential, layer.cat_embeddings[index])

    def test_int_default_uses_embedding_table(self):
        layer = EmbeddingLayer({}, _cat_info(1), {}, _emb_cfg(cat_encoding="int"))
        assert isinstance(self._cat_embedding_seq(layer)[0], nn.Embedding)
        out = layer([], [torch.randint(0, 5, (B,))], [])
        assert out.shape == (B, 1, 16)

    def test_one_hot_encoding_builds_and_runs(self):
        layer = EmbeddingLayer({}, _cat_info(1), {}, _emb_cfg(cat_encoding="one-hot"))
        seq = self._cat_embedding_seq(layer)
        assert isinstance(seq[0], OneHotEncoding)
        linear = seq[1]
        assert isinstance(linear, nn.Linear)
        assert linear.in_features == 6  # categories + 1
        out = layer([], [torch.randint(0, 5, (B,))], [])
        assert out.shape == (B, 1, 16)

    def test_linear_encoding_builds_and_runs(self):
        layer = EmbeddingLayer({}, _cat_info(1), {}, _emb_cfg(cat_encoding="linear"))
        linear = self._cat_embedding_seq(layer)[1]
        assert isinstance(linear, nn.Linear)
        assert linear.in_features == 1
        out = layer([], [torch.randint(0, 5, (B,))], [])
        assert out.shape == (B, 1, 16)

    def test_missing_cat_encoding_attribute_defaults_to_int(self):
        # Configs/namespaces that predate cat_encoding shouldn't change behavior.
        cfg = _emb_cfg()
        assert not hasattr(cfg, "cat_encoding")
        layer = EmbeddingLayer({}, _cat_info(1), {}, cfg)
        assert isinstance(self._cat_embedding_seq(layer)[0], nn.Embedding)


class TestOneHotEncoding:
    def test_shape(self):
        enc = OneHotEncoding(num_categories=5)
        out = enc(torch.randint(0, 5, (B,)))
        assert out.shape == (B, 5)


class TestScaledPolynomialLayer:
    def test_forward_runs(self):
        from deeptab.nn.blocks.common import ScaledPolynomialLayer

        # With degree=2 and 1 input feature, PolynomialFeatures generates exactly
        # 2 columns (x, x^2), matching self.weights shape (degree=2,).
        layer = ScaledPolynomialLayer(degree=2)
        out = layer(torch.randn(B, 1))
        assert out.shape[0] == B


class TestLinearBatchEnsembleLayer:
    def test_2d_input(self):
        layer = LinearBatchEnsembleLayer(in_features=8, out_features=16, ensemble_size=E)
        assert layer(torch.randn(B, 8)).shape == (B, E, 16)

    def test_3d_input(self):
        layer = LinearBatchEnsembleLayer(in_features=8, out_features=16, ensemble_size=E)
        assert layer(torch.randn(B, E, 8)).shape == (B, E, 16)

    def test_ensemble_mismatch_raises(self):
        layer = LinearBatchEnsembleLayer(in_features=8, out_features=16, ensemble_size=E)
        with pytest.raises(ValueError):
            layer(torch.randn(B, E + 1, 8))

    @pytest.mark.parametrize("init", ["ones", "random-signs", "normal"])
    def test_scaling_inits(self, init):
        layer = LinearBatchEnsembleLayer(in_features=8, out_features=16, ensemble_size=E, scaling_init=init)
        assert layer(torch.randn(B, 8)).shape == (B, E, 16)

    def test_no_input_scaling(self):
        layer = LinearBatchEnsembleLayer(
            in_features=8, out_features=16, ensemble_size=E, ensemble_scaling_in=False, ensemble_scaling_out=False
        )
        assert layer(torch.randn(B, 8)).shape == (B, E, 16)

    def test_ensemble_bias(self):
        layer = LinearBatchEnsembleLayer(in_features=8, out_features=16, ensemble_size=E, ensemble_bias=True)
        assert layer(torch.randn(B, 8)).shape == (B, E, 16)


class TestMultiHeadAttentionBatchEnsemble:
    def _mha(self, projections=None, **kw):
        kw.setdefault("embed_dim", D)
        kw.setdefault("num_heads", H)
        kw.setdefault("ensemble_size", E)
        if projections is not None:
            kw["batch_ensemble_projections"] = projections
        return MultiHeadAttentionBatchEnsemble(**kw)

    def test_forward_shape(self):
        x = torch.randn(B, S, E, D)
        assert self._mha()(x, x, x).shape == (B, S, E, D)

    @pytest.mark.parametrize("projection", ["query", "key", "value", "out_proj"])
    @pytest.mark.parametrize("non_contiguous", [False, True])
    def test_projection_preserves_sequence_and_ensemble_axes(self, projection, non_contiguous):
        layer = self._mha(projections=[])
        inputs = torch.randn(B, S, E, D)
        if non_contiguous:
            inputs = torch.randn(B, E, S, D).transpose(1, 2)
        projection_name = {"query": "q_proj", "key": "k_proj", "value": "v_proj", "out_proj": "out_proj"}[projection]
        projection_layer = getattr(layer, projection_name)
        torch.testing.assert_close(
            layer.process_projection(inputs, projection_layer, projection), projection_layer(inputs)
        )

    def test_attention_matches_independent_members(self):
        layer = self._mha(scaling_init="ones")
        reference = nn.MultiheadAttention(D, H, batch_first=True, dropout=0.0)
        with torch.no_grad():
            reference.in_proj_weight.copy_(torch.cat([layer.q_proj.weight, layer.k_proj.weight, layer.v_proj.weight]))
            reference.in_proj_bias.copy_(torch.cat([layer.q_proj.bias, layer.k_proj.bias, layer.v_proj.bias]))
            reference.out_proj.load_state_dict(layer.out_proj.state_dict())
        inputs = torch.randn(B, S, E, D, requires_grad=True)
        expected = torch.stack(
            [
                reference(inputs[:, :, member], inputs[:, :, member], inputs[:, :, member], need_weights=False)[0]
                for member in range(E)
            ],
            dim=2,
        )
        output = layer(inputs, inputs, inputs)
        torch.testing.assert_close(output, expected)
        output.square().sum().backward()
        assert inputs.grad is not None

    def test_embed_not_divisible_raises(self):
        with pytest.raises(ValueError):
            MultiHeadAttentionBatchEnsemble(embed_dim=10, num_heads=3, ensemble_size=E)

    def test_ensemble_mismatch_raises(self):
        mha = self._mha()
        with pytest.raises(ValueError):
            mha(torch.randn(B, S, E + 1, D), torch.randn(B, S, E + 1, D), torch.randn(B, S, E + 1, D))

    @pytest.mark.parametrize("proj", [["key"], ["value"], ["out_proj"], ["query", "key", "value"]])
    def test_various_projections(self, proj):
        x = torch.randn(B, S, E, D)
        assert self._mha(projections=proj)(x, x, x).shape == (B, S, E, D)

    def test_with_mask(self):
        x = torch.randn(B, S, E, D)
        mask = torch.ones(B, S)
        assert self._mha()(x, x, x, mask=mask).shape == (B, S, E, D)

    def test_invalid_projection_raises(self):
        with pytest.raises(ValueError):
            self._mha(projections=["invalid"])

    @pytest.mark.parametrize("init", ["ones", "random-signs", "normal"])
    def test_scaling_inits(self, init):
        x = torch.randn(B, S, E, D)
        assert self._mha(scaling_init=init)(x, x, x).shape == (B, S, E, D)


class TestRNNBatchEnsembleLayer:
    def test_3d_input(self):
        rnn = RNNBatchEnsembleLayer(input_size=8, hidden_size=16, ensemble_size=E)
        out, _ = rnn(torch.randn(B, S, 8))
        assert out.shape == (B, S, E, 16)

    def test_4d_input(self):
        rnn = RNNBatchEnsembleLayer(input_size=8, hidden_size=16, ensemble_size=E)
        out, _ = rnn(torch.randn(B, S, E, 8))
        assert out.shape == (B, S, E, 16)

    def test_ensemble_mismatch_4d_raises(self):
        rnn = RNNBatchEnsembleLayer(input_size=8, hidden_size=16, ensemble_size=E)
        with pytest.raises(ValueError):
            rnn(torch.randn(B, S, E + 1, 8))

    def test_invalid_shape_raises(self):
        rnn = RNNBatchEnsembleLayer(input_size=8, hidden_size=16, ensemble_size=E)
        with pytest.raises(ValueError):
            rnn(torch.randn(B, 8))  # 2D

    @pytest.mark.parametrize("init", ["ones", "random-signs", "normal"])
    def test_scaling_inits(self, init):
        rnn = RNNBatchEnsembleLayer(input_size=8, hidden_size=16, ensemble_size=E, scaling_init=init)
        out, _ = rnn(torch.randn(B, S, 8))
        assert out.shape == (B, S, E, 16)

    def test_no_scaling(self):
        rnn = RNNBatchEnsembleLayer(
            input_size=8, hidden_size=16, ensemble_size=E, ensemble_scaling_in=False, ensemble_scaling_out=False
        )
        out, _ = rnn(torch.randn(B, S, 8))
        assert out.shape == (B, S, E, 16)

    def test_ensemble_bias(self):
        rnn = RNNBatchEnsembleLayer(input_size=8, hidden_size=16, ensemble_size=E, ensemble_bias=True)
        out, _ = rnn(torch.randn(B, S, 8))
        assert out.shape == (B, S, E, 16)


class TestmLSTMblock:
    def test_forward_shape(self):
        # hidden_size and num_layers: BlockDiagonal needs hidden_size % num_layers == 0
        block = mLSTMblock(input_size=8, hidden_size=8, num_layers=2)
        out, _ = block(torch.randn(B, S, 8))
        assert out.shape == (B, S, 8)

    def test_2d_input_raises(self):
        block = mLSTMblock(input_size=8, hidden_size=8, num_layers=1)
        with pytest.raises(ValueError):
            block(torch.randn(B, 8))

    def test_state_reinit_on_batch_change(self):
        block = mLSTMblock(input_size=8, hidden_size=8, num_layers=2)
        out1, _ = block(torch.randn(B, S, 8))
        out2, _ = block(torch.randn(B * 2, S, 8))
        assert out1.shape[0] == B
        assert out2.shape[0] == B * 2


class TestsLSTMblock:
    def test_forward_runs(self):
        block = sLSTMblock(input_size=8, hidden_size=8, num_layers=2)
        out, _ = block(torch.randn(B, S, 8))
        assert out.shape == (B, S, 8)

    def test_state_reinit_on_batch_change(self):
        block = sLSTMblock(input_size=8, hidden_size=8, num_layers=2)
        block(torch.randn(B, S, 8))
        block(torch.randn(B * 2, S, 8))  # must not raise


@pytest.mark.parametrize("block_class", [mLSTMblock, sLSTMblock])
def test_lstm_blocks_are_batch_independent_and_explicitly_stateful(block_class):
    block = block_class(input_size=8, hidden_size=8, num_layers=2, dropout=0.0).double().eval()
    inputs = torch.randn(4, 6, 8, dtype=torch.double, requires_grad=True)
    output, state = block(inputs)
    independent = torch.cat([block(row.unsqueeze(0))[0] for row in inputs], dim=0)
    torch.testing.assert_close(output, independent)
    torch.testing.assert_close(output, block(inputs)[0], rtol=0, atol=0)
    assert output.shape == inputs.shape
    assert all(tensor.shape == (4, 8) for tensor in state)
    assert all(tensor.dtype == inputs.dtype for tensor in state)
    continued, next_state = block(inputs, state=state)
    assert not torch.allclose(output, continued)
    assert all(torch.isfinite(tensor).all() for tensor in next_state)
    continued.square().sum().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    with pytest.raises(ValueError, match="state"):
        block(inputs[:1], state=state)


def test_slstm_explicit_state_matches_whole_sequence():
    block = sLSTMblock(input_size=8, hidden_size=12, num_layers=2, dropout=0.0).eval()
    inputs = torch.randn(4, 6, 8)
    expected, _ = block(inputs)
    first, state = block(inputs[:, :2])
    second, _ = block(inputs[:, 2:], state=state)
    torch.testing.assert_close(torch.cat([first, second], dim=1), expected)


def _convrnn_cfg(model_type="RNN", n_layers=2, residuals=False, rnn_dropout=0.0):
    return SimpleNamespace(
        model_type=model_type,
        d_model=8,
        dim_feedforward=8,
        n_layers=n_layers,
        rnn_dropout=rnn_dropout,
        bias=True,
        conv_bias=True,
        rnn_activation="relu",
        d_conv=3,
        residuals=residuals,
        dilation=1,
    )


class TestConvRNN:
    @pytest.mark.parametrize("model_type", ["RNN", "LSTM", "GRU"])
    def test_standard_rnn_types(self, model_type):
        rnn = ConvRNN(_convrnn_cfg(model_type=model_type))
        out, _ = rnn(torch.randn(B, S, 8))
        assert out.shape == (B, S, 8)

    def test_mlstm(self):
        # n_layers=1 for BlockDiagonal: hidden_size=8, num_layers=1 → 8%1==0
        rnn = ConvRNN(_convrnn_cfg(model_type="mLSTM", n_layers=1))
        out, _ = rnn(torch.randn(B, S, 8))
        assert out.shape == (B, S, 8)

    def test_slstm(self):
        rnn = ConvRNN(_convrnn_cfg(model_type="sLSTM", n_layers=1))
        out, _ = rnn(torch.randn(B, S, 8))
        assert out is not None  # sLSTM reduces batch/seq dims internally

    def test_residuals(self):
        rnn = ConvRNN(_convrnn_cfg(residuals=True))
        out, _ = rnn(torch.randn(B, S, 8))
        assert out.shape == (B, S, 8)

    def test_dropout_changes_output_across_calls_in_train_mode(self):
        # rnn_dropout was silently a no-op: each stacked layer is its own
        # single-layer RNN module, and the per-layer "dropout" kwarg has no
        # effect on a single-layer RNN.
        torch.manual_seed(0)
        rnn = ConvRNN(_convrnn_cfg(rnn_dropout=0.9))
        rnn.train()
        x = torch.randn(B, S, 8)
        out1, _ = rnn(x)
        out2, _ = rnn(x)
        assert not torch.allclose(out1, out2)

    def test_zero_dropout_is_deterministic(self):
        torch.manual_seed(0)
        rnn = ConvRNN(_convrnn_cfg(rnn_dropout=0.0))
        rnn.eval()
        x = torch.randn(B, S, 8)
        out1, _ = rnn(x)
        out2, _ = rnn(x)
        assert torch.allclose(out1, out2)

    def test_no_pytorch_dropout_warning_emitted(self):
        # PyTorch previously warned "dropout... expects num_layers greater than
        # 1, but got ... num_layers=1" at *construction* time, since each
        # stacked block is its own single-layer RNN module.
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            rnn = ConvRNN(_convrnn_cfg(rnn_dropout=0.5))
            rnn.train()
            rnn(torch.randn(B, S, 8))


def _ensemble_convrnn_cfg(model_type="full"):
    return SimpleNamespace(
        d_model=8,
        dim_feedforward=8,
        ensemble_size=E,
        n_layers=2,
        rnn_dropout=0.0,
        bias=True,
        conv_bias=True,
        rnn_activation=torch.tanh,
        d_conv=3,
        residuals=False,
        ensemble_scaling_in=True,
        ensemble_scaling_out=True,
        ensemble_bias=False,
        scaling_init="ones",
        model_type=model_type,
    )


class TestEnsembleConvRNN:
    def test_full_model_type(self):
        rnn = EnsembleConvRNN(_ensemble_convrnn_cfg("full"))
        out, _ = rnn(torch.randn(B, S, 8))
        assert out.shape == (B, S, E, 8)

    def test_mini_model_type(self):
        rnn = EnsembleConvRNN(_ensemble_convrnn_cfg("mini"))
        out, _ = rnn(torch.randn(B, S, 8))
        assert out.shape == (B, S, E, 8)

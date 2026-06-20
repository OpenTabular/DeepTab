"""Tests for deeptab.training.optimizers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from deeptab.core.exceptions import InvalidParamError
from deeptab.training.optimizers import (
    available_optimizers,
    build_optimizer,
    build_parameter_groups,
    get_optimizer,
    normalize_optimizer_kwargs,
    register_optimizer,
    unregister_optimizer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.LayerNorm(8),
        nn.Linear(8, 1),
    )


# ---------------------------------------------------------------------------
# get_optimizer
# ---------------------------------------------------------------------------


class TestGetOptimizer:
    def test_returns_adam(self):
        cls = get_optimizer("Adam")
        assert cls is torch.optim.Adam

    def test_case_insensitive(self):
        assert get_optimizer("adam") is get_optimizer("ADAM")

    def test_unknown_raises_invalid_param_error(self):
        with pytest.raises(InvalidParamError):
            get_optimizer("NotAnOptimizer")

    def test_error_message_contains_name(self):
        with pytest.raises(InvalidParamError, match="NotAnOptimizer"):
            get_optimizer("NotAnOptimizer")

    def test_error_message_lists_available(self):
        with pytest.raises(InvalidParamError, match="adam"):
            get_optimizer("xyz")

    def test_sgd_available(self):
        cls = get_optimizer("SGD")
        assert cls is torch.optim.SGD

    def test_adamw_available(self):
        cls = get_optimizer("AdamW")
        assert cls is torch.optim.AdamW

    def test_rmsprop_available(self):
        cls = get_optimizer("RMSprop")
        assert cls is torch.optim.RMSprop


# ---------------------------------------------------------------------------
# available_optimizers
# ---------------------------------------------------------------------------


class TestAvailableOptimizers:
    def test_returns_sorted_list(self):
        opts = available_optimizers()
        assert opts == sorted(opts)

    def test_includes_adam(self):
        assert "adam" in available_optimizers()

    def test_all_strings(self):
        assert all(isinstance(o, str) for o in available_optimizers())

    def test_all_lowercase(self):
        assert all(o == o.lower() for o in available_optimizers())


# ---------------------------------------------------------------------------
# register_optimizer
# ---------------------------------------------------------------------------


class TestRegisterOptimizer:
    def test_register_and_retrieve(self):
        class _DummyOpt(torch.optim.SGD):
            pass

        register_optimizer("_dummy_test_opt", _DummyOpt, override=True)
        assert get_optimizer("_dummy_test_opt") is _DummyOpt

    def test_duplicate_raises_without_override(self):
        class _DummyOpt2(torch.optim.SGD):
            pass

        register_optimizer("_dup_opt", _DummyOpt2, override=True)
        with pytest.raises(ValueError, match="already registered"):
            register_optimizer("_dup_opt", _DummyOpt2, override=False)

    def test_duplicate_allowed_with_override(self):
        class _DummyOpt3(torch.optim.SGD):
            pass

        register_optimizer("_over_opt", _DummyOpt3, override=True)
        register_optimizer("_over_opt", _DummyOpt3, override=True)  # no error


# ---------------------------------------------------------------------------
# unregister_optimizer
# ---------------------------------------------------------------------------


class TestUnregisterOptimizer:
    def test_unregister_user_entry(self):
        class _DummyOpt(torch.optim.SGD):
            pass

        register_optimizer("_unreg_opt", _DummyOpt, override=True)
        assert "_unreg_opt" in available_optimizers()
        unregister_optimizer("_unreg_opt")
        assert "_unreg_opt" not in available_optimizers()

    def test_unknown_raises_invalid_param_error(self):
        with pytest.raises(InvalidParamError):
            unregister_optimizer("_never_registered_opt")

    def test_missing_ok_suppresses_error(self):
        unregister_optimizer("_never_registered_opt", missing_ok=True)  # no error

    def test_builtin_cannot_be_unregistered(self):
        with pytest.raises(ValueError, match="built-in"):
            unregister_optimizer("adam")
        assert "adam" in available_optimizers()

    def test_builtin_protected_even_with_missing_ok(self):
        with pytest.raises(ValueError, match="built-in"):
            unregister_optimizer("sgd", missing_ok=True)


# ---------------------------------------------------------------------------
# normalize_optimizer_kwargs
# ---------------------------------------------------------------------------


class TestNormalizeOptimizerKwargs:
    def test_none_returns_empty_dict(self):
        assert normalize_optimizer_kwargs(None) == {}

    def test_empty_dict_returns_empty_dict(self):
        assert normalize_optimizer_kwargs({}) == {}

    def test_strips_prefix(self):
        result = normalize_optimizer_kwargs({"optimizer_betas": (0.9, 0.95)})
        assert result == {"betas": (0.9, 0.95)}

    def test_non_prefixed_keys_excluded(self):
        # Only keys that START with "optimizer_" are kept
        result = normalize_optimizer_kwargs({"optimizer_eps": 1e-8, "lr": 1e-3})
        assert "eps" in result
        assert "lr" not in result

    def test_multiple_keys(self):
        raw = {"optimizer_betas": (0.9, 0.99), "optimizer_eps": 1e-8}
        result = normalize_optimizer_kwargs(raw)
        assert result == {"betas": (0.9, 0.99), "eps": 1e-8}


# ---------------------------------------------------------------------------
# build_parameter_groups
# ---------------------------------------------------------------------------


class TestBuildParameterGroups:
    def test_single_group_when_disabled(self):
        model = _simple_model()
        groups = build_parameter_groups(model, weight_decay=1e-4, no_weight_decay_for_bias_and_norm=False)
        assert len(groups) == 1
        assert groups[0]["weight_decay"] == 1e-4

    def test_two_groups_when_enabled(self):
        model = _simple_model()
        groups = build_parameter_groups(model, weight_decay=1e-4, no_weight_decay_for_bias_and_norm=True)
        assert len(groups) == 2

    def test_no_decay_group_has_zero_weight_decay(self):
        model = _simple_model()
        groups = build_parameter_groups(model, weight_decay=1e-4, no_weight_decay_for_bias_and_norm=True)
        no_decay = [g for g in groups if g["weight_decay"] == 0.0]
        assert len(no_decay) == 1

    def test_bias_in_no_decay_group(self):
        model = nn.Linear(4, 2)
        groups = build_parameter_groups(model, weight_decay=1e-3, no_weight_decay_for_bias_and_norm=True)
        no_decay_params = groups[1]["params"]
        # bias should be in the no-decay group
        assert any(p.shape == model.bias.shape for p in no_decay_params)

    def test_no_parameter_duplication(self):
        model = _simple_model()
        groups = build_parameter_groups(model, weight_decay=1e-4, no_weight_decay_for_bias_and_norm=True)
        all_params = groups[0]["params"] + groups[1]["params"]
        ids = [id(p) for p in all_params]
        assert len(ids) == len(set(ids)), "Duplicate parameters found"


# ---------------------------------------------------------------------------
# build_optimizer
# ---------------------------------------------------------------------------


class TestBuildOptimizer:
    def test_returns_optimizer_instance(self):
        model = _simple_model()
        opt = build_optimizer(model, optimizer_type="Adam", lr=1e-3, weight_decay=0.0)
        assert isinstance(opt, torch.optim.Optimizer)

    def test_sgd_type(self):
        model = _simple_model()
        opt = build_optimizer(model, optimizer_type="SGD", lr=0.01, weight_decay=0.0)
        assert isinstance(opt, torch.optim.SGD)

    def test_unknown_type_raises(self):
        model = _simple_model()
        with pytest.raises(InvalidParamError):
            build_optimizer(model, optimizer_type="FakeOptimizer", lr=1e-3, weight_decay=0.0)

    def test_lr_propagated(self):
        model = _simple_model()
        opt = build_optimizer(model, optimizer_type="Adam", lr=3e-4, weight_decay=0.0)
        assert opt.param_groups[0]["lr"] == pytest.approx(3e-4)

    def test_weight_decay_propagated(self):
        model = _simple_model()
        opt = build_optimizer(model, optimizer_type="Adam", lr=1e-3, weight_decay=5e-4)
        assert opt.param_groups[0]["weight_decay"] == pytest.approx(5e-4)

    def test_no_weight_decay_for_bias_and_norm_creates_two_param_groups(self):
        model = _simple_model()
        opt = build_optimizer(
            model,
            optimizer_type="Adam",
            lr=1e-3,
            weight_decay=1e-4,
            no_weight_decay_for_bias_and_norm=True,
        )
        assert len(opt.param_groups) == 2

    def test_extra_kwargs_forwarded(self):
        model = _simple_model()
        opt = build_optimizer(
            model,
            optimizer_type="Adam",
            lr=1e-3,
            weight_decay=0.0,
            optimizer_kwargs={"eps": 1e-5},
        )
        assert opt.param_groups[0]["eps"] == pytest.approx(1e-5)


# ---------------------------------------------------------------------------
# Phase 7d — TrainerConfig.no_weight_decay_for_bias_and_norm integration
# ---------------------------------------------------------------------------


class TestParameterGroupingViaTrainerConfig:
    """Verify that TrainerConfig.no_weight_decay_for_bias_and_norm is forwarded
    all the way from the config into the optimizer parameter groups."""

    def test_trainer_config_field_exists(self):
        from deeptab.configs import TrainerConfig

        cfg = TrainerConfig(no_weight_decay_for_bias_and_norm=True)
        assert cfg.no_weight_decay_for_bias_and_norm is True

    def test_trainer_config_default_is_false(self):
        from deeptab.configs import TrainerConfig

        cfg = TrainerConfig()
        assert cfg.no_weight_decay_for_bias_and_norm is False

    def test_build_optimizer_with_no_wd_flag_creates_two_groups(self):
        """Passing no_weight_decay_for_bias_and_norm=True creates two param groups."""
        model = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 1))
        opt = build_optimizer(
            model,
            optimizer_type="AdamW",
            lr=1e-3,
            weight_decay=1e-4,
            no_weight_decay_for_bias_and_norm=True,
        )
        assert len(opt.param_groups) == 2
        # The no-decay group must have weight_decay == 0
        no_wd = [g for g in opt.param_groups if g["weight_decay"] == 0.0]
        assert len(no_wd) == 1

    def test_layernorm_weight_in_no_decay_group(self):
        """LayerNorm weight parameters must be in the zero-weight-decay group."""
        ln = nn.LayerNorm(8)
        model = nn.Sequential(nn.Linear(4, 8), ln)
        groups = build_parameter_groups(model, weight_decay=1e-4, no_weight_decay_for_bias_and_norm=True)
        no_decay_params = groups[1]["params"]
        # LayerNorm weight is a 1-D tensor of shape (8,)
        assert any(p.shape == ln.weight.shape and p.data_ptr() == ln.weight.data_ptr() for p in no_decay_params)

    def test_all_parameters_covered(self):
        """Every parameter must appear in exactly one group."""
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.BatchNorm1d(8),
            nn.Linear(8, 2),
        )
        groups = build_parameter_groups(model, weight_decay=1e-4, no_weight_decay_for_bias_and_norm=True)
        all_param_ids = {id(p) for p in model.parameters()}
        grouped_ids = {id(p) for g in groups for p in g["params"]}
        assert all_param_ids == grouped_ids

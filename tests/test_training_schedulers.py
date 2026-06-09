"""Tests for deeptab.training.schedulers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from deeptab.core.exceptions import InvalidParamError
from deeptab.training.schedulers import available_schedulers, build_scheduler, get_scheduler, register_scheduler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_optimizer() -> torch.optim.Optimizer:
    model = nn.Linear(4, 2)
    return torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------------------------------------------------------------------------
# get_scheduler
# ---------------------------------------------------------------------------


class TestGetScheduler:
    def test_returns_reduce_lr_on_plateau(self):
        cls = get_scheduler("ReduceLROnPlateau")
        assert cls is torch.optim.lr_scheduler.ReduceLROnPlateau

    def test_case_insensitive(self):
        assert get_scheduler("reducelronplateau") is get_scheduler("REDUCELRONPLATEAU")

    def test_unknown_raises_invalid_param_error(self):
        with pytest.raises(InvalidParamError):
            get_scheduler("NotAScheduler")

    def test_error_message_contains_name(self):
        with pytest.raises(InvalidParamError, match="NotAScheduler"):
            get_scheduler("NotAScheduler")

    def test_error_message_lists_available(self):
        with pytest.raises(InvalidParamError, match="reducelronplateau"):
            get_scheduler("xyz")

    def test_steplr_available(self):
        cls = get_scheduler("StepLR")
        assert cls is torch.optim.lr_scheduler.StepLR

    def test_cosine_available(self):
        cls = get_scheduler("CosineAnnealingLR")
        assert cls is torch.optim.lr_scheduler.CosineAnnealingLR


# ---------------------------------------------------------------------------
# available_schedulers
# ---------------------------------------------------------------------------


class TestAvailableSchedulers:
    def test_returns_sorted_list(self):
        scheds = available_schedulers()
        assert scheds == sorted(scheds)

    def test_includes_plateau(self):
        assert "reducelronplateau" in available_schedulers()

    def test_all_strings(self):
        assert all(isinstance(s, str) for s in available_schedulers())

    def test_all_lowercase(self):
        assert all(s == s.lower() for s in available_schedulers())


# ---------------------------------------------------------------------------
# register_scheduler
# ---------------------------------------------------------------------------


class TestRegisterScheduler:
    def test_register_and_retrieve(self):
        class _DummySched(torch.optim.lr_scheduler.StepLR):
            pass

        register_scheduler("_dummy_test_sched", _DummySched, override=True)
        assert get_scheduler("_dummy_test_sched") is _DummySched

    def test_duplicate_raises_without_override(self):
        class _DupSched(torch.optim.lr_scheduler.StepLR):
            pass

        register_scheduler("_dup_sched", _DupSched, override=True)
        with pytest.raises(ValueError, match="already registered"):
            register_scheduler("_dup_sched", _DupSched, override=False)

    def test_duplicate_allowed_with_override(self):
        class _OverSched(torch.optim.lr_scheduler.StepLR):
            pass

        register_scheduler("_over_sched", _OverSched, override=True)
        register_scheduler("_over_sched", _OverSched, override=True)  # no error


# ---------------------------------------------------------------------------
# build_scheduler
# ---------------------------------------------------------------------------


class TestBuildScheduler:
    def test_none_returns_none(self):
        opt = _simple_optimizer()
        assert build_scheduler(opt, scheduler_type=None) is None

    def test_string_none_returns_none(self):
        opt = _simple_optimizer()
        assert build_scheduler(opt, scheduler_type="none") is None

    def test_returns_dict(self):
        opt = _simple_optimizer()
        cfg = build_scheduler(opt, scheduler_type="ReduceLROnPlateau")
        assert isinstance(cfg, dict)

    def test_dict_has_scheduler_key(self):
        opt = _simple_optimizer()
        cfg = build_scheduler(opt, scheduler_type="ReduceLROnPlateau")
        assert cfg is not None
        assert "scheduler" in cfg

    def test_plateau_dict_has_monitor(self):
        opt = _simple_optimizer()
        cfg = build_scheduler(opt, scheduler_type="ReduceLROnPlateau", monitor="val_auc")
        assert cfg is not None
        assert cfg["monitor"] == "val_auc"

    def test_default_interval_is_epoch(self):
        opt = _simple_optimizer()
        cfg = build_scheduler(opt, scheduler_type="ReduceLROnPlateau")
        assert cfg is not None
        assert cfg["interval"] == "epoch"

    def test_custom_interval(self):
        opt = _simple_optimizer()
        cfg = build_scheduler(opt, scheduler_type="ReduceLROnPlateau", interval="step")
        assert cfg is not None
        assert cfg["interval"] == "step"

    def test_custom_frequency(self):
        opt = _simple_optimizer()
        cfg = build_scheduler(opt, scheduler_type="ReduceLROnPlateau", frequency=2)
        assert cfg is not None
        assert cfg["frequency"] == 2

    def test_lr_factor_forwarded_to_plateau(self):
        opt = _simple_optimizer()
        cfg = build_scheduler(opt, scheduler_type="ReduceLROnPlateau", lr_factor=0.5)
        assert cfg is not None
        sched = cfg["scheduler"]
        assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert sched.factor == pytest.approx(0.5)

    def test_lr_patience_forwarded_to_plateau(self):
        opt = _simple_optimizer()
        cfg = build_scheduler(opt, scheduler_type="ReduceLROnPlateau", lr_patience=7)
        assert cfg is not None
        sched = cfg["scheduler"]
        assert sched.patience == 7

    def test_mode_forwarded_to_plateau(self):
        opt = _simple_optimizer()
        cfg = build_scheduler(opt, scheduler_type="ReduceLROnPlateau", mode="max")
        assert cfg is not None
        sched = cfg["scheduler"]
        assert sched.mode == "max"

    def test_scheduler_kwargs_override_defaults(self):
        opt = _simple_optimizer()
        cfg = build_scheduler(
            opt,
            scheduler_type="ReduceLROnPlateau",
            lr_factor=0.1,
            scheduler_kwargs={"factor": 0.9},
        )
        assert cfg is not None
        sched = cfg["scheduler"]
        assert sched.factor == pytest.approx(0.9)

    def test_unknown_scheduler_raises(self):
        opt = _simple_optimizer()
        with pytest.raises(InvalidParamError):
            build_scheduler(opt, scheduler_type="FakeScheduler")

    def test_steplr_has_no_monitor_key(self):
        opt = _simple_optimizer()
        cfg = build_scheduler(
            opt,
            scheduler_type="StepLR",
            scheduler_kwargs={"step_size": 10},
        )
        assert cfg is not None
        assert "monitor" not in cfg

    def test_steplr_instance_type(self):
        opt = _simple_optimizer()
        cfg = build_scheduler(
            opt,
            scheduler_type="StepLR",
            scheduler_kwargs={"step_size": 5},
        )
        assert cfg is not None
        assert isinstance(cfg["scheduler"], torch.optim.lr_scheduler.StepLR)


# ---------------------------------------------------------------------------
# build_default_task_loss
# ---------------------------------------------------------------------------


class TestBuildDefaultTaskLoss:
    def test_regression_returns_mse(self):
        from deeptab.training.losses import build_default_task_loss

        loss = build_default_task_loss(num_classes=1)
        assert isinstance(loss, nn.MSELoss)

    def test_binary_returns_bce(self):
        from deeptab.training.losses import build_default_task_loss

        loss = build_default_task_loss(num_classes=2)
        assert isinstance(loss, nn.BCEWithLogitsLoss)

    def test_multiclass_returns_ce(self):
        from deeptab.training.losses import build_default_task_loss

        loss = build_default_task_loss(num_classes=5)
        assert isinstance(loss, nn.CrossEntropyLoss)

    def test_lss_returns_none(self):
        from deeptab.training.losses import build_default_task_loss

        assert build_default_task_loss(num_classes=1, lss=True) is None

    def test_lss_binary_returns_none(self):
        from deeptab.training.losses import build_default_task_loss

        assert build_default_task_loss(num_classes=2, lss=True) is None

"""Tests for the hardware-detection utilities in :mod:`deeptab.core.hardware`.

These verify the structured report shape, that the helpers never raise on a
machine without a GPU, and that ``detailed=True`` adds the expected keys.
"""

from __future__ import annotations

import torch

from deeptab import print_hardware_info
from deeptab.core.hardware import get_hardware_info


def test_summary_has_expected_top_level_keys() -> None:
    info = get_hardware_info()
    assert set(info) == {"platform", "cpu", "cuda", "mps", "recommended_accelerator"}


def test_platform_and_cpu_fields_are_populated() -> None:
    info = get_hardware_info()
    assert info["platform"]["torch"] == torch.__version__
    assert isinstance(info["cpu"]["logical_cores"], int)
    assert info["cpu"]["logical_cores"] >= 1


def test_availability_flags_match_torch() -> None:
    info = get_hardware_info()
    assert info["cuda"]["available"] == torch.cuda.is_available()
    assert info["cuda"]["device_count"] == (torch.cuda.device_count() if torch.cuda.is_available() else 0)
    assert isinstance(info["mps"]["available"], bool)


def test_recommended_accelerator_is_consistent() -> None:
    info = get_hardware_info()
    accel = info["recommended_accelerator"]
    assert accel in {"cuda", "mps", "cpu"}
    if info["cuda"]["available"]:
        assert accel == "cuda"
    elif info["mps"]["available"]:
        assert accel == "mps"
    else:
        assert accel == "cpu"


def test_detailed_adds_build_versions() -> None:
    info = get_hardware_info(detailed=True)
    assert "built_version" in info["cuda"]
    assert "cudnn_version" in info["cuda"]
    assert "processor" in info["platform"]
    assert info["platform"]["executable"]


def test_detailed_cuda_devices_carry_memory_fields() -> None:
    info = get_hardware_info(detailed=True)
    for device in info["cuda"]["devices"]:
        assert {"total_memory_gb", "free_memory_gb", "compute_capability"} <= set(device)


def test_print_does_not_raise(capsys) -> None:
    print_hardware_info()
    print_hardware_info(detailed=True)
    out = capsys.readouterr().out
    assert "DeepTab hardware report" in out
    assert "Recommended accelerator:" in out

from __future__ import annotations

import os
import platform
import sys
from typing import Any

import torch

__all__ = ["get_hardware_info", "print_hardware_info"]


def _mps_available() -> bool:
    """Return ``True`` when the Apple Silicon MPS backend is usable."""
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is None:
        return False
    try:
        return bool(mps.is_available())
    except Exception:  # pragma: no cover - defensive, backend should not raise
        return False


def _mps_built() -> bool:
    """Return ``True`` when this PyTorch build includes the MPS backend."""
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is None or not hasattr(mps, "is_built"):
        return False
    try:
        return bool(mps.is_built())
    except Exception:  # pragma: no cover - defensive
        return False


def _recommended_accelerator(cuda_available: bool, mps_available: bool) -> str:
    """Pick the accelerator DeepTab would use by default."""
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def _cuda_devices(detailed: bool) -> list[dict[str, Any]]:
    """Return one entry per visible CUDA device."""
    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        entry: dict[str, Any] = {
            "index": index,
            "name": torch.cuda.get_device_name(index),
        }
        if detailed:
            major, minor = torch.cuda.get_device_capability(index)
            props = torch.cuda.get_device_properties(index)
            free, _total = torch.cuda.mem_get_info(index)
            entry.update(
                {
                    "compute_capability": f"{major}.{minor}",
                    "multi_processor_count": props.multi_processor_count,
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    "free_memory_gb": round(free / 1024**3, 2),
                }
            )
        devices.append(entry)
    return devices


def _mps_detail() -> dict[str, Any]:
    """Return MPS memory counters when the build exposes them."""
    detail: dict[str, Any] = {}
    mps = getattr(torch, "mps", None)
    if mps is None:
        return detail
    if hasattr(mps, "current_allocated_memory"):
        detail["allocated_memory_gb"] = round(mps.current_allocated_memory() / 1024**3, 3)
    if hasattr(mps, "driver_allocated_memory"):
        detail["driver_allocated_memory_gb"] = round(mps.driver_allocated_memory() / 1024**3, 3)
    return detail


def get_hardware_info(detailed: bool = False) -> dict[str, Any]:
    """Return the compute hardware DeepTab can see on this machine.

    The report covers the CPU, NVIDIA CUDA GPUs, and the Apple Silicon MPS
    backend, plus the ``accelerator`` value DeepTab would pick by default. It is
    safe to call at import time and never raises on machines without a GPU.

    Parameters
    ----------
    detailed : bool, default=False
        When ``False``, return a compact summary: core count, device counts,
        availability flags, and the recommended accelerator. When ``True``,
        also include per-GPU memory, compute capability, multiprocessor count,
        the CUDA / cuDNN versions PyTorch was built against, and MPS memory
        counters where the build exposes them.

    Returns
    -------
    dict
        A nested dictionary with the keys ``platform``, ``cpu``, ``cuda``,
        ``mps``, and ``recommended_accelerator``.

    Examples
    --------
    >>> from deeptab import get_hardware_info
    >>> info = get_hardware_info()
    >>> sorted(info)
    ['cpu', 'cuda', 'mps', 'platform', 'recommended_accelerator']
    >>> info["cpu"]["logical_cores"] >= 1
    True
    """
    cuda_available = torch.cuda.is_available()
    mps_available = _mps_available()

    info: dict[str, Any] = {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "cpu": {
            "logical_cores": os.cpu_count(),
        },
        "cuda": {
            "available": cuda_available,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "devices": _cuda_devices(detailed) if cuda_available else [],
        },
        "mps": {
            "available": mps_available,
            "built": _mps_built(),
        },
        "recommended_accelerator": _recommended_accelerator(cuda_available, mps_available),
    }

    if detailed:
        info["platform"]["processor"] = platform.processor()
        info["platform"]["python_implementation"] = platform.python_implementation()
        info["platform"]["executable"] = sys.executable
        info["cuda"]["built_version"] = torch.version.cuda
        cudnn = getattr(torch.backends, "cudnn", None)
        info["cuda"]["cudnn_version"] = cudnn.version() if cuda_available and cudnn is not None else None
        if mps_available:
            info["mps"].update(_mps_detail())

    return info


def print_hardware_info(detailed: bool = False) -> None:
    """Print :func:`get_hardware_info` as a readable report.

    Parameters
    ----------
    detailed : bool, default=False
        Forwarded to :func:`get_hardware_info`. When ``True``, the report adds
        per-GPU memory and capability, build versions, and MPS memory counters.

    Examples
    --------
    >>> from deeptab import print_hardware_info
    >>> print_hardware_info()  # doctest: +SKIP
    DeepTab hardware report
    -----------------------
    Platform: Darwin (arm64), Python 3.11.8, PyTorch 2.2.0
    ...
    """
    info = get_hardware_info(detailed=detailed)
    plat = info["platform"]
    cpu = info["cpu"]
    cuda = info["cuda"]
    mps = info["mps"]

    lines = [
        "DeepTab hardware report",
        "-----------------------",
        f"Platform: {plat['system']} ({plat['machine']}), Python {plat['python']}, PyTorch {plat['torch']}",
        f"CPU: {cpu['logical_cores']} logical cores",
    ]

    if cuda["available"]:
        lines.append(f"CUDA: available, {cuda['device_count']} device(s)")
        for device in cuda["devices"]:
            if detailed:
                lines.append(
                    f"  [{device['index']}] {device['name']}: "
                    f"{device['total_memory_gb']} GB total, "
                    f"{device['free_memory_gb']} GB free, "
                    f"compute capability {device['compute_capability']}, "
                    f"{device['multi_processor_count']} SMs"
                )
            else:
                lines.append(f"  [{device['index']}] {device['name']}")
        if detailed:
            lines.append(f"  Built against CUDA {cuda['built_version']}, cuDNN {cuda['cudnn_version']}")
    else:
        lines.append("CUDA: not available")

    if mps["available"]:
        mps_line = "MPS (Apple Silicon): available"
        if detailed and "driver_allocated_memory_gb" in mps:
            allocated = mps["driver_allocated_memory_gb"]
            if allocated > 0:
                mps_line += f", {allocated} GB driver-allocated"
            else:
                mps_line += ", none allocated yet"
        lines.append(mps_line)
    else:
        lines.append(f"MPS (Apple Silicon): not available (built: {mps['built']})")

    lines.append(f"Recommended accelerator: {info['recommended_accelerator']}")

    print("\n".join(lines))

"""Global-seed utilities for reproducible training.

Calling :func:`set_seed` before training seeds every RNG layer that DeepTab
touches (Python built-in ``random``, NumPy, PyTorch CPU/CUDA/MPS) and
optionally enables PyTorch's full deterministic mode.

Platform support
----------------
The helper is designed to work identically on Windows, macOS, and Linux,
and on CPU, CUDA (NVIDIA), and MPS (Apple Silicon) devices.

* **CPU** — always seeded via ``torch.manual_seed``.
* **CUDA** — seeded via ``torch.cuda.manual_seed_all`` when
  ``torch.cuda.is_available()`` is ``True``; cuDNN determinism flags are
  also set in that case.
* **MPS** — seeded via ``torch.mps.manual_seed`` when the MPS backend is
  available (PyTorch ≥ 1.12, macOS 12.3+).
* **PYTHONHASHSEED** — written to ``os.environ`` so that child processes
  (e.g. DataLoader workers) inherit a deterministic hash seed.  Note that
  changing ``PYTHONHASHSEED`` in the *current* process has no effect on the
  hash values already computed by that process; restart the interpreter if
  you need hash-determinism from the very first import.

Usage
-----
Pass ``random_state`` to any estimator constructor to have seeding done
automatically on every :meth:`fit` call::

    model = MLPRegressor(random_state=42)
    model.fit(X_train, y_train)

For manual control, call :func:`set_seed` directly or use the
:func:`seed_context` context manager::

    from deeptab.core.reproducibility import set_seed, seed_context

    set_seed(42)
    # … all subsequent calls share this seed …

    with seed_context(42):
        model.fit(X_train, y_train)
"""

from __future__ import annotations

import os
import random
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np
import torch

__all__ = ["seed_context", "set_seed"]


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed every RNG layer used by DeepTab.

    Sets the following in order so that a single integer reproduces the full
    training pipeline — data splitting, weight initialisation, dropout masks,
    and DataLoader shuffling.

    Seeded layers, in order:

    * ``random.seed(seed)`` — Python built-in RNG.
    * ``os.environ["PYTHONHASHSEED"]`` — propagated to child processes
      (DataLoader workers, subprocesses).  Has no effect on hash values
      already computed in the *current* process.
    * ``numpy.random.seed(seed)`` — NumPy legacy RNG used by preprocessing.
    * ``torch.manual_seed(seed)`` — PyTorch CPU RNG (all platforms).
    * ``torch.cuda.manual_seed_all(seed)`` — all CUDA device RNGs
      (only when ``torch.cuda.is_available()``).
    * ``torch.backends.cudnn.deterministic = True`` and
      ``torch.backends.cudnn.benchmark = False`` — force deterministic
      cuDNN kernels and disable auto-tuning
      (only when ``torch.cuda.is_available()``).
    * ``torch.mps.manual_seed(seed)`` — Apple Silicon MPS RNG
      (only when ``torch.backends.mps.is_available()``).

    Parameters
    ----------
    seed : int
        Non-negative integer seed.  Must be in the range ``[0, 2**32 - 1]``.
    deterministic : bool, optional
        When ``True``, additionally call
        ``torch.use_deterministic_algorithms(True)``.  This forces every
        backend (CUDA, MPS, CPU) to use a deterministic kernel where one
        exists, and raises ``RuntimeError`` for ops with no deterministic
        variant.  Defaults to ``False``.

    Examples
    --------
    >>> from deeptab.core.reproducibility import set_seed
    >>> set_seed(42)
    >>> import torch
    >>> t1 = torch.randn(5)
    >>> set_seed(42)
    >>> t2 = torch.randn(5)
    >>> (t1 == t2).all().item()
    True
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    # Python / NumPy
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # PyTorch CPU (always present)
    torch.manual_seed(seed)

    # CUDA — guard so the call is a true no-op on CPU-only and MPS-only hosts
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # MPS (Apple Silicon) — available from PyTorch 1.12 / macOS 12.3+
    if hasattr(torch, "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


@contextmanager
def seed_context(seed: int, *, deterministic: bool = False) -> Generator[None, None, None]:
    """Context manager that seeds all RNGs on entry.

    Equivalent to calling :func:`set_seed` but expressed as a ``with``
    statement for locally scoped seeding.

    .. note::
        This does **not** restore the previous RNG state on exit.  The new
        seed takes effect for the entire remainder of the process unless
        overridden by another :func:`set_seed` call.  Restoring global RNG
        state across multiple frameworks is fragile and not recommended for
        training pipelines.

    Parameters
    ----------
    seed : int
        Non-negative integer seed.
    deterministic : bool, optional
        Passed through to :func:`set_seed`.

    Examples
    --------
    >>> from deeptab.core.reproducibility import seed_context
    >>> with seed_context(42):
    ...     model.fit(X_train, y_train)
    """
    set_seed(seed, deterministic=deterministic)
    yield

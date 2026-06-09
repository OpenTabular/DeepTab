"""LR-scheduler registry and Lightning-compatible factory for DeepTab.

Background
----------
Previously ``TaskModel.configure_optimizers`` hard-coded
``ReduceLROnPlateau`` with ``mode='min'`` and ``monitor='val_loss'``.
That is wrong whenever the user sets ``TrainerConfig(mode='max')`` or
``TrainerConfig(monitor='val_auc')`` because early stopping then follows
the correct metric/direction while the scheduler watches a different,
possibly opposing metric.

What this module provides
-------------------------
1. A registry of standard PyTorch schedulers under predictable lowercase
   names (see *Built-in schedulers* below).
2. :func:`build_scheduler` — returns a Lightning-compatible dict (or
   ``None`` when the scheduler is disabled).
3. Correct forwarding of ``mode`` and ``monitor`` to ``ReduceLROnPlateau``
   so both early stopping and the scheduler track the same metric.
4. Backward-compatible defaults: ``ReduceLROnPlateau`` remains the default
   and the legacy ``lr_patience`` / ``lr_factor`` fields still take effect.

Built-in schedulers
-------------------
All standard ``torch.optim.lr_scheduler`` classes are registered at import
time under their original (case-insensitive) names::

    constantlr, cosineannealinglr, cosineannealingwarmrestarts,
    cycliclr, exponentiallr, linearlr, multisteplr, onecyclelr,
    reducelronplateau, sequentiallr, steplr

Basic usage
-----------
:func:`build_scheduler` is called automatically by
``TaskModel.configure_optimizers``.  You rarely need it directly, but it is
useful when building custom training loops::

    from deeptab.training.schedulers import build_scheduler
    import torch.nn as nn, torch.optim as optim

    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Default: ReduceLROnPlateau watching val_loss (minimised)
    sched_cfg = build_scheduler(optimizer)

    # Cosine annealing (no monitor needed)
    sched_cfg = build_scheduler(
        optimizer,
        scheduler_type="CosineAnnealingLR",
        scheduler_kwargs={"T_max": 100},
    )

    # Disabled
    sched_cfg = build_scheduler(optimizer, scheduler_type=None)
    # sched_cfg is None

Using via TrainerConfig
-----------------------
The most common configuration path is through
:class:`~deeptab.configs.TrainerConfig`::

    from deeptab.configs import TrainerConfig

    # Switch to cosine annealing
    tc = TrainerConfig(
        scheduler_type="CosineAnnealingLR",
        scheduler_kwargs={"T_max": 50},
    )

    # Maximise AUC (early stopping AND scheduler aligned)
    tc = TrainerConfig(
        monitor="val_auc",
        mode="max",
        scheduler_type="ReduceLROnPlateau",
        lr_patience=5,
        lr_factor=0.5,
    )

Registering a custom scheduler
-------------------------------
::

    from deeptab.training.schedulers import register_scheduler

    register_scheduler("warmup_cosine", MyWarmupCosineScheduler)
    tc = TrainerConfig(scheduler_type="warmup_cosine")

See Also
--------
:mod:`deeptab.training.optimizers` : Companion optimizer registry.
:class:`~deeptab.configs.TrainerConfig` : Config object that drives
    ``scheduler_type``, ``scheduler_kwargs``, ``monitor``, ``mode``,
    ``lr_patience``, and ``lr_factor``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.optim.lr_scheduler as _lr_sched

__all__ = [
    "available_schedulers",
    "build_scheduler",
    "get_scheduler",
    "register_scheduler",
]

_SCHEDULER_REGISTRY: dict[str, type] = {}

# Schedulers that need a 'monitor' key in the Lightning dict
_PLATEAU_SCHEDULERS: frozenset[str] = frozenset({"reducelronplateau"})

# Schedulers where 'mode' is a valid constructor kwarg
_SCHEDULERS_WITH_MODE: frozenset[str] = frozenset({"reducelronplateau"})


def _register_torch_defaults() -> None:
    names = [
        "ReduceLROnPlateau",
        "StepLR",
        "MultiStepLR",
        "ExponentialLR",
        "CosineAnnealingLR",
        "CosineAnnealingWarmRestarts",
        "OneCycleLR",
        "CyclicLR",
        "ConstantLR",
        "LinearLR",
        "SequentialLR",
    ]
    for name in names:
        cls = getattr(_lr_sched, name, None)
        if cls is not None:
            _SCHEDULER_REGISTRY[name.lower()] = cls


_register_torch_defaults()


def register_scheduler(name: str, factory: type, *, override: bool = False) -> None:
    """Register a custom LR scheduler under a string name.

    Once registered, the scheduler is available everywhere that accepts a
    ``scheduler_type`` string — including
    :class:`~deeptab.configs.TrainerConfig` and :func:`build_scheduler`.

    Parameters
    ----------
    name : str
        Case-insensitive lookup key.  Stored as lowercase internally so
        ``"StepLR"`` and ``"steplr"`` refer to the same entry.
    factory : type
        A scheduler class accepted by PyTorch / Lightning, i.e. any class
        whose constructor takes ``(optimizer, **kwargs)`` and whose instances
        expose a ``step()`` method.
    override : bool, default=False
        Allow overriding an existing registration.  Set to ``True`` when you
        intentionally want to replace a built-in or previously registered
        scheduler.

    Raises
    ------
    ValueError
        If *name* is already registered and *override* is ``False``.

    Examples
    --------
    >>> from deeptab.training.schedulers import register_scheduler
    >>> register_scheduler("warmup_cosine", MyWarmupCosineScheduler)
    >>> from deeptab.configs import TrainerConfig
    >>> tc = TrainerConfig(scheduler_type="warmup_cosine")

    Notes
    -----
    Registration is **process-global**.  In distributed training (DDP) each
    worker imports independently, so register your scheduler in every worker
    or in a module that is imported at the top of your training script.

    See Also
    --------
    :func:`available_schedulers` : Inspect all registered names.
    :func:`get_scheduler` : Retrieve a class by name without instantiating it.
    """
    key = name.lower()
    if key in _SCHEDULER_REGISTRY and not override:
        raise ValueError(f"Scheduler {name!r} is already registered. Pass override=True to replace it.")
    _SCHEDULER_REGISTRY[key] = factory


def get_scheduler(name: str) -> type:
    """Return the scheduler class for the given name (case-insensitive).

    This is a low-level look-up used internally by :func:`build_scheduler`.
    Most users should call :func:`build_scheduler` directly.

    Parameters
    ----------
    name : str
        Scheduler name as registered.  Case-insensitive (``"StepLR"``,
        ``"steplr"``, and ``"STEPLR"`` all work).

    Returns
    -------
    type
        The registered scheduler class.

    Raises
    ------
    ~deeptab.core.exceptions.InvalidParamError
        If *name* is not in the registry.  The error message lists all
        available names.

    Examples
    --------
    >>> from deeptab.training.schedulers import get_scheduler
    >>> import torch.optim as optim, torch.nn as nn
    >>> cls = get_scheduler("StepLR")
    >>> model = nn.Linear(4, 1)
    >>> opt = optim.Adam(model.parameters(), lr=1e-3)
    >>> sched = cls(opt, step_size=10, gamma=0.5)

    >>> get_scheduler("NotAScheduler")  # raises InvalidParamError

    See Also
    --------
    :func:`available_schedulers` : List all valid names.
    :func:`build_scheduler` : Higher-level factory returning a Lightning dict.
    """
    key = name.lower()
    if key not in _SCHEDULER_REGISTRY:
        from deeptab.core.exceptions import invalid_param_error

        raise invalid_param_error(
            "TrainerConfig",
            "scheduler_type",
            name,
            "must be a registered scheduler name",
            available_schedulers(),
        )
    return _SCHEDULER_REGISTRY[key]


def available_schedulers() -> list[str]:
    """Return a sorted list of registered scheduler names (lowercase).

    Returns
    -------
    list of str
        Every scheduler currently in the registry, in alphabetical order.
        All names are lowercase regardless of the capitalisation used during
        registration.

    Examples
    --------
    >>> from deeptab.training.schedulers import available_schedulers
    >>> available_schedulers()          # doctest: +NORMALIZE_WHITESPACE
    ['constantlr', 'cosineannealinglr', 'cosineannealingwarmrestarts',
     'cycliclr', 'exponentiallr', 'linearlr', 'multisteplr', 'onecyclelr',
     'reducelronplateau', 'sequentiallr', 'steplr']

    Guard before registering a custom scheduler::

        if "warmup_cosine" not in available_schedulers():
            register_scheduler("warmup_cosine", MyWarmupCosineScheduler)
    """
    return sorted(_SCHEDULER_REGISTRY.keys())


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_type: str | None = "ReduceLROnPlateau",
    scheduler_kwargs: dict[str, Any] | None = None,
    lr_factor: float = 0.1,
    lr_patience: int = 10,
    monitor: str = "val_loss",
    mode: str = "min",
    interval: str = "epoch",
    frequency: int = 1,
) -> dict[str, Any] | None:
    """Build a Lightning-compatible scheduler configuration dict.

    Returns a dict in the format expected by PyTorch Lightning's
    ``configure_optimizers`` return value, or ``None`` when the scheduler is
    disabled.  The dict is passed directly as the ``lr_scheduler`` value in
    the ``{"optimizer": ..., "lr_scheduler": ...}`` return of
    ``configure_optimizers``.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer instance to attach the scheduler to.
    scheduler_type : str or None, default="ReduceLROnPlateau"
        Scheduler name (case-insensitive) or ``None`` / ``"none"`` to
        disable the scheduler entirely.  Use :func:`available_schedulers`
        for a full list of built-in names or :func:`register_scheduler` to
        add your own.
    scheduler_kwargs : dict or None, default=None
        Explicit keyword arguments forwarded to the scheduler constructor.
        For ``ReduceLROnPlateau``, ``"factor"`` and ``"patience"`` are
        synthesised from *lr_factor* / *lr_patience* when absent here —
        explicit values in *scheduler_kwargs* always take precedence.
    lr_factor : float, default=0.1
        Backward-compatibility field used as ``factor`` for
        ``ReduceLROnPlateau`` when *scheduler_kwargs* does not specify it.
        Ignored for all other schedulers unless included in
        *scheduler_kwargs*.
    lr_patience : int, default=10
        Backward-compatibility field used as ``patience`` for
        ``ReduceLROnPlateau`` when *scheduler_kwargs* does not specify it.
        Ignored for all other schedulers unless included in
        *scheduler_kwargs*.
    monitor : str, default="val_loss"
        Metric name for the Lightning scheduler dict.  Also passed as the
        ``mode`` companion to ``ReduceLROnPlateau`` via *mode*.
        Should match the ``monitor`` field of
        :class:`~deeptab.configs.TrainerConfig` exactly.
    mode : str, default="min"
        ``"min"`` or ``"max"``.  Passed to ``ReduceLROnPlateau`` to align
        it with the early-stopping direction set in ``TrainerConfig``.
        Ignored for schedulers that do not accept ``mode``.
    interval : str, default="epoch"
        Lightning scheduling granularity: ``"epoch"`` (step after every
        validation epoch) or ``"step"`` (step after every training step).
    frequency : int, default=1
        How many *interval* units to wait between scheduler steps.
        ``frequency=2`` with ``interval="epoch"`` steps every 2 epochs.

    Returns
    -------
    dict or None
        A Lightning scheduler config dict with keys ``"scheduler"``,
        ``"interval"``, ``"frequency"``, and (for plateau schedulers)
        ``"monitor"``.  Returns ``None`` when *scheduler_type* is ``None``
        or ``"none"``.

    Raises
    ------
    ~deeptab.core.exceptions.InvalidParamError
        If *scheduler_type* is a non-``None`` string that is not registered.

    Examples
    --------
    **Default ReduceLROnPlateau** (backward-compatible)::

        from deeptab.training.schedulers import build_scheduler
        import torch.nn as nn, torch.optim as optim

        model = nn.Linear(10, 1)
        opt = optim.Adam(model.parameters(), lr=1e-3)

        cfg = build_scheduler(opt)
        # cfg["monitor"] == "val_loss"
        # cfg["scheduler"].patience == 10

    **Align with a maximise-AUC TrainerConfig**::

        cfg = build_scheduler(
            opt,
            scheduler_type="ReduceLROnPlateau",
            monitor="val_auc",
            mode="max",
            lr_patience=5,
            lr_factor=0.5,
        )
        # cfg["scheduler"].mode == "max"
        # cfg["monitor"] == "val_auc"

    **Cosine annealing (no monitor needed)**::

        cfg = build_scheduler(
            opt,
            scheduler_type="CosineAnnealingLR",
            scheduler_kwargs={"T_max": 100, "eta_min": 1e-6},
        )
        # "monitor" key is absent from cfg

    **StepLR at training-step granularity**::

        cfg = build_scheduler(
            opt,
            scheduler_type="StepLR",
            scheduler_kwargs={"step_size": 500, "gamma": 0.5},
            interval="step",
            frequency=1,
        )

    **Disable the scheduler**::

        cfg = build_scheduler(opt, scheduler_type=None)
        assert cfg is None

    Notes
    -----
    ``ReduceLROnPlateau`` is the **only** built-in scheduler that requires
    Lightning to feed back the monitored metric value at each step.
    :func:`build_scheduler` detects this automatically and adds
    ``"monitor"`` to the returned dict.  All other schedulers step
    unconditionally based on ``interval`` / ``frequency``.

    The precedence chain for ``ReduceLROnPlateau`` kwargs is:

    1. Explicit keys in *scheduler_kwargs* (highest priority).
    2. *lr_factor* / *lr_patience* for ``"factor"`` / ``"patience"``.
    3. PyTorch defaults (lowest priority).

    See Also
    --------
    :func:`register_scheduler` : Register a custom scheduler class.
    :func:`available_schedulers` : List all registered names.
    :func:`build_optimizer` : Companion optimizer factory.
    :class:`~deeptab.configs.TrainerConfig` : Config object that wires
        ``scheduler_type``, ``scheduler_kwargs``, ``monitor``, ``mode``,
        ``lr_patience``, ``lr_factor``, ``scheduler_interval``, and
        ``scheduler_frequency`` into :class:`~deeptab.training.TaskModel`.
    """
    if scheduler_type is None or scheduler_type.lower() == "none":
        return None

    key = scheduler_type.lower()
    cls = get_scheduler(scheduler_type)

    kwargs: dict[str, Any] = {}

    # Inject mode for schedulers that accept it
    if key in _SCHEDULERS_WITH_MODE:
        kwargs["mode"] = mode

    # Synthesise factor/patience for ReduceLROnPlateau from legacy fields
    if key == "reducelronplateau":
        kwargs.setdefault("factor", lr_factor)
        kwargs.setdefault("patience", lr_patience)

    # Caller-provided kwargs take precedence
    if scheduler_kwargs:
        kwargs.update(scheduler_kwargs)

    scheduler_instance = cls(optimizer, **kwargs)

    config: dict[str, Any] = {
        "scheduler": scheduler_instance,
        "interval": interval,
        "frequency": frequency,
    }

    # Plateau schedulers need Lightning to pass the monitored value in
    if key in _PLATEAU_SCHEDULERS:
        config["monitor"] = monitor

    return config

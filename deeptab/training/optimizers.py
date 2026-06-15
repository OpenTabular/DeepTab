"""Optimizer registry and factory for DeepTab training.

See :func:`build_optimizer` (the primary entry point), :func:`register_optimizer`
and :func:`unregister_optimizer` (extension points), and
:func:`available_optimizers` (the list of built-in names) for usage details.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "available_optimizers",
    "build_optimizer",
    "build_parameter_groups",
    "get_optimizer",
    "normalize_optimizer_kwargs",
    "register_optimizer",
    "unregister_optimizer",
]

# Registry: lowercase key -> optimizer class
_OPTIMIZER_REGISTRY: dict[str, type[torch.optim.Optimizer]] = {}

# Names registered by DeepTab itself at import time.  These are protected:
# end-users may override them (intentionally, with ``override=True``) but may
# not remove them via :func:`unregister_optimizer`.
_BUILTIN_OPTIMIZERS: frozenset[str] = frozenset()


def _register_torch_defaults() -> None:
    global _BUILTIN_OPTIMIZERS
    names = [
        "Adadelta",
        "Adagrad",
        "Adam",
        "AdamW",
        "Adamax",
        "ASGD",
        "LBFGS",
        "NAdam",
        "RAdam",
        "RMSprop",
        "Rprop",
        "SGD",
        "SparseAdam",
    ]
    for name in names:
        cls = getattr(torch.optim, name, None)
        if cls is not None:
            _OPTIMIZER_REGISTRY[name.lower()] = cls
    _BUILTIN_OPTIMIZERS = frozenset(_OPTIMIZER_REGISTRY.keys())


_register_torch_defaults()


def register_optimizer(
    name: str,
    factory: type[torch.optim.Optimizer],
    *,
    override: bool = False,
) -> None:
    """Register a custom optimizer under a string name.

    Once registered, the optimizer is available everywhere that accepts an
    ``optimizer_type`` string — including :class:`~deeptab.configs.TrainerConfig`
    and :func:`build_optimizer`.

    Parameters
    ----------
    name : str
        Case-insensitive lookup key (e.g. ``"muon"``).  Stored as lowercase
        internally so ``"Adam"`` and ``"adam"`` refer to the same entry.
    factory : type[torch.optim.Optimizer]
        An optimizer class or any callable that accepts ``(params, **kwargs)``
        and returns a ``torch.optim.Optimizer`` instance.
    override : bool, default=False
        Allow overriding an existing registration.  Defaults to ``False`` to
        prevent accidental shadowing of built-in names.  Set to ``True`` when
        you intentionally want to replace a registered class.

    Raises
    ------
    ValueError
        If *name* is already registered and *override* is ``False``.

    Examples
    --------
    Register a third-party optimizer and use it via ``TrainerConfig``:

    >>> from deeptab.training.optimizers import register_optimizer
    >>> import torch.optim as optim
    >>> register_optimizer("sgdm", optim.SGD)
    >>> from deeptab.configs import TrainerConfig
    >>> tc = TrainerConfig(optimizer_type="sgdm", lr=0.01)

    Replace an existing entry (e.g. swap Adam for a custom variant):

    >>> register_optimizer("adam", MyCustomAdam, override=True)

    Notes
    -----
    Registration is **process-global** — it applies to the entire Python
    process.  In multi-process training (DDP) each worker runs its own
    import, so you must call ``register_optimizer`` in every worker, or
    (more robustly) in a module that is imported at the top of your
    training script.

    See Also
    --------
    :func:`available_optimizers` : Inspect all registered names.
    :func:`get_optimizer` : Retrieve a class by name without building an instance.
    """
    key = name.lower()
    if key in _OPTIMIZER_REGISTRY and not override:
        raise ValueError(f"Optimizer {name!r} is already registered. Pass override=True to replace it.")
    _OPTIMIZER_REGISTRY[key] = factory


def unregister_optimizer(name: str, *, missing_ok: bool = False) -> None:
    """Remove a **user-registered** optimizer from the registry.

    Use this to undo a previous :func:`register_optimizer` call — for example
    to free up a name or to reset state between experiments.  Only optimizers
    that you registered yourself can be removed; the optimizers that ship with
    DeepTab are protected and cannot be unregistered (removing them would break
    every estimator in the process).

    Parameters
    ----------
    name : str
        Case-insensitive name of the optimizer to remove.
    missing_ok : bool, default=False
        If ``True``, silently return when *name* is not registered instead of
        raising.  Useful for idempotent teardown (e.g. in notebooks or tests
        that may run more than once).

    Raises
    ------
    ValueError
        If *name* is a built-in DeepTab optimizer.  Built-ins are protected
        and can only be replaced via ``register_optimizer(..., override=True)``,
        never removed.
    ~deeptab.core.exceptions.InvalidParamError
        If *name* is not registered and *missing_ok* is ``False``.  The error
        message lists the available names.

    Examples
    --------
    >>> from deeptab.training.optimizers import register_optimizer, unregister_optimizer
    >>> import torch.optim as optim
    >>> register_optimizer("sgdm", optim.SGD)
    >>> unregister_optimizer("sgdm")
    >>> unregister_optimizer("sgdm", missing_ok=True)  # no error, already gone
    >>> unregister_optimizer("adam")  # raises ValueError: built-in, protected

    Notes
    -----
    Like registration, removal is **process-global**.

    See Also
    --------
    :func:`register_optimizer` : Add or replace an optimizer.
    :func:`available_optimizers` : Inspect the current registry.
    """
    key = name.lower()
    if key in _BUILTIN_OPTIMIZERS:
        raise ValueError(
            f"Optimizer {name!r} is a built-in DeepTab optimizer and cannot be unregistered. "
            "Built-ins can be replaced with register_optimizer(..., override=True) but not removed."
        )
    if key not in _OPTIMIZER_REGISTRY:
        if missing_ok:
            return
        from deeptab.core.exceptions import invalid_param_error

        raise invalid_param_error(
            "unregister_optimizer",
            "name",
            name,
            "must be a user-registered optimizer name",
            sorted(set(available_optimizers()) - _BUILTIN_OPTIMIZERS),
        )
    del _OPTIMIZER_REGISTRY[key]


def get_optimizer(name: str) -> type[torch.optim.Optimizer]:
    """Return the optimizer class for the given name (case-insensitive).

    This is a low-level look-up used internally by :func:`build_optimizer`.
    Most users should call :func:`build_optimizer` directly.

    Parameters
    ----------
    name : str
        Optimizer name as registered.  Case-insensitive (``"Adam"``,
        ``"adam"``, and ``"ADAM"`` all work).

    Returns
    -------
    type[torch.optim.Optimizer]
        The registered optimizer class.

    Raises
    ------
    ~deeptab.core.exceptions.InvalidParamError
        If *name* is not in the registry.  The error message lists all
        available names so the user can correct the typo immediately.

    Examples
    --------
    >>> from deeptab.training.optimizers import get_optimizer
    >>> import torch.nn as nn
    >>> cls = get_optimizer("AdamW")
    >>> model = nn.Linear(4, 1)
    >>> opt = cls(model.parameters(), lr=1e-3, weight_decay=1e-2)

    >>> get_optimizer("typo")  # raises InvalidParamError

    See Also
    --------
    :func:`available_optimizers` : List all valid names.
    :func:`build_optimizer` : Higher-level factory that also handles parameter
        grouping and kwargs normalisation.
    """
    key = name.lower()
    if key not in _OPTIMIZER_REGISTRY:
        from deeptab.core.exceptions import invalid_param_error

        raise invalid_param_error(
            "TrainerConfig",
            "optimizer_type",
            name,
            "must be a registered optimizer name",
            available_optimizers(),
        )
    return _OPTIMIZER_REGISTRY[key]


def available_optimizers() -> list[str]:
    """Return a sorted list of registered optimizer names (lowercase).

    Returns
    -------
    list of str
        Every optimizer currently in the registry, in alphabetical order.
        All names are lowercase regardless of the capitalisation used during
        registration.

    Examples
    --------
    >>> from deeptab.training.optimizers import available_optimizers
    >>> available_optimizers()                  # doctest: +NORMALIZE_WHITESPACE
    ['adadelta', 'adagrad', 'adam', 'adamax', 'adamw', 'asgd',
     'lbfgs', 'nadam', 'radam', 'rmsprop', 'rprop', 'sgd', 'sparseadam']

    Use this when unsure whether a custom optimizer has been registered::

        if "muon" not in available_optimizers():
            register_optimizer("muon", MuonOptimizer)
    """
    return sorted(_OPTIMIZER_REGISTRY.keys())


def normalize_optimizer_kwargs(optimizer_args: dict[str, Any] | None) -> dict[str, Any]:
    """Strip the legacy ``optimizer_`` prefix from optimizer kwargs.

    The legacy flat-kwargs API accepted keys like
    ``optimizer_betas=(0.9, 0.95)`` and stripped the prefix before forwarding
    them to the PyTorch constructor.  This helper centralises that behaviour
    and also handles ``None`` safely (previously a runtime crash in
    ``TaskModel.__init__``).

    Parameters
    ----------
    optimizer_args : dict or None
        Raw dict (possibly with ``optimizer_``-prefixed keys) or ``None``.
        Keys that do **not** start with ``"optimizer_"`` are silently dropped
        so that accidentally passing the full ``TrainerConfig`` dict is safe.

    Returns
    -------
    dict
        Cleaned kwargs ready to pass to ``optimizer_class(params, **kwargs)``.
        Returns an empty dict when *optimizer_args* is ``None`` or empty.

    Examples
    --------
    >>> from deeptab.training.optimizers import normalize_optimizer_kwargs
    >>> normalize_optimizer_kwargs({"optimizer_betas": (0.9, 0.95), "optimizer_eps": 1e-8})
    {'betas': (0.9, 0.95), 'eps': 1e-08}

    >>> normalize_optimizer_kwargs(None)
    {}

    >>> normalize_optimizer_kwargs({"lr": 1e-3})  # non-prefixed key is dropped
    {}

    Notes
    -----
    This function is called automatically by ``TaskModel.__init__``.  You
    only need to call it directly when building an optimizer outside of
    ``TaskModel``, e.g. in a custom training loop.
    """
    if not optimizer_args:
        return {}
    return {
        key.removeprefix("optimizer_"): value for key, value in optimizer_args.items() if key.startswith("optimizer_")
    }


def build_parameter_groups(
    module: nn.Module,
    *,
    weight_decay: float,
    no_weight_decay_for_bias_and_norm: bool = True,
) -> list[dict[str, Any]]:
    """Split module parameters into two groups for selective weight decay.

    Applying weight decay to bias vectors and normalisation-layer parameters
    is generally harmful:

    - **Bias terms** shift the activation distribution; regularising them
      competes with the optimiser's ability to find the correct offset.
    - **LayerNorm / BatchNorm scale & shift** parameters shrink toward zero
      when regularised, which breaks the normalisation invariant.

    This split is recommended whenever you use transformer-style architectures
    (``FTTransformer``, ``TabTransformer``) or any model with embedding layers.
    Enable it via ``TrainerConfig(no_weight_decay_for_bias_and_norm=True)``.

    Parameters
    ----------
    module : nn.Module
        The full model whose parameters are to be split
        (typically ``TaskModel.estimator``).
    weight_decay : float
        Weight decay coefficient applied to the *decay* group.
    no_weight_decay_for_bias_and_norm : bool, default=True
        When ``True``, bias parameters and parameters of
        :class:`~torch.nn.LayerNorm`, :class:`~torch.nn.BatchNorm1d`,
        :class:`~torch.nn.BatchNorm2d`, and :class:`~torch.nn.GroupNorm`
        layers are placed in a second group with ``weight_decay=0.0``.
        When ``False``, a single group containing all parameters is returned.

    Returns
    -------
    list of dict
        A list of PyTorch parameter-group dicts suitable for passing directly
        to any ``torch.optim`` constructor as the ``params`` argument.
        When *no_weight_decay_for_bias_and_norm* is ``True`` the list has
        exactly two elements; otherwise one.

    Examples
    --------
    >>> import torch.nn as nn, torch.optim as optim
    >>> from deeptab.training.optimizers import build_parameter_groups
    >>> model = nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16), nn.Linear(16, 1))
    >>> groups = build_parameter_groups(model, weight_decay=1e-4)
    >>> len(groups)  # decay group + no-decay group
    2
    >>> groups[1]["weight_decay"]
    0.0
    >>> opt = optim.AdamW(groups, lr=1e-3)  # weight_decay set per group

    Notes
    -----
    No parameter is ever duplicated between the two groups.  The function
    tracks parameter identity (``id(p)``) across all sub-modules, so shared
    parameters (e.g. tied embeddings) are assigned exactly once.

    References
    ----------
    Andrej Karpathy, *minGPT* — parameter grouping pattern:
    https://github.com/karpathy/minGPT

    See Also
    --------
    :func:`build_optimizer` : High-level factory that calls this function
        automatically when ``no_weight_decay_for_bias_and_norm=True``.
    """
    if not no_weight_decay_for_bias_and_norm:
        return [{"params": module.parameters(), "weight_decay": weight_decay}]

    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    no_decay_types = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)

    seen: set[int] = set()
    for mod in module.modules():
        for param_name, param in mod.named_parameters(recurse=False):
            if id(param) in seen:
                continue
            seen.add(id(param))
            if isinstance(mod, no_decay_types) or param_name.endswith("bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def build_optimizer(
    module_or_params: Any,
    *,
    optimizer_type: str = "Adam",
    lr: float = 1e-4,
    weight_decay: float = 1e-6,
    optimizer_kwargs: dict[str, Any] | None = None,
    no_weight_decay_for_bias_and_norm: bool = False,
) -> torch.optim.Optimizer:
    """Build and return a fully configured optimizer.

    This is the primary entry point of the optimizer registry.  It is called
    automatically by ``TaskModel.configure_optimizers`` using the values from
    :class:`~deeptab.configs.TrainerConfig`, but you can also call it
    directly in custom training loops.

    Parameters
    ----------
    module_or_params : nn.Module or iterable of Parameter
        Either a full ``nn.Module`` (recommended — enables parameter grouping)
        or a raw iterable of ``torch.nn.Parameter`` objects.
    optimizer_type : str, default="Adam"
        Registered optimizer name, case-insensitive (e.g. ``"Adam"``,
        ``"adamw"``, ``"SGD"``).  Use :func:`available_optimizers` to list
        all valid names, or :func:`register_optimizer` to add your own.
    lr : float, default=1e-4
        Learning rate passed to the optimizer constructor.
    weight_decay : float, default=1e-6
        L2 weight-decay coefficient.  When *no_weight_decay_for_bias_and_norm*
        is ``True``, this value applies only to the decay parameter group (see
        :func:`build_parameter_groups`).
    optimizer_kwargs : dict or None, default=None
        Extra keyword arguments forwarded verbatim to the optimizer constructor
        after ``lr`` and ``weight_decay``.  Keys that start with
        ``"optimizer_"`` should be stripped first via
        :func:`normalize_optimizer_kwargs` (done automatically inside
        ``TaskModel``).
    no_weight_decay_for_bias_and_norm : bool, default=False
        When ``True`` and *module_or_params* is an ``nn.Module``, parameters
        are split into two groups: bias and normalisation params receive
        ``weight_decay=0.0`` while all others receive the specified
        *weight_decay*.  Recommended for transformer-style architectures.

    Returns
    -------
    torch.optim.Optimizer
        A ready-to-use optimizer with ``lr`` and ``weight_decay`` set on the
        appropriate parameter groups.

    Raises
    ------
    ~deeptab.core.exceptions.InvalidParamError
        If *optimizer_type* is not registered.

    Examples
    --------
    **Standard Adam (default)**::

        from deeptab.training.optimizers import build_optimizer
        import torch.nn as nn

        model = nn.Linear(10, 1)
        opt = build_optimizer(model, optimizer_type="Adam", lr=1e-3)

    **AdamW with custom betas**::

        opt = build_optimizer(
            model,
            optimizer_type="AdamW",
            lr=3e-4,
            weight_decay=1e-2,
            optimizer_kwargs={"betas": (0.9, 0.95), "eps": 1e-8},
        )

    **Selective weight decay for transformer models**::

        opt = build_optimizer(
            model,
            optimizer_type="AdamW",
            lr=1e-3,
            weight_decay=1e-2,
            no_weight_decay_for_bias_and_norm=True,
        )
        len(opt.param_groups)  # 2: decay group + no-decay group

    **Raw parameter iterable** (e.g. for partial fine-tuning)::

        params = [p for p in model.parameters() if p.requires_grad]
        opt = build_optimizer(params, optimizer_type="SGD", lr=0.01, weight_decay=0.0)

    Notes
    -----
    When *no_weight_decay_for_bias_and_norm* is ``True`` and
    *module_or_params* is an ``nn.Module``, ``weight_decay`` is embedded
    inside the parameter groups returned by :func:`build_parameter_groups`.
    The optimizer constructor is therefore called **without** a top-level
    ``weight_decay`` argument — the per-group values take precedence.

    See Also
    --------
    :func:`build_parameter_groups` : Selective weight-decay parameter split.
    :func:`normalize_optimizer_kwargs` : Strip legacy ``optimizer_`` prefix.
    :func:`register_optimizer` : Register a custom optimizer class.
    :mod:`deeptab.training.schedulers` : Companion LR-scheduler factory.
    """
    cls = get_optimizer(optimizer_type)
    extra: dict[str, Any] = optimizer_kwargs or {}

    if no_weight_decay_for_bias_and_norm and isinstance(module_or_params, nn.Module):
        params: Any = build_parameter_groups(
            module_or_params,
            weight_decay=weight_decay,
            no_weight_decay_for_bias_and_norm=True,
        )
        # weight_decay is embedded in param groups; don't pass it again
        return cls(params, lr=lr, **extra)  # type: ignore[call-arg]

    raw_params = module_or_params.parameters() if isinstance(module_or_params, nn.Module) else module_or_params
    return cls(raw_params, lr=lr, weight_decay=weight_decay, **extra)  # type: ignore[call-arg]

"""Training loss functions and class-imbalance utilities used across DeepTab models.

New in v2.0.0.

Classification losses follow the same class-based, registry-driven design as the
distributional losses in :mod:`deeptab.distributions`: every concrete loss is an
``nn.Module`` subclass of :class:`BaseLoss` exposing a uniform
``forward(logits, targets) -> Tensor`` interface and registering itself under a
string ``name``.  This makes losses addressable from configs / HPO search spaces
(``loss_fct="focal"``) and keeps new losses trivial to add — subclass
:class:`BaseLoss`, give it a ``name``, and (optionally) override
``from_class_weights`` to describe how class weights map onto its parameters.

Helpers for imbalanced classification targets:

* :func:`compute_class_weights` — turn a sklearn-style ``class_weight`` argument
  (``"balanced"``, a mapping, or an array) into a per-class weight vector.
* :func:`build_classification_loss` — resolve a loss spec (``None``, a registry
  name, or an ``nn.Module``) into a ready-to-use loss, applying class weights.
* :func:`build_weighted_classification_loss` — construct the default weighted
  loss (binary :class:`WeightedBCEWithLogitsLoss` or multiclass
  :class:`WeightedCrossEntropyLoss`) from a per-class weight vector.

Available registered losses:

* ``"bce"`` — :class:`WeightedBCEWithLogitsLoss` (binary).
* ``"cross_entropy"`` — :class:`WeightedCrossEntropyLoss` (multiclass).
* ``"focal"`` — :class:`FocalLoss` (binary or multiclass; best for extreme imbalance).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BaseLoss",
    "FocalLoss",
    "WeightedBCEWithLogitsLoss",
    "WeightedCrossEntropyLoss",
    "build_classification_loss",
    "build_weighted_classification_loss",
    "compute_class_weights",
    "get_loss",
]


def compute_class_weights(
    class_weight: str | Mapping[Any, float] | np.ndarray | list | None,
    y: np.ndarray,
    classes: np.ndarray | None = None,
) -> np.ndarray | None:
    """Compute a per-class weight vector following scikit-learn conventions.

    Parameters
    ----------
    class_weight : {"balanced"}, mapping, array-like, or None
        * ``None`` — return ``None`` (no weighting).
        * ``"balanced"`` — weights are ``n_samples / (n_classes * bincount(y))``,
          matching ``sklearn.utils.class_weight.compute_class_weight``.
        * mapping — ``{class_label: weight}``; classes not present default to 1.0.
        * array-like — one weight per class, ordered to match ``classes``.
    y : ndarray of shape (n_samples,)
        Training target labels.
    classes : ndarray, optional
        Ordered array of unique class labels. If ``None``, inferred from ``y``
        via ``np.unique``.

    Returns
    -------
    weights : ndarray of shape (n_classes,) or None
        Per-class weights aligned with ``classes``; ``None`` when
        ``class_weight`` is ``None``.

    Raises
    ------
    ValueError
        If ``class_weight`` is an unrecognised string, or an array whose length
        does not match the number of classes.
    """
    if class_weight is None:
        return None

    y = np.asarray(y)
    classes = np.unique(y) if classes is None else np.asarray(classes)

    n_classes = len(classes)

    if isinstance(class_weight, str):
        if class_weight != "balanced":
            raise ValueError(f"Unsupported class_weight string {class_weight!r}; expected 'balanced'.")
        # n_samples / (n_classes * count_per_class)
        counts = np.array([(y == c).sum() for c in classes], dtype=np.float64)
        if (counts == 0).any():
            raise ValueError("Cannot use class_weight='balanced' when a class has zero samples in y.")
        weights = len(y) / (n_classes * counts)
        return weights.astype(np.float64)

    if isinstance(class_weight, Mapping):
        return np.array([float(class_weight.get(c, 1.0)) for c in classes], dtype=np.float64)

    # array-like
    weights = np.asarray(class_weight, dtype=np.float64)
    if weights.shape[0] != n_classes:
        raise ValueError(f"class_weight array has length {weights.shape[0]} but there are {n_classes} classes.")
    return weights


def build_weighted_classification_loss(
    class_weights: np.ndarray | None,
    num_classes: int,
    device: str | torch.device | None = None,
) -> nn.Module | None:
    """Build the default weighted classification loss from a per-class weight vector.

    Parameters
    ----------
    class_weights : ndarray of shape (n_classes,) or None
        Per-class weights produced by :func:`compute_class_weights`.  When
        ``None``, this function returns ``None`` so the caller can fall back to
        the default unweighted loss.
    num_classes : int
        Number of target classes. ``2`` selects a binary loss
        (:class:`WeightedBCEWithLogitsLoss` with ``pos_weight``); ``> 2``
        selects :class:`WeightedCrossEntropyLoss` with ``weight``.
    device : str or torch.device, optional
        Device on which to allocate the weight tensors. The loss is also a
        submodule of the Lightning module, so its buffers move automatically on
        ``.to(device)``; this argument simply allows eager placement.

    Returns
    -------
    loss : nn.Module or None
        A configured weighted loss module, or ``None`` when ``class_weights`` is
        ``None``.

    Notes
    -----
    For binary targets the positive-class weight passed to
    :class:`WeightedBCEWithLogitsLoss` is ``class_weights[1] / class_weights[0]``,
    which is the standard way to express ``scale_pos_weight`` from
    gradient-boosting libraries in terms of a balanced class-weight vector.
    """
    if class_weights is None:
        return None

    weights = torch.as_tensor(np.asarray(class_weights), dtype=torch.float32, device=device)

    if num_classes == 2:
        # BCEWithLogitsLoss expects a single positive-class weight (scalar tensor).
        pos_weight = (weights[1] / weights[0]).reshape(1)
        return WeightedBCEWithLogitsLoss(pos_weight=pos_weight)

    return WeightedCrossEntropyLoss(weight=weights)


class BaseLoss(nn.Module):
    """Base class for DeepTab classification losses.

    Mirrors :class:`deeptab.distributions.base.BaseDistribution`: every concrete
    loss is an ``nn.Module`` subclass exposing a uniform
    ``forward(logits, targets) -> Tensor`` interface, and registers itself under
    a string ``name`` so it can be selected from configs or HPO search spaces.

    To add a new loss, subclass :class:`BaseLoss` with a ``name`` keyword and
    implement :meth:`forward`. Override :meth:`from_class_weights` to describe how
    a per-class weight vector maps onto the loss's own parameters.

    Attributes
    ----------
    expects_class_indices : bool
        ``True`` when ``forward`` consumes integer class-index targets of shape
        ``(N,)`` (cross-entropy style); ``False`` for binary targets of shape
        ``(N, 1)``.  Used by the Lightning module to dispatch ensemble losses
        correctly.
    """

    expects_class_indices: bool = False
    loss_name: str | None = None

    _registry: ClassVar[dict[str, type[BaseLoss]]] = {}

    def __init_subclass__(cls, name: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.loss_name = name
        if name is not None:
            BaseLoss._registry[name] = cls

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Loss subclasses must implement forward().")

    @classmethod
    def from_class_weights(
        cls,
        class_weights: np.ndarray | None,
        num_classes: int,
        **kwargs: Any,
    ) -> BaseLoss:
        """Build the loss from a per-class weight vector.

        The base implementation ignores the weights; subclasses override this to
        translate ``class_weights`` into ``pos_weight`` / ``weight`` / ``alpha``.
        """
        return cls(**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """Return the sorted list of registered loss names."""
        return sorted(BaseLoss._registry)


class WeightedBCEWithLogitsLoss(BaseLoss, name="bce"):
    """Binary cross-entropy with logits and an optional positive-class weight.

    Parameters
    ----------
    pos_weight : Tensor, optional
        Weight of the positive class, as accepted by
        :class:`torch.nn.BCEWithLogitsLoss`. ``> 1`` up-weights the minority
        positive class.
    """

    expects_class_indices = False

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    @property
    def pos_weight(self) -> torch.Tensor | None:
        return self._loss.pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._loss(logits, targets)

    @classmethod
    def from_class_weights(cls, class_weights, num_classes, **kwargs):
        pos_weight = None
        if class_weights is not None:
            weights = torch.as_tensor(np.asarray(class_weights), dtype=torch.float32)
            pos_weight = (weights[1] / weights[0]).reshape(1)
        return cls(pos_weight=pos_weight, **kwargs)


class WeightedCrossEntropyLoss(BaseLoss, name="cross_entropy"):
    """Multiclass cross-entropy with an optional per-class weight vector.

    Parameters
    ----------
    weight : Tensor, optional
        Per-class weights, as accepted by :class:`torch.nn.CrossEntropyLoss`.
    """

    expects_class_indices = True

    def __init__(self, weight: torch.Tensor | None = None):
        super().__init__()
        self._loss = nn.CrossEntropyLoss(weight=weight)

    @property
    def weight(self) -> torch.Tensor | None:
        return self._loss.weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._loss(logits, targets)

    @classmethod
    def from_class_weights(cls, class_weights, num_classes, **kwargs):
        weight = None
        if class_weights is not None:
            weight = torch.as_tensor(np.asarray(class_weights), dtype=torch.float32)
        return cls(weight=weight, **kwargs)


class FocalLoss(BaseLoss, name="focal"):
    r"""Focal loss (Lin et al., 2017) for imbalanced classification.

    Focal loss down-weights well-classified (easy) examples by a factor of
    :math:`(1 - p_t)^\gamma`, concentrating training on the hard, typically
    minority-class, examples. It often outperforms simple class weighting under
    extreme imbalance.

    Parameters
    ----------
    gamma : float, default=2.0
        Focusing parameter. ``0`` reduces to (weighted) cross-entropy; larger
        values increasingly down-weight easy examples.
    alpha : Tensor, float, or None, default=None
        Class-balancing factor. For binary targets a float in ``[0, 1]`` weights
        the positive class. For multiclass targets a length-``num_classes``
        tensor weights each class.
    num_classes : int, default=2
        ``2`` selects the binary formulation (logits of shape ``(N, 1)``);
        ``> 2`` selects the multiclass formulation (logits of shape ``(N, C)``).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | float | None = None,
        num_classes: int = 2,
    ):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.expects_class_indices = num_classes > 2
        self.register_buffer("alpha_weight", alpha if isinstance(alpha, torch.Tensor) else None)
        self.alpha_scalar = float(alpha) if (alpha is not None and not isinstance(alpha, torch.Tensor)) else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.num_classes > 2:
            return self._multiclass_forward(logits, targets)
        return self._binary_forward(logits, targets)

    def _binary_forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.reshape(-1)
        targets = targets.reshape(-1).to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = (1 - p_t).clamp(min=0) ** self.gamma * bce
        if self.alpha_scalar is not None:
            alpha_t = self.alpha_scalar * targets + (1 - self.alpha_scalar) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()

    def _multiclass_forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.reshape(-1).long()
        log_p = F.log_softmax(logits, dim=-1)
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1 - pt).clamp(min=0) ** self.gamma) * log_pt
        if isinstance(self.alpha_weight, torch.Tensor):
            loss = self.alpha_weight.gather(0, targets) * loss
        return loss.mean()

    @classmethod
    def from_class_weights(cls, class_weights, num_classes, **kwargs):
        alpha: torch.Tensor | float | None = None
        if class_weights is not None:
            weights = np.asarray(class_weights, dtype=np.float64)
            if num_classes == 2:
                # Map the two-class weights onto a single positive-class alpha in [0, 1].
                alpha = float(weights[1] / (weights[0] + weights[1]))
            else:
                alpha = torch.as_tensor(weights, dtype=torch.float32)
        return cls(num_classes=num_classes, alpha=alpha, **kwargs)


def get_loss(name: str) -> type[BaseLoss]:
    """Look up a registered loss class by name.

    Parameters
    ----------
    name : str
        Registered loss name (see :meth:`BaseLoss.available`).

    Returns
    -------
    type[BaseLoss]
        The loss class.

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    """
    try:
        return BaseLoss._registry[name]
    except KeyError:
        raise ValueError(f"Unknown loss {name!r}; available losses: {BaseLoss.available()}") from None


def build_classification_loss(
    loss: str | nn.Module | None = None,
    *,
    num_classes: int,
    class_weights: np.ndarray | None = None,
    **loss_kwargs: Any,
) -> nn.Module | None:
    """Resolve a loss specification into a ready-to-use loss module.

    Parameters
    ----------
    loss : str, nn.Module, or None
        * ``nn.Module`` — returned as-is (takes precedence over ``class_weights``).
        * ``str`` — a registered loss name (e.g. ``"focal"``), built via
          :meth:`BaseLoss.from_class_weights` so any ``class_weights`` are applied.
        * ``None`` — fall back to the default weighted loss from
          :func:`build_weighted_classification_loss` (or ``None`` when no weights).
    num_classes : int
        Number of target classes.
    class_weights : ndarray, optional
        Per-class weight vector from :func:`compute_class_weights`.
    **loss_kwargs
        Extra keyword arguments forwarded to the loss constructor (e.g.
        ``gamma`` for :class:`FocalLoss`).

    Returns
    -------
    nn.Module or None
        The resolved loss, or ``None`` to signal the caller should use its
        task default.
    """
    if isinstance(loss, nn.Module):
        return loss
    if loss is None:
        return build_weighted_classification_loss(class_weights, num_classes)
    if isinstance(loss, str):
        return get_loss(loss).from_class_weights(class_weights, num_classes, **loss_kwargs)
    raise TypeError(f"loss must be None, a registered name, or an nn.Module, got {type(loss).__name__}.")

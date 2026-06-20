from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .autoint import AutoInt
    from .enode import ENODE
    from .ft_transformer import FTTransformer
    from .mambatab import MambaTab
    from .mambattention import MambAttention
    from .mambular import Mambular
    from .mlp import MLP
    from .ndtf import NDTF
    from .node import NODE
    from .resnet import ResNet
    from .saint import SAINT
    from .tabm import TabM
    from .tabr import TabR
    from .tabtransformer import TabTransformer
    from .tabularnn import TabulaRNN

_REGISTRY: dict[str, tuple[str, str]] = {
    "AutoInt": (".autoint", "AutoInt"),
    "ENODE": (".enode", "ENODE"),
    "FTTransformer": (".ft_transformer", "FTTransformer"),
    "MambaTab": (".mambatab", "MambaTab"),
    "MambAttention": (".mambattention", "MambAttention"),
    "Mambular": (".mambular", "Mambular"),
    "MLP": (".mlp", "MLP"),
    "NDTF": (".ndtf", "NDTF"),
    "NODE": (".node", "NODE"),
    "ResNet": (".resnet", "ResNet"),
    "SAINT": (".saint", "SAINT"),
    "TabM": (".tabm", "TabM"),
    "TabR": (".tabr", "TabR"),
    "TabTransformer": (".tabtransformer", "TabTransformer"),
    "TabulaRNN": (".tabularnn", "TabulaRNN"),
}

__all__ = list(_REGISTRY.keys())


def __getattr__(name: str):
    if name in _REGISTRY:
        import importlib

        module_path, class_name = _REGISTRY[name]
        module = importlib.import_module(module_path, package=__name__)
        obj = getattr(module, class_name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

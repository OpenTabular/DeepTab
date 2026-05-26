from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .modern_nca import ModernNCA
    from .tangos import Tangos
    from .trompt import Trompt

_REGISTRY: dict[str, tuple[str, str]] = {
    "ModernNCA": (".modern_nca", "ModernNCA"),
    "Tangos": (".tangos", "Tangos"),
    "Trompt": (".trompt", "Trompt"),
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

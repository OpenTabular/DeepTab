import importlib
import warnings

from .autoint import AutoIntClassifier, AutoIntLSS, AutoIntRegressor
from .enode import ENODELSS, ENODEClassifier, ENODERegressor
from .fttransformer import (
    FTTransformerClassifier,
    FTTransformerLSS,
    FTTransformerRegressor,
)
from .mambatab import MambaTabClassifier, MambaTabLSS, MambaTabRegressor
from .mambattention import (
    MambAttentionClassifier,
    MambAttentionLSS,
    MambAttentionRegressor,
)
from .mambular import MambularClassifier, MambularLSS, MambularRegressor
from .mlp import MLPLSS, MLPClassifier, MLPRegressor
from .ndtf import NDTFLSS, NDTFClassifier, NDTFRegressor
from .node import NODELSS, NODEClassifier, NODERegressor
from .resnet import ResNetClassifier, ResNetLSS, ResNetRegressor
from .saint import SAINTLSS, SAINTClassifier, SAINTRegressor
from .tabm import TabMClassifier, TabMLSS, TabMRegressor
from .tabr import TabRClassifier, TabRLSS, TabRRegressor
from .tabtransformer import (
    TabTransformerClassifier,
    TabTransformerLSS,
    TabTransformerRegressor,
)
from .tabularnn import TabulaRNNClassifier, TabulaRNNLSS, TabulaRNNRegressor
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor

__all__ = [
    "ENODELSS",
    "MLPLSS",
    "NDTFLSS",
    "NODELSS",
    "SAINTLSS",
    "AutoIntClassifier",
    "AutoIntLSS",
    "AutoIntRegressor",
    "ENODEClassifier",
    "ENODERegressor",
    "FTTransformerClassifier",
    "FTTransformerLSS",
    "FTTransformerRegressor",
    "MLPClassifier",
    "MLPRegressor",
    "MambAttentionClassifier",
    "MambAttentionLSS",
    "MambAttentionRegressor",
    "MambaTabClassifier",
    "MambaTabLSS",
    "MambaTabRegressor",
    "MambularClassifier",
    "MambularLSS",
    "MambularRegressor",
    "NDTFClassifier",
    "NDTFRegressor",
    "NODEClassifier",
    "NODERegressor",
    "ResNetClassifier",
    "ResNetLSS",
    "ResNetRegressor",
    "SAINTClassifier",
    "SAINTRegressor",
    "SklearnBaseClassifier",
    "SklearnBaseLSS",
    "SklearnBaseRegressor",
    "TabMClassifier",
    "TabMLSS",
    "TabMRegressor",
    "TabRClassifier",
    "TabRLSS",
    "TabRRegressor",
    "TabTransformerClassifier",
    "TabTransformerLSS",
    "TabTransformerRegressor",
    "TabulaRNNClassifier",
    "TabulaRNNLSS",
    "TabulaRNNRegressor",
]

# ---------------------------------------------------------------------------
# Backwards-compatibility shim for experimental models
# ---------------------------------------------------------------------------

_EXPERIMENTAL_COMPAT: dict[str, str] = {
    "ModernNCAClassifier": "deeptab.models.experimental",
    "ModernNCALSS": "deeptab.models.experimental",
    "ModernNCARegressor": "deeptab.models.experimental",
    "TangosClassifier": "deeptab.models.experimental",
    "TangosLSS": "deeptab.models.experimental",
    "TangosRegressor": "deeptab.models.experimental",
    "TromptClassifier": "deeptab.models.experimental",
    "TromptLSS": "deeptab.models.experimental",
    "TromptRegressor": "deeptab.models.experimental",
}


def __getattr__(name: str):
    if name in _EXPERIMENTAL_COMPAT:
        new_path = _EXPERIMENTAL_COMPAT[name]
        warnings.warn(
            f"{name!r} has moved to '{new_path}'. Update your import: from {new_path} import {name}",
            DeprecationWarning,
            stacklevel=2,
        )
        mod = importlib.import_module(new_path)
        return getattr(mod, name)
    raise AttributeError(f"module 'deeptab.models' has no attribute {name!r}")

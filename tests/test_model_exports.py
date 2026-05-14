"""
Tests verifying the stable / experimental model export split.

Stable models must be importable from ``deeptab.models``.
Experimental models must be importable from ``deeptab.models.experimental``
and must *not* appear in ``deeptab.models.__all__``.
"""

import importlib

import pytest

# ---------------------------------------------------------------------------
# Expected exports
# ---------------------------------------------------------------------------

STABLE_CLASSES = [
    "AutoIntClassifier",
    "AutoIntLSS",
    "AutoIntRegressor",
    "ENODEClassifier",
    "ENODELSS",
    "ENODERegressor",
    "FTTransformerClassifier",
    "FTTransformerLSS",
    "FTTransformerRegressor",
    "MambAttentionClassifier",
    "MambAttentionLSS",
    "MambAttentionRegressor",
    "MambaTabClassifier",
    "MambaTabLSS",
    "MambaTabRegressor",
    "MambularClassifier",
    "MambularLSS",
    "MambularRegressor",
    "MLPClassifier",
    "MLPLSS",
    "MLPRegressor",
    "NDTFClassifier",
    "NDTFLSS",
    "NDTFRegressor",
    "NODEClassifier",
    "NODELSS",
    "NODERegressor",
    "ResNetClassifier",
    "ResNetLSS",
    "ResNetRegressor",
    "SAINTClassifier",
    "SAINTLSS",
    "SAINTRegressor",
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

EXPERIMENTAL_CLASSES = [
    "ModernNCAClassifier",
    "ModernNCALSS",
    "ModernNCARegressor",
    "TangosClassifier",
    "TangosLSS",
    "TangosRegressor",
    "TromptClassifier",
    "TromptLSS",
    "TromptRegressor",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import(module_path: str, attr: str):
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


# ---------------------------------------------------------------------------
# Stable models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("class_name", STABLE_CLASSES)
def test_stable_model_importable(class_name: str):
    """Every stable class is importable from deeptab.models."""
    cls = _import("deeptab.models", class_name)
    assert cls is not None


@pytest.mark.parametrize("class_name", STABLE_CLASSES)
def test_stable_model_in_all(class_name: str):
    """Every stable class is listed in deeptab.models.__all__."""
    import deeptab.models as m

    assert class_name in m.__all__, f"{class_name!r} missing from deeptab.models.__all__"


# ---------------------------------------------------------------------------
# Experimental models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("class_name", EXPERIMENTAL_CLASSES)
def test_experimental_model_importable(class_name: str):
    """Every experimental class is importable from deeptab.models.experimental."""
    cls = _import("deeptab.models.experimental", class_name)
    assert cls is not None


@pytest.mark.parametrize("class_name", EXPERIMENTAL_CLASSES)
def test_experimental_model_in_experimental_all(class_name: str):
    """Every experimental class is listed in deeptab.models.experimental.__all__."""
    import deeptab.models.experimental as exp

    assert class_name in exp.__all__, f"{class_name!r} missing from deeptab.models.experimental.__all__"


@pytest.mark.parametrize("class_name", EXPERIMENTAL_CLASSES)
def test_experimental_model_not_in_stable_all(class_name: str):
    """Experimental classes must not leak into deeptab.models.__all__."""
    import deeptab.models as m

    assert class_name not in m.__all__, f"{class_name!r} should not be in deeptab.models.__all__"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_stable_import_paths():
    """All stable entries in MODEL_REGISTRY have import_path == 'deeptab.models'."""
    from deeptab.models._registry import MODEL_REGISTRY

    for name, info in MODEL_REGISTRY.items():
        if info.status == "stable":
            assert info.import_path == "deeptab.models", (
                f"{name}: expected import_path 'deeptab.models', got {info.import_path!r}"
            )


def test_registry_experimental_import_paths():
    """All experimental entries have import_path == 'deeptab.models.experimental'."""
    from deeptab.models._registry import MODEL_REGISTRY

    for name, info in MODEL_REGISTRY.items():
        if info.status == "experimental":
            assert info.import_path == "deeptab.models.experimental", (
                f"{name}: expected 'deeptab.models.experimental', got {info.import_path!r}"
            )

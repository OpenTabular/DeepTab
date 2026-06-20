"""Internal mixin classes that compose ``SklearnBase``.

Each mixin owns a single concern. ``SklearnBase`` inherits from all of them
in the order shown below; this MRO is the only contract between the mixins —
no mixin imports another.

MRO (outermost → innermost)::

    SklearnBase(
        _ObservabilityMixin,   # lifecycle event dispatch
        _FitMixin,             # _build_model + fit + _pretrain
        _PredictMixin,         # predict (abstract) + encode + _score
        _SerializationMixin,   # save / load
        _HyperparameterMixin,  # optimize_hparams
        InspectionMixin,       # get_number_of_params + diagnostics
        BaseEstimator,         # sklearn get_params / set_params / clone
    )

Note
----
These classes are internal implementation details. Import from
``deeptab.models`` (e.g. ``MLPClassifier``) rather than from this package
directly.
"""

from deeptab.models._mixins.fit import _FitMixin
from deeptab.models._mixins.hpo import _HyperparameterMixin
from deeptab.models._mixins.observability import _ObservabilityMixin
from deeptab.models._mixins.predict import _PredictMixin
from deeptab.models._mixins.serialization import _SerializationMixin

__all__ = [
    "_FitMixin",
    "_HyperparameterMixin",
    "_ObservabilityMixin",
    "_PredictMixin",
    "_SerializationMixin",
]

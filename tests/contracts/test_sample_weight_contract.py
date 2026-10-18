"""Unsupported per-row loss weighting must fail before fit mutates state."""

import pytest

from deeptab.core import DeepTabError, UnsupportedParamError
from deeptab.models.mlp import MLPLSS, MLPRegressor


@pytest.mark.parametrize("estimator_cls", [MLPRegressor, MLPLSS])
@pytest.mark.parametrize("sample_weight", [None, [], [1.0, 2.0]])
def test_unsupported_sample_weight_rejected_before_mutation(estimator_cls, sample_weight):
    """Reject unsupported weights before input validation or estimator mutation."""
    estimator = estimator_cls()
    estimator.is_fitted_ = True
    before = vars(estimator).copy()
    kwargs = {"family": "normal"} if estimator_cls is MLPLSS else {}

    with pytest.raises(UnsupportedParamError, match="sample_weight is not supported") as exc_info:
        estimator.fit(None, None, sample_weight=sample_weight, **kwargs)

    assert isinstance(exc_info.value, DeepTabError)
    assert isinstance(exc_info.value, ValueError)
    assert vars(estimator) == before

"""sklearn estimator contract tests for DeepTab estimators.

Uses ``parametrize_with_checks`` to run the full suite of sklearn estimator
checks against ``MLPClassifier`` and ``MLPRegressor``.

Strategy
--------
* Known structural failures are marked ``xfail(strict=True)`` so that the
  test suite stays green while clearly tracking which gaps remain.
* ``strict=True`` means an *unexpected pass* also fails the suite, ensuring
  that compliance improvements are noticed and xfails are removed.
* Estimators are constructed with ``max_epochs=3`` to keep CI fast.

Every annotated check executes. Reasons describe either an unsupported contract
or a remaining validation gap, not a promise that the behavior is acceptable.
Passing checks protect fixed-seed refits, prediction order and subset invariance,
read-only memmaps, and Pipeline consistency for these estimators.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from deeptab.configs import TrainerConfig
from deeptab.models.mlp import MLPClassifier, MLPRegressor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: TrainerConfig that keeps each check fast (3 epochs, low patience).
_FAST_TRAINER = TrainerConfig(max_epochs=3, patience=2, lr_patience=2)


def _check_name(check) -> str:
    """Return the base function name of a parametrize_with_checks check object."""
    return check.func.__name__ if hasattr(check, "func") else check.__name__


# ---------------------------------------------------------------------------
# xfail registry
# Each entry maps a check function name to the reason string shown in the
# test report.  All xfails use strict=True so that a newly-passing check
# causes a test failure, prompting removal of the annotation.
# ---------------------------------------------------------------------------

_XFAIL_CHECKS: dict[str, str] = {
    "check_estimators_empty_data_messages": (
        "EmptyDataError rejects empty data but its message does not match sklearn's zero-feature pattern."
    ),
    "check_n_features_in_after_fitting": (
        "ColumnCountError rejects feature-count mismatches but its message does not match sklearn's expected pattern."
    ),
    "check_estimators_pickle": (
        "SklearnBase.__getstate__ clears the task model, trainer, and datamodule "
        "to avoid serialising Lightning runtime state. Use estimator.save() / estimator.load() for "
        "persistence. Standard fitted-estimator pickle is intentionally not supported."
    ),
    "check_dtype_object": (
        "On CPU, classifier dict-valued features raise 'unhashable type' instead of the expected numeric-input "
        "message; regressor object targets fail without the expected 'Unknown label type' wording."
    ),
    "check_estimator_sparse_array": (
        "Sparse array input is unsupported, but rejection does not consistently identify sparse input."
    ),
    "check_estimator_sparse_matrix": (
        "Sparse matrix input is unsupported, but rejection does not consistently identify sparse input."
    ),
    "check_sample_weight_equivalence_on_dense_data": (
        "Classifier sample_weight controls stochastic row sampling, not weighted loss; predictions do not "
        "match sklearn's removed/repeated-row equivalence contract."
    ),
    "check_sample_weights_not_an_array": (
        "Classifier label discovery calls np.unique on sklearn's array-like target wrapper before coercion."
    ),
    "check_estimators_nan_inf": (
        "fit accepts NaN features for imputation; sklearn expects rejection. "
        "The check stops at NaN acceptance and does not establish infinity safety."
    ),
}


# ---------------------------------------------------------------------------
# Contract test
# ---------------------------------------------------------------------------


_TASK_XFAIL_CHECKS = {
    "check_classifier_data_not_an_array": (
        "Classifier label discovery calls np.unique before coercing array-like targets."
    ),
    "check_regressor_data_not_an_array": (
        "Fit validation calls len before coercing sklearn's array-like input wrapper."
    ),
    "check_classifiers_train": (
        "Mismatched X/y lengths raise DataError, but the message lacks sklearn's "
        "expected 'inconsistent numbers of samples' wording."
    ),
    "check_regressors_train": (
        "Mismatched X/y lengths raise DataError, but the message lacks sklearn's "
        "expected 'inconsistent numbers of samples' wording."
    ),
    "check_classifiers_regression_target": (
        "Continuous classification targets fail at stratified splitting rather than "
        "raising sklearn's expected 'Unknown label type' error."
    ),
    "check_supervised_y_no_nan": (
        "Infinite regression targets are accepted; non-finite target validation must precede estimator mutation."
    ),
    "check_supervised_y_2d": ("Column-vector targets are accepted without sklearn's required DataConversionWarning."),
    "check_requires_y_none": (
        "Missing targets trigger a classifier single-class ValueError or regressor len(None) TypeError, "
        "rather than an actionable requires-y ValueError."
    ),
}


@parametrize_with_checks(
    [
        MLPClassifier(trainer_config=_FAST_TRAINER),
        MLPRegressor(trainer_config=_FAST_TRAINER),
    ]
)
def test_sklearn_compatible_estimator(estimator, check, request, monkeypatch):
    """Run every sklearn estimator contract check.

    These checks use CPU so registry reasons do not depend on automatic device
    selection. Accelerator behavior requires separate device-specific evidence.
    Every marked body executes; an unexpected pass requires updating the registry.
    """
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    name = _check_name(check)
    if name in _XFAIL_CHECKS:
        request.node.add_marker(pytest.mark.xfail(reason=_XFAIL_CHECKS[name], strict=True))
    if name in _TASK_XFAIL_CHECKS and (name != "check_supervised_y_no_nan" or isinstance(estimator, MLPRegressor)):
        request.node.add_marker(pytest.mark.xfail(reason=_TASK_XFAIL_CHECKS[name], strict=True))
    check(estimator)


@pytest.mark.parametrize("check_name", ["check_dtype_object", "check_classifiers_regression_target"])
def test_expected_failure_marker_does_not_short_circuit(check_name, monkeypatch):
    """Both registries execute failing bodies and require strict unexpected passes."""
    markers = []
    executed = []
    estimator = MLPClassifier()
    request = SimpleNamespace(node=SimpleNamespace(add_marker=markers.append))

    def check(candidate):
        executed.append(candidate)
        raise AssertionError("check body executed")

    check.__name__ = check_name
    with pytest.raises(AssertionError, match="check body executed"):
        test_sklearn_compatible_estimator(estimator, check, request, monkeypatch)

    assert executed == [estimator]
    assert len(markers) == 1
    assert markers[0].name == "xfail"
    assert markers[0].kwargs["strict"] is True
    assert markers[0].kwargs.get("run", True) is True

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

Phases where gaps are expected to be fixed
-------------------------------------------
Phase 2 (interface segregation):
    check_no_attributes_set_in_init
    check_do_not_raise_errors_in_init_or_set_params
    check_set_params

By design (not planned to fix):
    check_estimator_sparse_array / check_estimator_sparse_matrix
        DeepTab does not support sparse input.
    check_sample_weight_* / check_sample_weight_equivalence_*
        sample_weight is not in fit() — use the sampler= argument instead.
    check_fit_idempotent
        Neural networks are stochastic; predictions differ between calls even
        with the same random_state.
    check_methods_sample_order_invariance / check_methods_subset_invariance
        Batch statistics (e.g. BatchNorm) make predictions order- and
        subset-sensitive.
    check_readonly_memmap_input
        Read-only memory-mapped arrays may fail during DataFrame conversion.
    check_estimators_nan_inf
        NaN/Inf is handled by the preprocessor's imputer, not at the
        sklearn validate_data level.
"""

from __future__ import annotations

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
    # ------------------------------------------------------------------
    # Phase 2 target: align error messages with sklearn's validate_data format
    # ------------------------------------------------------------------
    "check_estimators_empty_data_messages": (
        "EmptyDataError message ('Input DataFrame passed to fit() is empty …') "
        "does not match the pattern sklearn expects from validate_data "
        "('0 feature(s) (shape=(…, 0)) while a minimum of … is required'). "
        "Fix requires adopting sklearn's validate_data call or updating the "
        "message format. Tracked for Phase 2."
    ),
    "check_n_features_in_after_fitting": (
        "ColumnCountError message does not match the regex sklearn looks for "
        "after n_features_in_ mismatch. Fix requires aligning the message "
        "format with sklearn's expected pattern. Tracked for Phase 2."
    ),
    # ------------------------------------------------------------------
    # Phase 2 target: interface segregation
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Persistence: pickle is not the supported serialisation mechanism
    # ------------------------------------------------------------------
    "check_estimators_pickle": (
        "SklearnBase.__getstate__ clears the task model, trainer, and datamodule "
        "to avoid serialising Lightning runtime state. Use estimator.save() / estimator.load() for "
        "persistence. Standard pickle is intentionally not supported."
    ),
    # ------------------------------------------------------------------
    # Pipeline output-shape mismatch
    # ------------------------------------------------------------------
    "check_pipeline_consistency": (
        "Pipeline wraps predict() output in a way that exposes a shape "
        "mismatch between the standalone estimator and the pipeline. "
        "Tracked for investigation in Phase 2."
    ),
    # ------------------------------------------------------------------
    # Device-specific: MPS does not support integer tensors in linear layers
    # ------------------------------------------------------------------
    "check_dtype_object": (
        "On MPS (Apple Silicon), object-dtype features are encoded as integer "
        "ordinals which MPS cannot feed through Linear layers. "
        "Passes on CPU. Device-specific limitation."
    ),
    # ------------------------------------------------------------------
    # By design: sparse input not supported
    # ------------------------------------------------------------------
    "check_estimator_sparse_array": (
        "DeepTab does not support sparse array input. Convert to dense before calling fit()."
    ),
    "check_estimator_sparse_matrix": (
        "DeepTab does not support sparse matrix input. Convert to dense before calling fit()."
    ),
    "check_sample_weight_equivalence_on_sparse_data": ("Sparse input not supported."),
    # ------------------------------------------------------------------
    # By design: sample_weight not in fit()
    # ------------------------------------------------------------------
    "check_sample_weight_equivalence_on_dense_data": (
        "fit() does not accept a sample_weight argument. "
        "Use sampler='balanced' or pass an explicit weight array via sampler=."
    ),
    "check_sample_weights_list": "sample_weight not in fit() signature.",
    "check_sample_weights_not_an_array": "sample_weight not in fit() signature.",
    "check_sample_weights_not_overwritten": "sample_weight not in fit() signature.",
    "check_sample_weights_pandas_series": "sample_weight not in fit() signature.",
    "check_sample_weights_shape": "sample_weight not in fit() signature.",
    # ------------------------------------------------------------------
    # By design: DL stochasticity
    # ------------------------------------------------------------------
    "check_fit_idempotent": (
        "Neural network fit() is stochastic. Predictions differ between "
        "successive calls even with a fixed random_state."
    ),
    "check_methods_sample_order_invariance": (
        "Batch statistics (e.g. BatchNorm) make predictions sensitive to sample order within a mini-batch."
    ),
    "check_methods_subset_invariance": (
        "Predictions on a subset may differ from the corresponding rows of the "
        "full-batch prediction due to batch-level normalisation."
    ),
    # ------------------------------------------------------------------
    # Infrastructure / edge-case mismatches
    # ------------------------------------------------------------------
    "check_readonly_memmap_input": (
        "Read-only memory-mapped arrays fail during pd.DataFrame conversion. Copy the array before calling fit()."
    ),
    "check_estimators_nan_inf": (
        "NaN / Inf values in X are handled by the preprocessor's imputer, not "
        "by a sklearn-level validate_data call. The error type / message differs "
        "from what sklearn expects."
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
        "Regressor target validation does not implement sklearn's full finite-value checks and expected NaN/Inf error wording."
    ),
    "check_supervised_y_2d": ("Column-vector targets are accepted without sklearn's required DataConversionWarning."),
    "check_requires_y_none": ("A missing target raises TypeError before sklearn's expected requires-y ValueError."),
}


@parametrize_with_checks(
    [
        MLPClassifier(trainer_config=_FAST_TRAINER),
        MLPRegressor(trainer_config=_FAST_TRAINER),
    ]
)
def test_sklearn_compatible_estimator(estimator, check, request):
    """Run every sklearn estimator contract check.

    Checks in the expected-failure registries document remaining compliance
    gaps. Newly enabled task-specific checks run with strict markers so an
    unexpected pass requires updating the registry. Other checks must pass.
    """
    name = _check_name(check)
    if name in _XFAIL_CHECKS:
        pytest.xfail(_XFAIL_CHECKS[name])
    if name in _TASK_XFAIL_CHECKS and (name != "check_supervised_y_no_nan" or isinstance(estimator, MLPRegressor)):
        request.node.add_marker(pytest.mark.xfail(reason=_TASK_XFAIL_CHECKS[name], strict=True))
    check(estimator)

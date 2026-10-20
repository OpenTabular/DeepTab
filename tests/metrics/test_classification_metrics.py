"""Tests for classification metrics."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from deeptab.metrics import (  # Classification; Distributional; Registry; Base; Regression
    AUPRC,
    AUROC,
    Accuracy,
    BrierScore,
    ExpectedCalibrationError,
    F1Score,
    LogLoss,
)

RNG = np.random.default_rng(42)

N = 100


@pytest.fixture
def clf_data_binary():
    """Binary classification labels and probability scores."""
    y_true = RNG.integers(0, 2, N)
    proba_pos = np.clip(y_true + RNG.normal(0.0, 0.2, N), 0.01, 0.99)
    proba = np.column_stack([1.0 - proba_pos, proba_pos])
    return y_true, proba


@pytest.fixture
def clf_data_multiclass():
    """3-class labels and probability matrix."""
    y_true = RNG.integers(0, 3, N)
    raw = RNG.dirichlet(alpha=[2.0, 2.0, 2.0], size=N)
    # Bias toward the true class
    for i, c in enumerate(y_true):
        raw[i, c] += 1.0
    proba = raw / raw.sum(axis=1, keepdims=True)
    return y_true, proba


class TestClassificationMetrics:
    def test_accuracy_perfect(self):
        y = np.array([0, 1, 2, 0])
        proba = np.eye(3)[[0, 1, 2, 0]]
        assert Accuracy()(y, proba) == pytest.approx(1.0)

    def test_accuracy_all_wrong(self):
        y = np.array([0, 0, 0])
        proba = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]])
        assert Accuracy()(y, proba) == pytest.approx(0.0)

    def test_accuracy_binary_1d_proba(self):
        y = np.array([0, 1, 1, 0])
        proba = np.array([0.1, 0.9, 0.8, 0.2])
        assert Accuracy()(y, proba) == pytest.approx(1.0)

    @pytest.mark.parametrize("classes", [["no", "yes"], ["blue", "green", "red"]], ids=["binary", "multiclass"])
    @pytest.mark.parametrize("dtype", [str, object], ids=["unicode", "object"])
    @pytest.mark.parametrize(
        ("metric", "reference"),
        [
            pytest.param(Accuracy(), accuracy_score, id="accuracy"),
            pytest.param(F1Score(average="macro"), partial(f1_score, average="macro"), id="f1-macro"),
            pytest.param(F1Score(average="weighted"), partial(f1_score, average="weighted"), id="f1-weighted"),
        ],
    )
    def test_string_labels_match_sklearn(self, classes, dtype, metric, reference):
        vocabulary = np.asarray(classes, dtype=dtype)
        y_true = vocabulary[np.arange(6) % len(classes)]
        predictions = y_true.copy()
        predictions[0] = vocabulary[1]
        assert metric(y_true, predictions) == pytest.approx(reference(y_true, predictions))

    @pytest.mark.parametrize("dtype", [str, object], ids=["unicode", "object"])
    @pytest.mark.parametrize("positive_label", ["no", "yes"])
    def test_f1_binary_string_positive_label(self, dtype, positive_label):
        y_true = np.asarray(["no", "yes", "no", "yes"], dtype=dtype)
        predictions = np.asarray(["no", "yes", "yes", "yes"], dtype=dtype)
        metric = F1Score(pos_label=positive_label)
        expected = f1_score(y_true, predictions, pos_label=positive_label)
        assert metric(y_true, predictions) == pytest.approx(expected)
        assert f"pos_label={positive_label!r}" in repr(metric)

    @pytest.mark.parametrize(
        ("targets", "predictions", "expected_labels"),
        [
            pytest.param([10, 20, 30, 10], np.array([10, 30, 30, 10]), [10, 30, 30, 10], id="integers"),
            pytest.param([-3, 2, -3, 2], np.array([-3.0, 2.0, 2.0, 2.0]), [-3, 2, 2, 2], id="integral-floats"),
            pytest.param([0, 1, 0, 1], np.array([0, 1, 1, 1], dtype=object), [0, 1, 1, 1], id="object-integers"),
            pytest.param([0, 1, 0, 1], np.array([0.1, 0.9, 0.5, 0.49]), [0, 1, 1, 0], id="binary-probabilities"),
            pytest.param(
                [0, 1, 0, 1], np.array([0.1, 0.9, 0.5, 0.49], dtype=object), [0, 1, 1, 0], id="object-probabilities"
            ),
            pytest.param([0, 1, 2, 0], np.eye(3)[[0, 2, 2, 0]], [0, 2, 2, 0], id="multiclass-probabilities"),
            pytest.param(
                [False, True, False, True],
                np.array([False, True, True, True]),
                [False, True, True, True],
                id="booleans",
            ),
        ],
    )
    def test_numeric_labels_and_probabilities_match_sklearn(self, targets, predictions, expected_labels):
        assert Accuracy()(targets, predictions) == pytest.approx(accuracy_score(targets, expected_labels))
        for average in ("macro", "weighted"):
            expected = f1_score(targets, expected_labels, average=average)
            assert F1Score(average=average)(targets, predictions) == pytest.approx(expected)

    def test_auroc_in_unit_interval(self, clf_data_binary):
        y_true, proba = clf_data_binary
        score = AUROC()(y_true, proba)
        assert 0.0 <= score <= 1.0

    def test_auroc_multiclass(self, clf_data_multiclass):
        y_true, proba = clf_data_multiclass
        score = AUROC()(y_true, proba)
        assert 0.0 <= score <= 1.0

    def test_auprc_in_unit_interval(self, clf_data_binary):
        y_true, proba = clf_data_binary
        assert 0.0 <= AUPRC()(y_true, proba) <= 1.0

    def test_logloss_nonnegative(self, clf_data_binary):
        y_true, proba = clf_data_binary
        assert LogLoss()(y_true, proba) >= 0.0

    def test_brier_in_unit_interval(self, clf_data_binary):
        y_true, proba = clf_data_binary
        assert 0.0 <= BrierScore()(y_true, proba) <= 1.0

    def test_ece_in_unit_interval(self, clf_data_binary):
        y_true, proba = clf_data_binary
        assert 0.0 <= ExpectedCalibrationError()(y_true, proba) <= 1.0

    def test_ece_zero_for_perfect_calibration(self):
        """A model that always predicts 100% confidence and is always right → ECE = 0."""
        y_true = np.array([0, 1, 0, 1])
        proba = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        assert ExpectedCalibrationError()(y_true, proba) == pytest.approx(0.0)

    def test_f1_perfect(self):
        y = np.array([0, 1, 0, 1])
        proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
        assert F1Score(average="binary")(y, proba) == pytest.approx(1.0)

    def test_f1_invalid_average(self):
        with pytest.raises(ValueError):
            F1Score(average="micro")

    def test_ece_counts_full_confidence(self):
        """Confidence exactly 1.0 must fall in the last bin, not be dropped."""
        y_true = np.array([0, 0, 0, 0])
        proba = np.array([[0.0, 1.0]] * 4)
        assert ExpectedCalibrationError()(y_true, proba) == pytest.approx(1.0)

    def test_accuracy_1d_multiclass_labels(self):
        """1-D integer labels must not be thresholded at 0.5 like probabilities."""
        y = np.array([0, 1, 2, 2])
        assert Accuracy()(y, y) == pytest.approx(1.0)

    def test_f1_1d_multiclass_labels(self):
        y = np.array([0, 1, 2, 2])
        assert F1Score(average="macro")(y, y) == pytest.approx(1.0)

    def test_ece_rejects_raw_logits(self):
        """Logits (unbounded) must be rejected, not silently mis-binned."""
        y_true = np.array([0, 1])
        logits = np.array([[-2.1, 4.7], [1.8, -0.6]])
        with pytest.raises(ValueError, match="probabilities"):
            ExpectedCalibrationError()(y_true, logits)

    def test_ece_rejects_out_of_range_class_indices(self):
        y_true = np.array([0, 3])
        proba = np.array([[0.9, 0.1], [0.1, 0.9]])
        with pytest.raises(ValueError, match="class indices"):
            ExpectedCalibrationError()(y_true, proba)

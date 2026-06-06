"""Tests for class-imbalance handling in DeepTab classifiers.

Covers the ``compute_class_weights`` / ``build_weighted_classification_loss``
helpers and the ``class_weight`` / ``loss_fct`` arguments threaded through the
classifier ``fit`` API.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from deeptab.models import MLPClassifier
from deeptab.training.losses import (
    BaseLoss,
    FocalLoss,
    WeightedBCEWithLogitsLoss,
    WeightedCrossEntropyLoss,
    build_classification_loss,
    build_weighted_classification_loss,
    compute_class_weights,
    get_loss,
)

RANDOM_STATE = 0
FIT_KWARGS: dict[str, Any] = {"max_epochs": 2, "batch_size": 64}


# ---------------------------------------------------------------------------
# compute_class_weights
# ---------------------------------------------------------------------------


class TestComputeClassWeights:
    def test_none_returns_none(self):
        assert compute_class_weights(None, np.array([0, 1, 1])) is None

    def test_balanced_matches_sklearn_formula(self):
        # 90 zeros, 10 ones -> n_samples / (n_classes * count)
        y = np.array([0] * 90 + [1] * 10)
        weights = compute_class_weights("balanced", y)
        assert weights is not None
        expected = np.array([100 / (2 * 90), 100 / (2 * 10)])
        np.testing.assert_allclose(weights, expected)

    def test_balanced_matches_sklearn_reference(self):
        sklearn_cw = pytest.importorskip("sklearn.utils.class_weight")
        y = np.array([0] * 70 + [1] * 20 + [2] * 10)
        classes = np.unique(y)
        expected = sklearn_cw.compute_class_weight("balanced", classes=classes, y=y)
        weights = compute_class_weights("balanced", y, classes=classes)
        assert weights is not None
        np.testing.assert_allclose(weights, expected)

    def test_mapping_uses_defaults_for_missing(self):
        y = np.array([0, 1, 2])
        weights = compute_class_weights({0: 2.0, 2: 3.0}, y)
        assert weights is not None
        np.testing.assert_allclose(weights, np.array([2.0, 1.0, 3.0]))

    def test_array_like_passed_through(self):
        y = np.array([0, 1])
        weights = compute_class_weights([0.25, 0.75], y)
        assert weights is not None
        np.testing.assert_allclose(weights, np.array([0.25, 0.75]))

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Unsupported class_weight"):
            compute_class_weights("auto", np.array([0, 1]))

    def test_array_wrong_length_raises(self):
        with pytest.raises(ValueError, match="length"):
            compute_class_weights([1.0, 2.0, 3.0], np.array([0, 1]))

    def test_balanced_zero_count_raises(self):
        y = np.array([0, 0, 0])
        classes = np.array([0, 1])
        with pytest.raises(ValueError, match="zero samples"):
            compute_class_weights("balanced", y, classes=classes)


# ---------------------------------------------------------------------------
# build_weighted_classification_loss
# ---------------------------------------------------------------------------


class TestBuildWeightedLoss:
    def test_none_returns_none(self):
        assert build_weighted_classification_loss(None, num_classes=2) is None

    def test_binary_returns_bce_with_pos_weight(self):
        weights = np.array([0.5, 2.0])
        loss = build_weighted_classification_loss(weights, num_classes=2)
        assert isinstance(loss, WeightedBCEWithLogitsLoss)
        assert loss.pos_weight is not None
        # pos_weight = w[1] / w[0]
        torch.testing.assert_close(loss.pos_weight, torch.tensor([4.0]))

    def test_multiclass_returns_cross_entropy_with_weight(self):
        weights = np.array([1.0, 2.0, 3.0])
        loss = build_weighted_classification_loss(weights, num_classes=3)
        assert isinstance(loss, WeightedCrossEntropyLoss)
        assert loss.weight is not None
        torch.testing.assert_close(loss.weight, torch.tensor([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Integration with the classifier API
# ---------------------------------------------------------------------------


def _imbalanced_binary_data(pos_fraction: float = 0.1):
    rng = np.random.default_rng(RANDOM_STATE)
    n = 200
    n_features = 5
    X = rng.standard_normal((n, n_features))
    n_pos = int(n * pos_fraction)
    y = np.array([1] * n_pos + [0] * (n - n_pos))
    rng.shuffle(y)
    df = pd.DataFrame({f"f{i}": X[:, i] for i in range(n_features)})
    return df, y


def _imbalanced_multiclass_data():
    rng = np.random.default_rng(RANDOM_STATE)
    n_features = 5
    y = np.array([0] * 120 + [1] * 50 + [2] * 30)
    X = rng.standard_normal((len(y), n_features))
    df = pd.DataFrame({f"f{i}": X[:, i] for i in range(n_features)})
    return df, y


class TestClassifierClassWeight:
    def test_balanced_binary_sets_pos_weight(self):
        X, y = _imbalanced_binary_data()
        clf = MLPClassifier()
        clf.fit(X, y, class_weight="balanced", random_state=RANDOM_STATE, **FIT_KWARGS)

        assert clf.task_model is not None
        loss = clf.task_model.loss_fct
        assert isinstance(loss, WeightedBCEWithLogitsLoss)
        assert loss.pos_weight is not None
        # minority (positive) class should be up-weighted -> pos_weight > 1
        assert loss.pos_weight.item() > 1.0

    def test_balanced_multiclass_sets_weight(self):
        X, y = _imbalanced_multiclass_data()
        clf = MLPClassifier()
        clf.fit(X, y, class_weight="balanced", random_state=RANDOM_STATE, **FIT_KWARGS)

        assert clf.task_model is not None
        loss = clf.task_model.loss_fct
        assert isinstance(loss, WeightedCrossEntropyLoss)
        assert loss.weight is not None
        assert loss.weight.shape[0] == 3
        # rarest class (label 2) should have the largest weight
        assert torch.argmax(loss.weight).item() == 2

    def test_default_has_no_class_weighting(self):
        X, y = _imbalanced_binary_data()
        clf = MLPClassifier()
        clf.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)

        assert clf.task_model is not None
        loss = clf.task_model.loss_fct
        assert isinstance(loss, nn.BCEWithLogitsLoss)
        assert loss.pos_weight is None

    def test_explicit_loss_fct_overrides_class_weight(self):
        X, y = _imbalanced_binary_data()
        custom = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([7.0]))
        clf = MLPClassifier()
        clf.fit(
            X,
            y,
            class_weight="balanced",
            loss_fct=custom,
            random_state=RANDOM_STATE,
            **FIT_KWARGS,
        )

        assert clf.task_model is not None
        loss = clf.task_model.loss_fct
        assert loss is custom
        assert isinstance(loss, nn.BCEWithLogitsLoss)
        torch.testing.assert_close(loss.pos_weight, torch.tensor([7.0]))

    def test_balanced_classifier_predicts(self):
        X, y = _imbalanced_binary_data()
        clf = MLPClassifier()
        clf.fit(X, y, class_weight="balanced", random_state=RANDOM_STATE, **FIT_KWARGS)
        preds = clf.predict(X)
        assert len(preds) == len(y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 2)


# ---------------------------------------------------------------------------
# Loss registry
# ---------------------------------------------------------------------------


class TestLossRegistry:
    def test_builtin_losses_registered(self):
        for name in ("bce", "cross_entropy", "focal"):
            assert name in BaseLoss.available()

    def test_get_loss_returns_class(self):
        assert get_loss("focal") is FocalLoss
        assert get_loss("bce") is WeightedBCEWithLogitsLoss
        assert get_loss("cross_entropy") is WeightedCrossEntropyLoss

    def test_get_loss_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown loss"):
            get_loss("does_not_exist")

    def test_subclass_auto_registers(self):
        class _CustomDummyLoss(BaseLoss, name="dummy_test_loss"):
            def forward(self, logits, targets):
                return logits.sum() * 0.0

        try:
            assert "dummy_test_loss" in BaseLoss.available()
            assert get_loss("dummy_test_loss") is _CustomDummyLoss
        finally:
            BaseLoss._registry.pop("dummy_test_loss", None)


# ---------------------------------------------------------------------------
# build_classification_loss resolver
# ---------------------------------------------------------------------------


class TestBuildClassificationLoss:
    def test_none_without_weights_returns_none(self):
        assert build_classification_loss(None, num_classes=2) is None

    def test_module_passed_through(self):
        custom = nn.BCEWithLogitsLoss()
        assert build_classification_loss(custom, num_classes=2) is custom

    def test_string_focal_binary(self):
        loss = build_classification_loss("focal", num_classes=2)
        assert isinstance(loss, FocalLoss)
        assert loss.expects_class_indices is False

    def test_string_focal_multiclass(self):
        loss = build_classification_loss("focal", num_classes=3)
        assert isinstance(loss, FocalLoss)
        assert loss.expects_class_indices is True

    def test_string_focal_with_class_weights_binary_alpha(self):
        weights = np.array([0.5, 2.0])
        loss = build_classification_loss("focal", num_classes=2, class_weights=weights)
        assert isinstance(loss, FocalLoss)
        # alpha = w[1] / (w[0] + w[1]) = 2.0 / 2.5 = 0.8
        assert loss.alpha_scalar == pytest.approx(0.8)

    def test_string_focal_with_class_weights_multiclass_alpha(self):
        weights = np.array([1.0, 2.0, 3.0])
        loss = build_classification_loss("focal", num_classes=3, class_weights=weights)
        assert isinstance(loss, FocalLoss)
        assert loss.alpha_weight is not None
        torch.testing.assert_close(loss.alpha_weight, torch.tensor([1.0, 2.0, 3.0]))

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="loss must be"):
            build_classification_loss(123, num_classes=2)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FocalLoss numerics
# ---------------------------------------------------------------------------


class TestFocalLoss:
    def test_gamma_zero_binary_matches_bce(self):
        torch.manual_seed(0)
        logits = torch.randn(32, 1)
        targets = (torch.rand(32, 1) > 0.5).float()
        focal = FocalLoss(gamma=0.0, num_classes=2)
        bce = nn.BCEWithLogitsLoss()
        torch.testing.assert_close(focal(logits, targets), bce(logits, targets))

    def test_gamma_zero_multiclass_matches_cross_entropy(self):
        torch.manual_seed(0)
        logits = torch.randn(32, 4)
        targets = torch.randint(0, 4, (32,))
        focal = FocalLoss(gamma=0.0, num_classes=4)
        ce = nn.CrossEntropyLoss()
        torch.testing.assert_close(focal(logits, targets), ce(logits, targets))

    def test_positive_gamma_downweights_easy_examples(self):
        # Confident-correct predictions -> focal loss should be far below CE.
        logits = torch.tensor([[5.0], [5.0], [5.0]])
        targets = torch.ones(3, 1)
        focal = FocalLoss(gamma=2.0, num_classes=2)(logits, targets)
        bce = nn.BCEWithLogitsLoss()(logits, targets)
        assert focal.item() < bce.item()

    def test_returns_scalar(self):
        loss = FocalLoss(gamma=2.0, num_classes=3)(torch.randn(8, 3), torch.randint(0, 3, (8,)))
        assert loss.ndim == 0


class TestClassifierFocalLoss:
    def test_focal_string_binary(self):
        X, y = _imbalanced_binary_data()
        clf = MLPClassifier()
        clf.fit(X, y, loss_fct="focal", class_weight="balanced", random_state=RANDOM_STATE, **FIT_KWARGS)
        assert clf.task_model is not None
        assert isinstance(clf.task_model.loss_fct, FocalLoss)
        assert clf.predict(X).shape[0] == len(y)

    def test_focal_string_multiclass(self):
        X, y = _imbalanced_multiclass_data()
        clf = MLPClassifier()
        clf.fit(X, y, loss_fct="focal", random_state=RANDOM_STATE, **FIT_KWARGS)
        assert clf.task_model is not None
        loss = clf.task_model.loss_fct
        assert isinstance(loss, FocalLoss)
        assert loss.expects_class_indices is True
        assert clf.predict_proba(X).shape == (len(y), 3)


# ---------------------------------------------------------------------------
# Weighted sampling
# ---------------------------------------------------------------------------


class TestWeightedSampling:
    def test_balanced_sampler_builds_weighted_sampler(self):
        from torch.utils.data import WeightedRandomSampler

        X, y = _imbalanced_binary_data()
        clf = MLPClassifier()
        clf.fit(X, y, balanced_sampler=True, random_state=RANDOM_STATE, **FIT_KWARGS)
        sampler = clf.data_module._build_train_sampler()
        assert isinstance(sampler, WeightedRandomSampler)
        # Minority rows must carry larger sampling weight than majority rows.
        weights = np.asarray(sampler.weights)
        y_train = np.asarray(clf.data_module.y_train)
        minority_w = weights[y_train == 1].mean()
        majority_w = weights[y_train == 0].mean()
        assert minority_w > majority_w

    def test_no_sampler_by_default(self):
        X, y = _imbalanced_binary_data()
        clf = MLPClassifier()
        clf.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)
        assert clf.data_module._build_train_sampler() is None

    def test_explicit_sample_weight_split_aligns(self):
        X, y = _imbalanced_binary_data()
        sample_weight = np.linspace(1.0, 2.0, num=len(y))
        clf = MLPClassifier()
        clf.fit(X, y, sample_weight=sample_weight, random_state=RANDOM_STATE, **FIT_KWARGS)
        train_weights = clf.data_module._train_sample_weights
        assert train_weights is not None
        # Weights were split alongside the train/val partition.
        assert clf.data_module is not None
        assert len(train_weights) == len(clf.data_module.y_train)  # type: ignore[arg-type]

    def test_sample_weight_wrong_length_raises(self):
        X, y = _imbalanced_binary_data()
        clf = MLPClassifier()
        with pytest.raises(ValueError, match="sample_weight"):
            clf.fit(X, y, sample_weight=np.ones(len(y) + 5), random_state=RANDOM_STATE, **FIT_KWARGS)

    def test_balanced_sampler_classifier_predicts(self):
        X, y = _imbalanced_multiclass_data()
        clf = MLPClassifier()
        clf.fit(X, y, balanced_sampler=True, random_state=RANDOM_STATE, **FIT_KWARGS)
        assert clf.predict_proba(X).shape == (len(y), 3)


# ---------------------------------------------------------------------------
# Ensemble dispatch (compute_loss must route weighted CE through the ensemble path)
# ---------------------------------------------------------------------------


class TestEnsembleWeightedLoss:
    def test_ensemble_multiclass_weighted_cross_entropy(self):
        from deeptab.models import TabMClassifier

        X, y = _imbalanced_multiclass_data()
        clf = TabMClassifier()
        clf.fit(X, y, class_weight="balanced", random_state=RANDOM_STATE, **FIT_KWARGS)
        assert clf.task_model is not None
        assert isinstance(clf.task_model.loss_fct, WeightedCrossEntropyLoss)
        assert getattr(clf.task_model.estimator, "returns_ensemble", False) is True
        assert clf.predict_proba(X).shape == (len(y), 3)

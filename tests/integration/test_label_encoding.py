"""Regression tests for issue #409: classifier labels were never encoded to
``0..K-1`` before reaching the loss/preprocessor, which crashed on string
labels and silently trained on the wrong targets for non-contiguous or
non-``{0,1}`` binary labels.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from deeptab.models import MLPClassifier
from deeptab.models.classifier_base import _encode_labels

RANDOM_STATE = 0
FIT_KWARGS: dict[str, Any] = {"max_epochs": 5, "batch_size": 64}


# ---------------------------------------------------------------------------
# _encode_labels unit tests
# ---------------------------------------------------------------------------


class TestEncodeLabels:
    def test_maps_to_contiguous_indices(self):
        classes = np.array([5, 7])
        y = np.array([5, 7, 5, 7])
        np.testing.assert_array_equal(_encode_labels(y, classes), [0, 1, 0, 1])

    def test_string_labels(self):
        classes = np.array(["cat", "dog", "fish"])
        y = np.array(["dog", "cat", "fish", "cat"])
        np.testing.assert_array_equal(_encode_labels(y, classes), [1, 0, 2, 0])

    def test_unseen_label_raises(self):
        classes = np.array([0, 1])
        with pytest.raises(ValueError, match=r"contains label\(s\) \[2\]"):
            _encode_labels(np.array([0, 1, 2]), classes, name="y_val")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _separable_binary_data(labels: tuple):
    rng = np.random.default_rng(RANDOM_STATE)
    n = 200
    num1 = rng.standard_normal(n)
    y = np.where(num1 > 0, labels[1], labels[0])
    X = pd.DataFrame({"num1": num1})
    return X, np.asarray(y)


def _multiclass_data(labels: tuple):
    rng = np.random.default_rng(RANDOM_STATE)
    n = 240
    num1 = rng.standard_normal(n)
    y = np.array([labels[i % 3] for i in range(n)])
    X = pd.DataFrame({"num1": num1, "num2": rng.standard_normal(n)})
    return X, y


# ---------------------------------------------------------------------------
# Integration: fit/predict round-trip through the classifier API
# ---------------------------------------------------------------------------


class TestClassifierLabelEncoding:
    def test_string_binary_labels_round_trip(self):
        X, y = _separable_binary_data(("no", "yes"))
        clf = MLPClassifier()
        clf.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)

        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({"no", "yes"})

        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 2)

    def test_string_multiclass_labels_round_trip(self):
        X, y = _multiclass_data(("cat", "dog", "fish"))
        clf = MLPClassifier()
        clf.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)

        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({"cat", "dog", "fish"})

        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 3)

    def test_non_contiguous_integer_labels_do_not_crash(self):
        X, y = _multiclass_data((10, 20, 30))
        clf = MLPClassifier()
        clf.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)

        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({10, 20, 30})

    def test_binary_labels_other_than_zero_one_are_accurate(self):
        X, y = _separable_binary_data((5, 7))
        clf = MLPClassifier()
        clf.fit(X, y, random_state=RANDOM_STATE, max_epochs=30, batch_size=32)

        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({5, 7})
        # Perfectly separable data: encoded labels should let the model learn
        # the true decision boundary rather than regressing against raw 5.0/7.0.
        accuracy = (preds == y).mean()
        assert accuracy > 0.9

    def test_plain_zero_to_k_minus_one_labels_unaffected(self):
        X, y = _multiclass_data((0, 1, 2))
        clf = MLPClassifier()
        clf.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)

        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({0, 1, 2})

    def test_unseen_validation_label_raises_clear_error(self):
        X, y = _multiclass_data(("cat", "dog", "fish"))
        X_train, y_train = X.iloc[:200], y[:200]
        X_val = X.iloc[200:]
        y_val = np.array(["cat", "dog", "bird"] * (len(X_val) // 3 + 1))[: len(X_val)]

        clf = MLPClassifier()
        with pytest.raises(ValueError, match="not seen during fit"):
            clf.fit(X_train, y_train, X_val=X_val, y_val=y_val, random_state=RANDOM_STATE, **FIT_KWARGS)


# ---------------------------------------------------------------------------
# (n,1) column-vector y must behave identically to (n,) y through the full
# fit/predict API, across label dtypes and binary/multiclass tasks.
# ---------------------------------------------------------------------------


class TestColumnVectorTargets:
    def test_multiclass_column_vector_matches_1d(self):
        X, y = _multiclass_data((0, 1, 2))

        clf_1d = MLPClassifier()
        clf_1d.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)
        preds_1d = clf_1d.predict(X)

        clf_2d = MLPClassifier()
        clf_2d.fit(X, y.reshape(-1, 1), random_state=RANDOM_STATE, **FIT_KWARGS)
        preds_2d = clf_2d.predict(X)

        assert preds_2d.shape == preds_1d.shape == (len(y),)
        np.testing.assert_array_equal(preds_1d, preds_2d)

    def test_string_binary_column_vector_round_trip(self):
        X, y = _separable_binary_data(("no", "yes"))
        clf = MLPClassifier()
        clf.fit(X, y.reshape(-1, 1), random_state=RANDOM_STATE, **FIT_KWARGS)

        preds = clf.predict(X)
        assert preds.shape == (len(y),)
        assert set(np.unique(preds)).issubset({"no", "yes"})

    def test_string_multiclass_column_vector_round_trip(self):
        X, y = _multiclass_data(("cat", "dog", "fish"))
        clf = MLPClassifier()
        clf.fit(X, y.reshape(-1, 1), random_state=RANDOM_STATE, **FIT_KWARGS)

        preds = clf.predict(X)
        assert preds.shape == (len(y),)
        assert set(np.unique(preds)).issubset({"cat", "dog", "fish"})

    def test_non_contiguous_integer_column_vector_does_not_crash(self):
        X, y = _multiclass_data((10, 20, 30))
        clf = MLPClassifier()
        clf.fit(X, y.reshape(-1, 1), random_state=RANDOM_STATE, **FIT_KWARGS)

        preds = clf.predict(X)
        assert preds.shape == (len(y),)
        assert set(np.unique(preds)).issubset({10, 20, 30})

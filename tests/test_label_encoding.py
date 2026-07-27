"""Regression tests: classifier labels must be encoded to 0..K-1 for training.

Before the fix, raw label values were used directly as CrossEntropy indices /
BCE targets: string labels crashed in the preprocessor, non-contiguous integer
labels crashed on CPU (or silently trained on garbage on MPS), and binary
labels other than {0, 1} silently produced BCE targets like 5.0/7.0.
"""

import numpy as np
import pandas as pd
import pytest

from deeptab.models import MLPClassifier

FIT_KW = dict(max_epochs=2, batch_size=16, accelerator="cpu")


def _make_X(n=80, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"num1": rng.randn(n), "cat1": rng.choice(["a", "b", "c"], n)})


class TestLabelEncoding:
    def test_binary_string_labels_round_trip(self):
        X = _make_X()
        y = np.random.RandomState(1).choice(["no", "yes"], 80)
        clf = MLPClassifier()
        clf.fit(X, y, **FIT_KW)
        preds = clf.predict(X)
        assert set(preds) <= {"no", "yes"}
        proba = clf.predict_proba(X)
        assert proba.shape == (80, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_multiclass_string_labels_round_trip(self):
        X = _make_X()
        y = np.random.RandomState(2).choice(["low", "mid", "high"], 80)
        clf = MLPClassifier()
        clf.fit(X, y, **FIT_KW)
        assert set(clf.predict(X)) <= {"low", "mid", "high"}
        assert clf.predict_proba(X).shape == (80, 3)

    def test_non_contiguous_int_labels(self):
        """CE previously received the raw values (e.g. target 30 with 3 logits)."""
        X = _make_X()
        y = np.random.RandomState(3).choice([10, 20, 30], 80)
        clf = MLPClassifier()
        clf.fit(X, y, **FIT_KW)
        assert set(clf.predict(X)) <= {10, 20, 30}

    def test_binary_non_01_labels_learn_signal(self):
        """BCE previously trained against raw targets 5.0/7.0 -> worse than chance."""
        rng = np.random.RandomState(0)
        X = pd.DataFrame({"num1": rng.randn(400)})
        y = np.where(X["num1"] > 0, 7, 5)
        clf = MLPClassifier()
        clf.fit(X, y, max_epochs=30, batch_size=32, accelerator="cpu")
        assert (clf.predict(X) == y).mean() > 0.9

    def test_unseen_y_val_label_raises(self):
        X = _make_X()
        y = np.random.RandomState(4).choice(["no", "yes"], 80)
        y_val = np.array(["no", "yes", "maybe", "yes"] * 5)
        clf = MLPClassifier()
        with pytest.raises(ValueError, match="labels not present"):
            clf.fit(X, y, X_val=_make_X(20, 5), y_val=y_val, **FIT_KW)

    def test_contiguous_int_labels_unchanged(self):
        X = _make_X()
        y = np.random.RandomState(5).choice([0, 1, 2], 80)
        clf = MLPClassifier()
        clf.fit(X, y, **FIT_KW)
        assert set(clf.predict(X)) <= {0, 1, 2}

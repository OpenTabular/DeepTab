"""Tests for estimator input validation."""

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("family", ["inversegamma", "lognormal"])
@pytest.mark.parametrize("invalid_target", [0.0, -1.0])
def test_lss_rejects_non_positive_targets_before_building(family, invalid_target):
    from deeptab.core.exceptions import DataError
    from deeptab.models import MLPLSS

    features = np.ones((6, 2))
    targets = np.array([1.0, 2.0, invalid_target, 3.0, 4.0, 5.0])
    model = MLPLSS()
    with pytest.raises(DataError, match="strictly positive"):
        model.fit(features, targets, family=family, max_epochs=1)
    assert model._data_module is None
    assert model._built is False


@pytest.mark.parametrize(
    "family, invalid_target",
    [("poisson", 1.5), ("negativebinom", 2.5), ("zip", 0.5), ("tweedie", -1.0), ("beta", 1.0)],
)
def test_lss_rejects_targets_outside_family_support_before_building(family, invalid_target):
    from deeptab.core.exceptions import DataError
    from deeptab.models import MLPLSS

    features = np.ones((6, 2))
    targets = np.array([0.0, 1.0, invalid_target, 0.0, 1.0, 0.0])
    if family == "beta":
        targets = np.array([0.2, 0.4, invalid_target, 0.6, 0.8, 0.5])
    model = MLPLSS()
    with pytest.raises(DataError, match=f"family='{family}'"):
        model.fit(features, targets, family=family, max_epochs=1)
    assert model._data_module is None
    assert model._built is False


class TestEdgeCaseInputs:
    """fit() with input shapes that previously crashed instead of failing cleanly or working."""

    @pytest.mark.parametrize("method", ["fit", "build_model"])
    @pytest.mark.parametrize("label", [0, 1, "only"])
    def test_single_class_is_rejected_before_building(self, method, label):
        from deeptab.models import MLPClassifier

        model = MLPClassifier()
        with pytest.raises(ValueError, match="at least 2 classes"):
            getattr(model, method)(np.ones((6, 2)), np.full(6, label))
        assert model._data_module is None
        assert model._task_model is None
        assert not model._built
        assert not hasattr(model, "classes_")

    @pytest.mark.parametrize("string_labels", [False, True])
    def test_fit_with_rare_class_only_in_unstratified_validation(self, string_labels, tmp_path):
        from sklearn.model_selection import train_test_split

        from deeptab.configs import MLPConfig, TrainerConfig
        from deeptab.data import TabularDataModule
        from deeptab.models import MLPClassifier

        features = np.random.default_rng(0).normal(size=(30, 2))
        targets = np.tile([0, 1], 15)
        _, validation_indices = train_test_split(np.arange(30), test_size=0.2, random_state=42)
        targets[validation_indices[0]] = 2
        if string_labels:
            targets = np.array(["common", "other", "rare"])[targets]
        model = MLPClassifier(
            model_config=MLPConfig(layer_sizes=[16]),
            trainer_config=TrainerConfig(max_epochs=1, batch_size=8, checkpoint_path=str(tmp_path)),
            random_state=42,
        )
        model.fit(features, targets, stratify=False, accelerator="cpu", logger=False, enable_progress_bar=False)
        assert isinstance(model._data_module, TabularDataModule)
        assert model._data_module.num_classes == 3
        assert model._data_module.y_train is not None
        assert len(np.unique(model._data_module.y_train)) == 2
        probabilities = model.predict_proba(features)
        assert probabilities.shape == (30, 3)
        assert np.isfinite(probabilities).all()

    def test_duplicate_columns_raise_duplicate_columns_error(self):
        from deeptab.core.exceptions import DuplicateColumnsError
        from deeptab.models import MLPRegressor

        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.standard_normal(40), "b": rng.standard_normal(40)})
        X.columns = ["a", "a"]
        y = rng.standard_normal(40)

        with pytest.raises(DuplicateColumnsError, match="a"):
            MLPRegressor().fit(X, y, max_epochs=1, batch_size=16)

    def test_list_of_lists_input_fits_successfully(self):
        from deeptab.models import MLPRegressor

        rng = np.random.default_rng(0)
        X = [[float(rng.standard_normal()), float(rng.standard_normal())] for _ in range(40)]
        y = list(rng.standard_normal(40))

        model = MLPRegressor()
        model.fit(X, y, max_epochs=1, batch_size=16)
        preds = model.predict(X)
        assert preds.shape == (40,)

    def test_all_missing_column_fits_and_predicts(self):
        from deeptab.models import MLPRegressor

        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.standard_normal(40), "allnan": np.full(40, np.nan)})
        y = rng.standard_normal(40)

        model = MLPRegressor()
        with pytest.warns(match="entirely NaN"):
            model.fit(X, y, max_epochs=1, batch_size=16)
        preds = model.predict(X)
        assert preds.shape == (40,)


class TestIntegerColumnLabels:
    """Integer column labels that are not 0..n-1 used to be read by position for
    selection but by label for type detection, crashing or encoding the wrong column."""

    @staticmethod
    def _fit_predict(features, targets):
        from deeptab.configs import TrainerConfig
        from deeptab.models import MLPRegressor

        model = MLPRegressor(trainer_config=TrainerConfig(max_epochs=1, batch_size=32), random_state=0)
        model.fit(features, targets, accelerator="cpu", logger=False, enable_progress_bar=False)
        return model, model.predict(features)

    def test_frame_after_dropping_a_positional_column(self):
        rng = np.random.default_rng(0)
        data = pd.DataFrame(rng.normal(size=(96, 4)))
        features, targets = data.drop(columns=0), data[0].to_numpy()
        _, predictions = self._fit_predict(features, targets)
        _, expected = self._fit_predict(features.rename(columns=str), targets)
        np.testing.assert_allclose(predictions, expected, rtol=1e-5)

    def test_reordered_integer_labels_keep_column_types(self):
        rng = np.random.default_rng(0)
        colours = np.array(["red", "green", "blue"])
        codes = rng.integers(0, 3, 96)
        features = pd.DataFrame({1: rng.normal(size=96), 0: colours[codes]})
        targets = features[1].to_numpy() * 3 + codes
        model, predictions = self._fit_predict(features, targets)
        string_model, expected = self._fit_predict(features.rename(columns=str), targets)
        assert model._data_module is not None
        assert string_model._data_module is not None
        categorical = model._data_module.cat_feature_info
        string_categorical = string_model._data_module.cat_feature_info
        assert categorical
        assert string_categorical
        assert [info["categories"] for info in categorical.values()] == [
            info["categories"] for info in string_categorical.values()
        ]
        np.testing.assert_allclose(predictions, expected, rtol=1e-5)

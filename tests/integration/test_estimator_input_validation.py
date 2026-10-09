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


class TestEdgeCaseInputs:
    """fit() with input shapes that previously crashed instead of failing cleanly or working."""

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

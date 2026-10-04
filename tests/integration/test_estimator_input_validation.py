"""Tests for estimator input validation."""

import numpy as np
import pandas as pd
import pytest


class TestEdgeCaseInputs:
    """fit() with input shapes that previously crashed instead of failing cleanly or working."""

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

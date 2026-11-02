"""Representative stable wrappers must handle or reject data boundaries explicitly."""

import numpy as np
import pytest

from deeptab.core.exceptions import ArchitectureRequirementError, DataError, NotFittedError

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("architecture", ["MLP", "TabM", "FTTransformer", "TabTransformer"])
@pytest.mark.parametrize("task", ["classification", "regression", "lss"])
@pytest.mark.parametrize(
    "case", ["one_feature", "categorical_only", "unseen_category", "missing_value", "infinity", "invalid_target"]
)
def test_stable_data_boundary(architecture, task, case, stable_model_factory, easy_problem):
    train, query, train_classes, _, train_values, _ = easy_problem
    targets = train_classes if task == "classification" else train_values
    model = stable_model_factory(architecture, task, epochs=1)
    options = {
        "accelerator": "cpu",
        "logger": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        **({"family": "normal"} if task == "lss" else {}),
    }
    train, query = train.copy(), query.copy()
    if case == "one_feature":
        train, query = train[["signal"]], query[["signal"]]
    elif case == "categorical_only":
        train, query = train[["category"]], query[["category"]]
    elif case == "missing_value":
        train.loc[train.index[0], "signal"] = np.nan
        query.loc[query.index[0], "signal"] = np.nan
    if case == "one_feature" and architecture == "TabTransformer":
        with pytest.raises(ArchitectureRequirementError, match="categorical"):
            model.fit(train, targets, **options)
        assert not model.is_fitted_
        return
    assert model.fit(train, targets, **options) is model
    expected = model.predict(query)
    assert np.isfinite(expected).all() and len(expected) == len(query)
    if case == "unseen_category":
        query["category"] = "previously-unseen"
        actual = model.predict(query)
        assert actual.shape == expected.shape and np.isfinite(actual).all()
    elif case in {"infinity", "invalid_target"}:
        previous = model._task_model
        invalid_targets = np.asarray(targets, dtype=float).copy()
        if case == "infinity":
            invalid_train = train.copy()
            invalid_train.loc[invalid_train.index[0], "signal"] = np.inf
            invalid_query = query.copy()
            invalid_query.loc[invalid_query.index[0], "signal"] = -np.inf
            with pytest.raises((DataError, ValueError), match=r"(?i)finite|inf"):
                model.predict(invalid_query)
        else:
            invalid_train = train
            invalid_targets[0] = np.inf
        with pytest.raises((DataError, ValueError), match=r"(?i)finite|inf"):
            model.fit(invalid_train, invalid_targets, **options)
        if case == "invalid_target":
            assert model.is_fitted_ and model._task_model is previous
            np.testing.assert_array_equal(model.predict(query), expected)
        else:
            assert not model.is_fitted_
            with pytest.raises(NotFittedError):
                model.predict(query)


@pytest.mark.parametrize(
    "architecture, options",
    [
        pytest.param("MLP", {"use_embeddings": False, "layer_norm": True}, id="raw-mlp"),
        pytest.param("MLP", {"use_embeddings": True, "layer_norm": True}, id="embedded-mlp"),
        pytest.param("FTTransformer", {"pooling_method": "avg", "norm": "LayerNorm"}, id="average-layernorm"),
        pytest.param("FTTransformer", {"pooling_method": "max", "norm": "RMSNorm"}, id="max-rmsnorm"),
    ],
)
@pytest.mark.parametrize("preprocessing", ["standardization", "ple"])
def test_external_embedding_preprocessing_and_normalization_reload(
    architecture, options, preprocessing, stable_model_factory, easy_problem, tmp_path
):
    train, query, _, _, targets, _ = easy_problem
    model = stable_model_factory(
        architecture,
        "regression",
        epochs=2,
        model_options=options,
        preprocessing_options={"numerical_method": preprocessing, "output_dim": 7},
    )
    train_vectors = np.column_stack([targets, targets**2]).astype(np.float32)
    query_vectors = np.column_stack([query["signal"], query["signal"] ** 2]).astype(np.float32)
    model.fit(
        train,
        targets,
        embeddings=train_vectors,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    expected = model.predict(query, embeddings=query_vectors)
    assert expected.shape == (len(query),) and np.isfinite(expected).all()
    assert model._data_module.embedding_feature_info
    path = str(tmp_path / "interaction.deeptab")
    model.save(path)
    actual = type(model).load(path, device="cpu").predict(query, embeddings=query_vectors)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        model.predict(query.iloc[:1], embeddings=query_vectors[:1]), expected[:1], rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("dtype", ["string", "convert_dtypes"])
def test_pandas_string_columns_with_missing_values(dtype):
    import pandas as pd

    from deeptab.configs import TrainerConfig
    from deeptab.models import MLPRegressor

    rng = np.random.default_rng(0)
    values = rng.choice(["berlin", "paris", "rome"], 80).astype(object)
    values[::7] = None
    reference = pd.DataFrame({"a": rng.normal(size=80), "city": values})
    targets = reference["a"].to_numpy() * 2
    features = reference.convert_dtypes() if dtype == "convert_dtypes" else reference.astype({"city": "string"})
    assert isinstance(features["city"].dtype, pd.StringDtype)
    fit_kwargs = {"accelerator": "cpu", "logger": False, "enable_progress_bar": False}

    predictions = []
    for frame in (features, reference):
        model = MLPRegressor(trainer_config=TrainerConfig(max_epochs=1, batch_size=16), random_state=0)
        predictions.append(model.fit(frame, targets, **fit_kwargs).predict(frame.iloc[:10]))
    np.testing.assert_allclose(predictions[0], predictions[1], rtol=1e-6)

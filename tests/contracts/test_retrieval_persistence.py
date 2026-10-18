"""Portable candidate-bank contracts and independent-process prediction parity."""

import copy
import json
import os
import platform
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from deeptab.architectures.tabr import TabR
from deeptab.configs import PreprocessingConfig, TabRConfig, TrainerConfig
from deeptab.configs.experimental.modernnca_config import ModernNCAConfig
from deeptab.core.serialization import build_save_bundle
from deeptab.models import TabRClassifier, TabRLSS, TabRRegressor
from deeptab.models.experimental import ModernNCAClassifier, ModernNCALSS, ModernNCARegressor
from deeptab.training import TaskModel

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
KINDS = ["regression", "binary", "multiclass", "normal", "categorical", "dirichlet"]


def _exact_index():
    helpers = runpy.run_path(str(ROOT / "tests/architectures/test_tabr_architecture.py"))
    dependencies = {
        "delu": SimpleNamespace(
            iter_batches=lambda rows, size: rows.split(size), nn=SimpleNamespace(Lambda=helpers["_Lambda"])
        ),
        "faiss": SimpleNamespace(IndexFlatL2=helpers["_ExactL2Index"]),
        "faiss_torch_utils": object(),
    }
    for name, replacement in dependencies.items():
        setattr(TabR, name, replacement)


def _fit_case(kind, architecture, directory, continuation=False):
    features = pd.DataFrame(
        np.random.default_rng(17).normal(size=(32, 3)).astype(np.float32),
        columns=pd.Index(["first", "second", "third"]),
    )
    features["category"] = np.array(["red", "blue", "green", "blue"] * 8, dtype=object)
    embeddings = np.random.default_rng(19).normal(size=(32, 2)).astype(np.float32)
    targets = features["first"].to_numpy().copy()
    if kind in {"binary", "multiclass", "categorical"}:
        targets = (np.arange(32) % (2 if kind == "binary" else 3)) * 20 + 10
    elif kind == "dirichlet":
        targets = np.random.default_rng(23).dirichlet([2, 3, 4], size=32).astype(np.float32)
    position = 0 if kind == "regression" else (1 if kind in {"binary", "multiclass"} else 2)
    if architecture == "modernnca":
        estimator = [ModernNCARegressor, ModernNCAClassifier, ModernNCALSS][position]
        config = ModernNCAConfig(dim=8, n_blocks=0, dropout=0, head_dropout=0)
    else:
        estimator = [TabRRegressor, TabRClassifier, TabRLSS][position]
        config = TabRConfig(
            d_main=8,
            context_size=4,
            encoder_n_blocks=0,
            predictor_n_blocks=0,
            mixer_normalization="auto",
            dropout0=0,
            dropout1=0,
            context_dropout=0,
            candidate_encoding_batch_size=8,
        )
    model = estimator(
        model_config=config,
        preprocessing_config=PreprocessingConfig(numerical_method="standardization", categorical_method="int"),
        trainer_config=TrainerConfig(max_epochs=1, batch_size=8, checkpoint_path=str(directory)),
        random_state=17,
    )
    options: dict[str, Any] = {
        "accelerator": "cpu",
        "logger": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "lr": 0.0,
    }
    if position == 2:
        options["family"] = kind
        embeddings = np.empty((32, 0), dtype=np.float32)
    embedding_options = (
        {"embeddings": embeddings[:24], "embeddings_val": embeddings[24:]} if embeddings.shape[1] else {}
    )
    model.fit(
        features[:24],
        targets[:24],
        X_val=features[24:],
        y_val=targets[24:],
        **embedding_options,
        **options,
    )
    if continuation:
        task = model._task_model
        assert isinstance(task, TaskModel) and task.train_features is not None and task.train_targets is not None
        previous_features = task.train_features[0][0].clone()
        previous_targets = task.train_targets.clone()
        features.loc[:, ["first", "second", "third"]] += 3
        targets = (
            np.roll(targets, 1, axis=0)
            if kind in {"binary", "multiclass", "categorical", "dirichlet"}
            else targets + 0.25
        )
        embeddings += 2
        embedding_options = (
            {"embeddings": embeddings[:24], "embeddings_val": embeddings[24:]} if embeddings.shape[1] else {}
        )
        model.fit(
            features[:24],
            targets[:24],
            X_val=features[24:],
            y_val=targets[24:],
            rebuild=False,
            **embedding_options,
            **options,
        )
        assert task.train_features is not None and task.train_targets is not None
        assert not torch.equal(task.train_features[0][0], previous_features)
        assert not torch.equal(task.train_targets, previous_targets)
    return model, features[24:], embeddings[24:]


def _predict(model, features, embeddings, classifier):
    method = model.predict_proba if classifier else model.predict
    return method(features, embeddings=embeddings) if embeddings.shape[1] else method(features)


@pytest.fixture
def exact_tabr(monkeypatch):
    for name in ("delu", "faiss", "faiss_torch_utils"):
        monkeypatch.setattr(TabR, name, getattr(TabR, name))
    _exact_index()


@pytest.mark.parametrize("architecture", ["modernnca", "tabr_exact", "tabr_native"])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("continuation", [False, True])
def test_retrieval_predictions_in_fresh_process(architecture, kind, continuation, tmp_path, request):
    if architecture == "tabr_native":
        request.getfixturevalue("native_tabr")
    if architecture == "tabr_exact":
        request.getfixturevalue("exact_tabr")
    model, features, embeddings = _fit_case(kind, architecture, tmp_path, continuation)
    assert isinstance(features, pd.DataFrame)
    classifier = kind in {"binary", "multiclass"}
    expected = {
        "full": _predict(model, features, embeddings, classifier),
        "reordered": _predict(model, features.iloc[::-1], embeddings[::-1].copy(), classifier),
        "single": _predict(model, features.iloc[:1], embeddings[:1], classifier),
    }
    path = tmp_path / "retrieval.deeptab"
    model.save(str(path))
    bundle = torch.load(path, weights_only=False, map_location="cpu")
    assert bundle["artifact_metadata"]["format_version"] == 3
    state = bundle["retrieval_state"]
    assert state["version"] == 1 and state["row_count"] == 24
    assert isinstance(model._task_model, TaskModel) and model._task_model.train_targets is not None
    torch.testing.assert_close(state["targets"], model._task_model.train_targets.cpu(), rtol=0, atol=0)
    assert model._task_model.train_features is not None
    for group, original in zip(state["features"], model._task_model.train_features, strict=True):
        if original is None:
            assert group is None
            continue
        assert group is not None and original is not None
        for saved, live in zip(group, original, strict=True):
            assert saved.device.type == "cpu" and not saved.requires_grad
            assert saved.data_ptr() != live.data_ptr()
            torch.testing.assert_close(saved, live.cpu(), rtol=0, atol=0)
    query = {
        "rows": features.to_dict(orient="records"),
        "columns": list(features.columns),
        "dtypes": {name: str(dtype) for name, dtype in features.dtypes.items()},
        "module": type(model).__module__,
        "class": type(model).__name__,
        "classifier": classifier,
        "exact_index": architecture == "tabr_exact",
    }
    (tmp_path / "query.json").write_text(json.dumps(query))
    np.save(tmp_path / "embeddings.npy", embeddings)
    code = """
import importlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
directory = Path(sys.argv[1])
query = json.loads((directory / 'query.json').read_text())
if query['exact_index']:
    import runpy
    runpy.run_path(sys.argv[2])['_exact_index']()
features = pd.DataFrame.from_records(query['rows'], columns=query['columns']).astype(query['dtypes'])
embeddings = np.load(directory / 'embeddings.npy', allow_pickle=False)
model = getattr(importlib.import_module(query['module']), query['class']).load(str(directory / 'retrieval.deeptab'), device='cpu')
method = model.predict_proba if query['classifier'] else model.predict
def predict(rows, vectors):
    return method(rows, embeddings=vectors) if vectors.shape[1] else method(rows)
np.savez(directory / 'restored.npz', full=predict(features, embeddings), reordered=predict(features.iloc[::-1], embeddings[::-1].copy()), single=predict(features.iloc[:1], embeddings[:1]), targets=model._task_model.train_targets.cpu().numpy())
"""
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": str(ROOT)}
    child = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code, str(tmp_path), str(Path(__file__).resolve())],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    with np.load(tmp_path / "restored.npz", allow_pickle=False) as restored:
        for name, prediction in expected.items():
            np.testing.assert_allclose(restored[name], prediction, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(restored["targets"], state["targets"].numpy())
    np.testing.assert_allclose(expected["reordered"], expected["full"][::-1], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(expected["single"], expected["full"][:1], rtol=1e-6, atol=1e-7)


@pytest.fixture(scope="module")
def saved_candidate(tmp_path_factory):
    directory = tmp_path_factory.mktemp("candidate_contract")
    model, _, _ = _fit_case("multiclass", "modernnca", directory)
    path = directory / "model.deeptab"
    model.save(str(path))
    return model, torch.load(path, weights_only=False, map_location="cpu")


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "version",
        "bool_version",
        "row_count",
        "target_rows",
        "target_width",
        "target_nan",
        "label_range",
        "label_fraction",
        "target_dtype",
        "feature_rows",
        "feature_width",
        "feature_nan",
        "feature_dtype",
        "feature_dtype_and_declaration",
        "target_dtype_and_declaration",
        "target_meta",
        "feature_count",
        "missing_group",
        "group_order",
        "schema",
        "dtypes",
        "class_mapping",
        "task_metadata",
    ],
)
def test_malformed_candidate_bank_is_rejected(saved_candidate, corruption):
    model, original = saved_candidate
    bundle = copy.deepcopy(original)
    state = bundle["retrieval_state"]
    if corruption == "missing":
        del bundle["retrieval_state"]
    elif corruption in {"version", "bool_version"}:
        state["version"] = 99 if corruption == "version" else True
    elif corruption == "row_count":
        state["row_count"] = 0
    elif corruption == "target_rows":
        state["targets"] = state["targets"][:-1]
    elif corruption == "target_width":
        state["targets"] = state["targets"][:, None].repeat(1, 2)
    elif corruption in {"target_nan", "label_fraction"}:
        state["targets"] = state["targets"].float()
        state["target_dtype"] = str(state["targets"].dtype)
        state["targets"][0] = float("nan") if corruption == "target_nan" else 0.5
    elif corruption == "label_range":
        state["targets"][0] = 100
    elif corruption == "target_dtype":
        state["target_dtype"] = "torch.float64"
    elif corruption == "target_dtype_and_declaration":
        state["targets"] = state["targets"].float()
        state["target_dtype"] = str(state["targets"].dtype)
    elif corruption == "target_meta":
        state["targets"] = state["targets"].to("meta")
    elif corruption in {"feature_rows", "feature_width"}:
        state["features"][0][0] = (
            state["features"][0][0][:-1] if corruption == "feature_rows" else state["features"][0][0].repeat(1, 2)
        )
    elif corruption == "feature_nan":
        state["features"][0][0][0] = float("nan")
    elif corruption == "feature_dtype":
        state["features"][0][0] = state["features"][0][0].double()
    elif corruption == "feature_dtype_and_declaration":
        state["features"][0][0] = state["features"][0][0].double()
        state["feature_dtypes"][0][0] = str(state["features"][0][0].dtype)
    elif corruption == "feature_count":
        state["features"][0].pop()
    elif corruption == "missing_group":
        state["features"] = (None, *state["features"][1:])
    elif corruption == "group_order":
        state["features"] = (state["features"][1], state["features"][0], state["features"][2])
    elif corruption == "schema":
        state["feature_schema"] = {}
    elif corruption == "class_mapping":
        bundle["classes_"] = bundle["classes_"][::-1].copy()
    elif corruption == "task_metadata":
        bundle["task_info"]["classes_"] = list(reversed(bundle["task_info"]["classes_"]))
    else:
        state["feature_dtypes"] = []
    with pytest.raises(ValueError, match="Retrieval candidate bank"):
        type(model)._load_from_bundle(bundle)


@pytest.mark.parametrize("method", ["predict", "validation", "test"])
@pytest.mark.parametrize("missing", ["features", "targets"])
def test_inference_never_falls_back_without_bank(saved_candidate, method, missing, monkeypatch):
    model, bundle = saved_candidate
    restored = type(model)._load_from_bundle(copy.deepcopy(bundle))
    task = restored._task_model
    assert isinstance(task, TaskModel) and task.train_features is not None and task.train_targets is not None
    features = tuple([feature[:1] for feature in group] if group is not None else [] for group in task.train_features)
    labels = task.train_targets[:1]
    monkeypatch.setattr(task, "train_features" if missing == "features" else "train_targets", None)
    with pytest.raises(RuntimeError, match="Retrieval candidate bank is missing"):
        if method == "predict":
            task.predict_step(features, 0)
        elif method == "validation":
            task.validation_step((features, labels), 0)
        else:
            task.test_step((features, labels), 0)
    with pytest.raises(ValueError, match="Retrieval candidate bank is missing"):
        restored.save("unused.deeptab")


@pytest.mark.parametrize("version", [1, 2])
def test_older_retrieval_artifacts_require_refitting(saved_candidate, version):
    model, original = saved_candidate
    bundle = copy.deepcopy(original)
    bundle["artifact_metadata"]["format_version"] = version
    del bundle["retrieval_state"]
    with pytest.raises(ValueError, match="Refit with the original training data"):
        type(model)._load_from_bundle(bundle)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_candidate_tensors_follow_task_device(saved_candidate, device):
    model, bundle = saved_candidate
    restored = type(model)._load_from_bundle(copy.deepcopy(bundle))
    task = restored._task_model
    assert isinstance(task, TaskModel)
    task.to(device)
    assert task.train_features is not None and task.train_targets is not None
    assert task.train_targets.device.type == device
    assert all(feature.device.type == device for group in task.train_features if group is not None for feature in group)


@pytest.mark.parametrize("version", [1, 2])
def test_plain_model_old_artifact_without_bank_still_loads(tmp_path, version):
    from deeptab.configs import MLPConfig
    from deeptab.models import MLPRegressor

    features = np.random.default_rng(31).normal(size=(24, 3))
    targets = features[:, 0]
    model = MLPRegressor(
        model_config=MLPConfig(layer_sizes=[8]),
        trainer_config=TrainerConfig(max_epochs=1, batch_size=8, checkpoint_path=str(tmp_path)),
    )
    model.fit(features, targets, accelerator="cpu", logger=False, enable_progress_bar=False)
    bundle = build_save_bundle(model, lss=False, family=None)
    assert bundle["retrieval_state"] is None
    del bundle["retrieval_state"]
    bundle["artifact_metadata"]["format_version"] = version
    loaded = MLPRegressor._load_from_bundle(bundle)
    np.testing.assert_array_equal(model.predict(features), loaded.predict(features))


def test_tabr_rebuilds_index_instead_of_serializing_it(exact_tabr, tmp_path, monkeypatch):
    model, features, embeddings = _fit_case("regression", "tabr_exact", tmp_path)
    expected = model.predict(features, embeddings=embeddings)
    task = model._task_model
    assert isinstance(task, TaskModel) and isinstance(task.estimator, TabR)
    assert task.estimator.search_index is not None

    def reject_serialization(instance):
        raise AssertionError("The runtime index must not be serialized.")

    monkeypatch.setattr(type(task.estimator.search_index), "__reduce__", reject_serialization)
    path = str(tmp_path / "tabr.deeptab")
    model.save(path)
    loaded = TabRRegressor.load(path)
    restored = loaded._task_model
    assert isinstance(restored, TaskModel) and isinstance(restored.estimator, TabR)
    assert restored.estimator.search_index is None
    np.testing.assert_allclose(loaded.predict(features, embeddings=embeddings), expected, rtol=1e-6, atol=1e-7)
    assert restored.estimator.search_index is not None

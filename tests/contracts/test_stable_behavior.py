"""Stable estimator output and fresh-process persistence contracts."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from deeptab.core.registry import MODEL_REGISTRY

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
STABLE_ARCHITECTURES = sorted(name for name, info in MODEL_REGISTRY.items() if info.status == "stable")


@pytest.mark.parametrize("architecture", STABLE_ARCHITECTURES)
@pytest.mark.parametrize("task", ["classification", "regression", "lss"])
def test_stable_output_and_fresh_process_reload(architecture, task, stable_model_factory, easy_problem, tmp_path):
    train, query, train_classes, _, train_values, _ = easy_problem
    model = stable_model_factory(architecture, task, epochs=2)
    targets = train_classes if task == "classification" else train_values
    options = {"accelerator": "cpu", "logger": False, "enable_progress_bar": False, "enable_model_summary": False}
    assert model.fit(train, targets, **options, **({"family": "normal"} if task == "lss" else {})) is model
    method = model.predict_proba if task == "classification" else model.predict
    full, single = method(query), method(query.iloc[:1])
    assert (
        full.shape == (len(query),)
        if task == "regression"
        else full.shape == (len(query), 3 if task == "classification" else 2)
    )
    assert single.shape == full[:1].shape
    assert np.isfinite(full).all() and np.isfinite(single).all()
    np.testing.assert_array_equal(model.feature_names_in_, train.columns)
    assert model.n_features_in_ == train.shape[1]
    if task == "classification":
        assert ((full >= 0) & (full <= 1)).all()
        np.testing.assert_allclose(full.sum(axis=1), 1, atol=1e-6)
        np.testing.assert_array_equal(model.classes_, [10, 30, 50])
        np.testing.assert_array_equal(model.predict(query), model.classes_[full.argmax(axis=1)])
    elif task == "lss":
        assert (full[:, 1] > 0).all()
        assert model.predict(query, raw=True).shape == (len(query), 2)
    order = np.concatenate([np.arange(start, min(start + 16, len(query)))[::-1] for start in range(0, len(query), 16)])
    np.testing.assert_allclose(method(query.iloc[order]), full[order], rtol=1e-5, atol=1e-6)
    if architecture != "SAINT":
        np.testing.assert_allclose(method(query.iloc[::-1]), full[::-1], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(single, full[:1], rtol=1e-5, atol=1e-6)
    model.save(str(tmp_path / "stable.deeptab"))
    query.to_json(tmp_path / "query.json", orient="table", index=False)
    metadata = {"module": type(model).__module__, "class": type(model).__name__, "task": task}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    code = """
import importlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
directory = Path(sys.argv[1])
metadata = json.loads((directory / 'metadata.json').read_text())
query = pd.read_json(directory / 'query.json', orient='table')
model = getattr(importlib.import_module(metadata['module']), metadata['class']).load(str(directory / 'stable.deeptab'), device='cpu')
method = model.predict_proba if metadata['task'] == 'classification' else model.predict
np.savez(directory / 'child.npz', full=method(query), single=method(query.iloc[:1]))
"""
    child = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=tmp_path,
        env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    with np.load(tmp_path / "child.npz", allow_pickle=False) as actual:
        np.testing.assert_allclose(actual["full"], full, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(actual["single"], single, rtol=1e-5, atol=1e-6)

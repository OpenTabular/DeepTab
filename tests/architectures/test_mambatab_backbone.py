"""Regression test: MambaTab must actually run its Mamba block.

forward() went initial_layer -> norm -> activation -> head and never called
self.mamba, so the model trained as a linear layer with an MLP head while all
Mamba parameters sat in the optimizer with no gradients.
"""

from typing import Any, cast

import numpy as np
import pandas as pd

from deeptab.models import MambaTabRegressor


def _backward_one_batch(model) -> Any:
    task = cast(Any, model._task_model)
    task.zero_grad()
    (num, cat, emb), labels = next(iter(cast(Any, model._data_module).train_dataloader()))
    preds = task(num, cat, emb)
    task.compute_loss(preds, labels).backward()
    return task


def test_mambatab_trains_its_mamba_block():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"num1": rng.randn(60), "num2": rng.rand(60)})
    y = rng.randn(60)

    model = MambaTabRegressor()
    model.fit(X, y, max_epochs=1, batch_size=16, accelerator="cpu")

    task = _backward_one_batch(model)
    mamba_params = [(name, p) for name, p in task.named_parameters() if ".mamba." in name]
    assert mamba_params, "expected mamba parameters on the estimator"
    assert all(p.grad is not None for _, p in mamba_params)

"""Regression test: SAINT's attention pre-norms and dropouts must be applied.

RowColTransformer builds nn.Sequential(LayerNorm, MultiheadAttention, Dropout)
but forward indexed only attn[1], so the LayerNorms and dropouts were dead
parameters and SAINT trained without normalization before either attention.
"""

import numpy as np
import pandas as pd

from deeptab.models import SAINTRegressor


def _backward_one_batch(model):
    task = model._task_model
    task.zero_grad()
    (num, cat, emb), labels = next(iter(model._data_module.train_dataloader()))
    preds = task(num, cat, emb)
    task.compute_loss(preds, labels).backward()
    return task


def test_saint_attention_layernorms_receive_gradients():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"num1": rng.randn(60), "num2": rng.rand(60)})
    y = rng.randn(60)

    model = SAINTRegressor()
    model.fit(X, y, max_epochs=1, batch_size=16, accelerator="cpu")

    task = _backward_one_batch(model)
    # LayerNorms sit at index 0 of the attention Sequentials (attn1, attn2).
    norm_params = [
        (name, p)
        for name, p in task.named_parameters()
        if "layers" in name and (".0.0." in name or ".2.0." in name) and "weight" in name
    ]
    assert norm_params, "expected attention LayerNorm parameters"
    assert all(p.grad is not None for _, p in norm_params)

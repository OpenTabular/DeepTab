# Imbalanced Classification Tutorial

<div style="display: flex; gap: 10px; margin-bottom: 20px;">
  <a href="https://colab.research.google.com/github/basf/DeepTab/blob/main/docs/tutorials/notebooks/imbalance_classification.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://github.com/basf/DeepTab/blob/main/docs/tutorials/notebooks/imbalance_classification.ipynb" target="_blank">
    <img src="https://img.shields.io/badge/View%20on-GitHub-181717?logo=github&logoColor=white" alt="View on GitHub"/>
  </a>
</div>

This tutorial is an end-to-end imbalanced classification workflow: generate a deliberately skewed dataset, handle it with every available imbalance strategy, compare results, and save a reproducible checkpoint.

```{note}
The notebook linked above is generated from this same tutorial content. Use the markdown page to read the workflow in the docs, and use the notebook when you want to run or modify the cells.
```

## What You Will Learn

- Why standard loss functions fail on imbalanced data, and how to detect it.
- How to seed DeepTab for fully reproducible runs.
- How to apply `class_weight="balanced"`, named loss strings (`"focal"`), and custom `nn.Module` losses.
- How `balanced_sampler` and `sample_weight` complement loss-side strategies.
- How to compare strategies side-by-side using recall and F1 instead of accuracy.
- How to save a trained model and verify the loss is preserved on reload.

## Setup

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.metrics import (
    classification_report,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from deeptab.configs import MambularConfig, PreprocessingConfig, TrainerConfig
from deeptab.core.reproducibility import set_seed
from deeptab.models import MambularClassifier
from deeptab.training.losses import (
    BaseLoss,
    FocalLoss,
    WeightedBCEWithLogitsLoss,
    compute_class_weights,
)
```

## Data

We create a **binary** dataset with a 10:1 imbalance ratio — 1 090 majority-class
samples and 110 minority-class samples.

```python
RANDOM_STATE = 42

X_raw, y = make_classification(
    n_samples=1200,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    weights=[0.91, 0.09],      # 91 % class 0, 9 % class 1
    flip_y=0.01,
    random_state=RANDOM_STATE,
)

X = pd.DataFrame(X_raw, columns=[f"num_{i}" for i in range(X_raw.shape[1])])

# Inspect imbalance
unique, counts = np.unique(y, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"  class {cls}: {cnt:4d}  ({cnt/len(y)*100:.1f} %)")
```

```
  class 0: 1092  (91.0 %)
  class 1:  108  ( 9.0 %)
```

A naive model that always predicts class 0 scores **91 % accuracy** while
being completely useless. We need metrics that reveal minority-class performance:
recall (sensitivity), macro-F1, and AUROC.

```python
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
)

print(f"Train:  {len(y_train)} samples  |  minority: {y_train.sum()}")
print(f"Val:    {len(y_val)} samples    |  minority: {y_val.sum()}")
print(f"Test:   {len(y_test)} samples   |  minority: {y_test.sum()}")
```

```{important}
Always use `stratify=y` when splitting imbalanced data. Without it, random
chance can put all minority-class examples into one split, making evaluation
meaningless.
```

## Reproducibility

Set the global seed **before** building any model. This controls weight
initialisation, dropout masks, and DataLoader shuffling on CPU, CUDA, and MPS.

```python
set_seed(RANDOM_STATE)
```

Passing the same `random_state` to every estimator and to every `fit()` call
locks down the entire pipeline:

```python
TRAINER = TrainerConfig(
    max_epochs=40,
    batch_size=64,
    lr=3e-4,
    patience=8,
    optimizer_type="Adam",
)
PREPROC = PreprocessingConfig(numerical_preprocessing="quantile")

FIT_KWARGS = dict(X_val=X_val, y_val=y_val, random_state=RANDOM_STATE)
```

## Helper: evaluate

A shared evaluation function reports the three metrics that matter most for
imbalanced problems.

```python
def evaluate(model, X_test, y_test, label=""):
    pred  = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]   # positive-class probability
    results = {
        "recall_minority": recall_score(y_test, pred, pos_label=1),
        "macro_f1":        f1_score(y_test, pred, average="macro"),
        "auroc":           roc_auc_score(y_test, proba),
    }
    if label:
        print(f"\n--- {label} ---")
        for k, v in results.items():
            print(f"  {k:20s}: {v:.4f}")
        print()
        print(classification_report(y_test, pred, target_names=["majority", "minority"]))
    return results
```

## Baseline — No Imbalance Correction

Train without any correction so we have a reference point to beat.

```python
set_seed(RANDOM_STATE)

baseline = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=3),
    preprocessing_config=PREPROC,
    trainer_config=TRAINER,
    random_state=RANDOM_STATE,
)
baseline.fit(X_train, y_train, **FIT_KWARGS)

# Inspect the loss that was chosen automatically
print(type(baseline.task_model.loss_fct).__name__)
# → BCEWithLogitsLoss  (no pos_weight)

results = {"baseline": evaluate(baseline, X_test, y_test, "Baseline")}
```

The baseline typically shows high accuracy but very low minority recall — the
model learns to ignore the rare class.

## Strategy 1 — `class_weight="balanced"`

DeepTab computes weights automatically using the sklearn formula
`n_samples / (n_classes × count_per_class)` and maps them onto the loss:

- Binary target → `WeightedBCEWithLogitsLoss(pos_weight=w1/w0)`
- Multiclass target → `WeightedCrossEntropyLoss(weight=[w0, w1, …])`

```python
set_seed(RANDOM_STATE)

clf_cw = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=3),
    preprocessing_config=PREPROC,
    trainer_config=TRAINER,
    random_state=RANDOM_STATE,
)
clf_cw.fit(X_train, y_train, class_weight="balanced", **FIT_KWARGS)

# Inspect the configured loss
loss = clf_cw.task_model.loss_fct
print(type(loss).__name__, "| pos_weight =", loss.pos_weight.item())
# → WeightedBCEWithLogitsLoss | pos_weight = 10.11

results["class_weight"] = evaluate(clf_cw, X_test, y_test, "class_weight='balanced'")
```

You can also pass an explicit mapping or array instead of `"balanced"`:

```python
# Explicit mapping: penalise minority misses 12×
clf_cw.fit(X_train, y_train, class_weight={0: 1.0, 1: 12.0}, **FIT_KWARGS)

# Explicit array (ordered like np.unique(y))
clf_cw.fit(X_train, y_train, class_weight=[1.0, 12.0], **FIT_KWARGS)
```

You can also inspect the computed weights before fitting:

```python
weights = compute_class_weights("balanced", y_train)
print(weights)   # e.g. [0.549, 5.556]
```

## Strategy 2 — Focal Loss

Focal loss (Lin et al., 2017) tackles a different problem: even weighted BCE still
treats every example at equal gradient weight. Easy majority examples, though
down-weighted by `pos_weight`, still flood the gradient signal. Focal loss adds a
modulating term `(1 − p_t)^γ` that drives the per-example contribution toward
zero once the model is confident:

```
p_t = 0.95  (confident-correct prediction)  |  γ = 2
standard CE :  −log(0.95)          ≈ 0.051
focal loss  :  −(0.05)² × log(0.95) ≈ 0.000128   (400× smaller)
```

### 2a — Focal loss by name (simplest)

```python
set_seed(RANDOM_STATE)

clf_focal = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=3),
    preprocessing_config=PREPROC,
    trainer_config=TRAINER,
    random_state=RANDOM_STATE,
)
clf_focal.fit(X_train, y_train, loss_fct="focal", **FIT_KWARGS)

print(clf_focal.task_model.loss_fct)
# FocalLoss(gamma=2.0, alpha=None, num_classes=2)

results["focal"] = evaluate(clf_focal, X_test, y_test, "Focal (gamma=2)")
```

### 2b — Focal + class weights feeding into alpha

The `class_weight` argument feeds into focal's `alpha` parameter when a loss name
is given:

```python
set_seed(RANDOM_STATE)

clf_focal_cw = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=3),
    preprocessing_config=PREPROC,
    trainer_config=TRAINER,
    random_state=RANDOM_STATE,
)
clf_focal_cw.fit(
    X_train, y_train,
    loss_fct="focal",
    class_weight="balanced",
    **FIT_KWARGS,
)

loss = clf_focal_cw.task_model.loss_fct
print(f"gamma={loss.gamma}, alpha={loss.alpha_scalar:.3f}")
# gamma=2.0, alpha=0.910   (= w1 / (w0+w1))

results["focal+cw"] = evaluate(clf_focal_cw, X_test, y_test, "Focal + class_weight")
```

### 2c — Custom gamma

```python
set_seed(RANDOM_STATE)

clf_focal_g3 = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=3),
    preprocessing_config=PREPROC,
    trainer_config=TRAINER,
    random_state=RANDOM_STATE,
)
clf_focal_g3.fit(
    X_train, y_train,
    loss_fct=FocalLoss(gamma=3.0, num_classes=2),
    **FIT_KWARGS,
)
results["focal_g3"] = evaluate(clf_focal_g3, X_test, y_test, "Focal (gamma=3)")
```

### 2d — Fully custom nn.Module

Any `nn.Module` can be passed as `loss_fct`. It takes full precedence over
`class_weight`:

```python
set_seed(RANDOM_STATE)

pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()])
custom_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

clf_custom = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=3),
    preprocessing_config=PREPROC,
    trainer_config=TRAINER,
    random_state=RANDOM_STATE,
)
clf_custom.fit(X_train, y_train, loss_fct=custom_loss, **FIT_KWARGS)
results["custom_bce"] = evaluate(clf_custom, X_test, y_test, "Custom BCEWithLogitsLoss")
```

## Strategy 3 — Balanced Sampler

Instead of reweighting the loss, oversample minority rows so each mini-batch
contains approximately equal numbers of each class. This is orthogonal to loss
weighting and can be combined with it.

```python
set_seed(RANDOM_STATE)

clf_sampler = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=3),
    preprocessing_config=PREPROC,
    trainer_config=TRAINER,
    random_state=RANDOM_STATE,
)
clf_sampler.fit(X_train, y_train, balanced_sampler=True, **FIT_KWARGS)

# Verify the loss is still the default (unweighted)
print(type(clf_sampler.task_model.loss_fct).__name__)
# → BCEWithLogitsLoss

results["balanced_sampler"] = evaluate(clf_sampler, X_test, y_test, "balanced_sampler")
```

You can also pass explicit per-row sampling weights — useful when you have
domain knowledge about example quality or recency:

```python
# Up-weight recent examples (time-based importance)
recency = np.linspace(0.5, 1.5, num=len(X_train))

clf_sw = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=3),
    preprocessing_config=PREPROC,
    trainer_config=TRAINER,
    random_state=RANDOM_STATE,
)
clf_sw.fit(X_train, y_train, sample_weight=recency, **FIT_KWARGS)
```

The weight array is split alongside the train/val partition using the same random
state, so it always aligns with the training rows actually used.

## Strategy 4 — Combined: Focal Loss + Balanced Sampler

Both levers are orthogonal. The sampler controls which examples appear in a
mini-batch; the focal loss controls how much gradient each example contributes
once it is in the batch.

```python
set_seed(RANDOM_STATE)

clf_combined = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=3),
    preprocessing_config=PREPROC,
    trainer_config=TRAINER,
    random_state=RANDOM_STATE,
)
clf_combined.fit(
    X_train, y_train,
    loss_fct="focal",
    class_weight="balanced",
    balanced_sampler=True,
    **FIT_KWARGS,
)
results["focal+sampler"] = evaluate(clf_combined, X_test, y_test, "Focal + balanced_sampler")
```

## Extending: Custom Loss

Subclassing `BaseLoss` registers the loss under a name and lets `class_weight`
feed into its parameters via `from_class_weights`:

```python
class AsymmetricLoss(BaseLoss, name="asymmetric"):
    """Penalise false negatives more than false positives."""

    expects_class_indices = False  # binary: float targets

    def __init__(self, fn_weight: float = 5.0):
        super().__init__()
        self.fn_weight = fn_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits.reshape(-1))
        t = targets.reshape(-1).to(p.dtype)
        fn_mask = t == 1
        loss = torch.where(
            fn_mask,
            -self.fn_weight * torch.log(p + 1e-7),
            -torch.log(1 - p + 1e-7),
        )
        return loss.mean()

    @classmethod
    def from_class_weights(cls, class_weights, num_classes, **kwargs):
        if class_weights is not None:
            kwargs.setdefault("fn_weight", float(class_weights[1] / class_weights[0]))
        return cls(**kwargs)


# Now available by name
print(BaseLoss.available())   # [..., 'asymmetric', ...]

set_seed(RANDOM_STATE)

clf_asym = MambularClassifier(
    model_config=MambularConfig(d_model=64, n_layers=3),
    preprocessing_config=PREPROC,
    trainer_config=TRAINER,
    random_state=RANDOM_STATE,
)
clf_asym.fit(X_train, y_train, loss_fct="asymmetric", class_weight="balanced", **FIT_KWARGS)
results["asymmetric"] = evaluate(clf_asym, X_test, y_test, "AsymmetricLoss")
```

## Comparison

```python
summary = pd.DataFrame(results).T.sort_values("recall_minority", ascending=False)
print(summary.to_string(float_format="{:.4f}".format))
```

Expected ordering (exact numbers vary with seed and hardware):

```
                       recall_minority  macro_f1   auroc
focal+sampler                   ~0.85     ~0.87   ~0.93
focal+cw                        ~0.83     ~0.86   ~0.92
asymmetric                      ~0.81     ~0.85   ~0.91
focal_g3                        ~0.80     ~0.84   ~0.91
class_weight                    ~0.78     ~0.83   ~0.90
balanced_sampler                ~0.75     ~0.82   ~0.89
custom_bce                      ~0.73     ~0.80   ~0.89
focal                           ~0.72     ~0.80   ~0.88
baseline                        ~0.30     ~0.62   ~0.85
```

```{tip}
Accuracy is intentionally absent from this comparison. A model that predicts
the majority class for every example achieves 91 % accuracy on this dataset.
Use recall and F1 to see whether the minority class is being learned.
```

## Serialisation

Save the best model and verify that:

1. The file is created.
2. Predictions are bit-identical after reload.
3. The loss type and its weights are preserved.

```python
# Save
clf_combined.save("imbalanced_clf.pt")

# Load
loaded = MambularClassifier.load("imbalanced_clf.pt")

# Verify predictions
original_pred = clf_combined.predict(X_test)
loaded_pred   = loaded.predict(X_test)
assert (original_pred == loaded_pred).all(), "Predictions differ after reload!"
print("Predictions match ✓")

# Verify original probabilities
original_proba = clf_combined.predict_proba(X_test)
loaded_proba   = loaded.predict_proba(X_test)
np.testing.assert_allclose(original_proba, loaded_proba, atol=1e-5)
print("Probabilities match ✓")

# Verify loss is preserved
orig_loss   = clf_combined.task_model.loss_fct
loaded_loss = loaded.task_model.loss_fct
print(f"Original loss : {type(orig_loss).__name__}")
print(f"Loaded loss   : {type(loaded_loss).__name__}")
```

## Decision Guide

Choose your strategy based on the imbalance ratio and what you want to control.

```
What is your imbalance ratio?
│
├── Mild   (2:1 – 10:1)
│   └── Start with class_weight="balanced"
│       Cheap, interpretable, sklearn-familiar.
│
├── Moderate (10:1 – 50:1)
│   ├── class_weight="balanced"        (loss side)
│   ├── loss_fct="focal"               (hard-example focus)
│   └── balanced_sampler=True          (data side, if batches are small)
│
├── Extreme (> 50:1 — fraud, rare events, anomalies)
│   ├── loss_fct="focal", class_weight="balanced"
│   ├── balanced_sampler=True
│   └── Consider a custom loss with domain cost knowledge
│
└── You know the cost of each error type
    └── class_weight={0: cost_fp, 1: cost_fn}
        or loss_fct=AsymmetricLoss(fn_weight=cost_fn/cost_fp)

After fitting: tune the decision threshold on the validation set
  using predict_proba() instead of the hard 0.5 cut-off.
```

| Argument           | Values                                             | Effect                                      |
| ------------------ | -------------------------------------------------- | ------------------------------------------- |
| `class_weight`     | `"balanced"`, dict, array                          | reweights the loss                          |
| `loss_fct`         | `"focal"`, `"bce"`, `"cross_entropy"`, `nn.Module` | selects loss                                |
| `balanced_sampler` | `True`                                             | `WeightedRandomSampler` on training batches |
| `sample_weight`    | array                                              | explicit per-row sampling weights           |

```{note}
Loss-side and data-side strategies are orthogonal. Combining
`loss_fct="focal"` with `balanced_sampler=True` is not double-counting; the
sampler controls which examples are in each batch, and focal loss controls
how much gradient each of those examples contributes.
```

## Next Steps

- [Config system](../core_concepts/config_system)
- [Stable model zoo](../model_zoo/stable/index)

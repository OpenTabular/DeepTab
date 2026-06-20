# Hyperparameter Optimization

<div style="display: flex; gap: 10px; margin-bottom: 20px;">
  <a href="https://colab.research.google.com/github/OpenTabular/DeepTab/blob/main/docs/tutorials/notebooks/hpo.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://github.com/OpenTabular/DeepTab/blob/main/docs/tutorials/notebooks/hpo.ipynb" target="_blank">
    <img src="https://img.shields.io/badge/View%20on-GitHub-181717?logo=github&logoColor=white" alt="View on GitHub"/>
  </a>
</div>

Default hyperparameters are a reasonable starting point, never the finish line.
Width, depth, dropout, and the activation function interact in ways that depend
on your data, and the only reliable way to find a good combination is to search.
DeepTab ships a single method, `optimize_hparams()`, that runs Gaussian-process
Bayesian optimization over a search space derived automatically from each model's
configuration, prunes unpromising trials early, and writes the winning settings
straight back into the estimator's config so the next `fit()` uses them.

This tutorial explains exactly what happens inside that method, then walks through
a complete, runnable example for each of the three task types DeepTab supports:
regression, distributional regression (the `*LSS` family), and classification.
The same method drives all three; only the data and one keyword change.

```{note}
The notebook linked above is generated from this same tutorial content. The
markdown page is the readable lesson; the notebook is the executable copy.
```

## What You Will Learn

- How `optimize_hparams()` turns a model config into a search space and what the objective actually measures.
- Why the search direction is the same for every task, and how epoch-level pruning saves time.
- How to tune a regressor, a distributional regressor, and a classifier with the same API.
- How to inspect the search space with `get_search_space()` before spending compute.
- How to fix parameters with `fixed_params` and override ranges with `custom_search_space`.

## How `optimize_hparams()` Works

The method is intentionally small on the surface and does a lot underneath. Here
is the full lifecycle of a single call, in order.

1. **Build the search space.** `get_search_space(config, fixed_params, custom_search_space)` walks the fields of the model's config dataclass. Every field that has a known range (for example `d_model`, `dropout`, `activation`) becomes a search dimension; every field listed in `fixed_params` is set on the config and excluded from the search.
2. **Establish a baseline.** The model is trained once with the current config to record a baseline validation loss and the validation loss reached at the pruning epoch. These two numbers seed the pruning thresholds.
3. **Run Bayesian optimization.** [`skopt.gp_minimize`](https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html) fits a Gaussian-process surrogate to the trials seen so far and proposes the next configuration where it expects the largest improvement. This is far more sample-efficient than grid or random search because each new trial is informed by all previous ones.
4. **Evaluate each trial.** For every proposed configuration the method writes the values onto the config, rebuilds the model with the task-aware builder, trains it (with pruning enabled), and measures the validation loss.
5. **Prune early.** If a trial's loss at `prune_epoch` is worse than 1.5x the best epoch loss seen so far, training for that trial stops early instead of running all `max_epochs`. Hopeless configurations are abandoned quickly.
6. **Write back the winner.** After all trials, the best configuration is written into `model.config`. The returned list is the raw best vector in search-space order; the durable result is the mutated `config`, so the very next `fit()` trains the tuned model.

### The objective: one direction for every task

The quantity being minimized is the Lightning **validation loss**, which is the
training objective itself:

| Task                      | Estimator suffix | Validation loss         |
| ------------------------- | ---------------- | ----------------------- |
| Regression                | `*Regressor`     | Mean squared error      |
| Classification            | `*Classifier`    | Cross-entropy           |
| Distributional regression | `*LSS`           | Negative log-likelihood |

Because the objective is always the training loss, it is always defined and
always lower-is-better. That keeps the optimizer's direction identical across
tasks and removes any mismatch between what the search optimizes and what the
model trains on. You never select the metric direction yourself.

### Key parameters

| Parameter             | Meaning                                                                                                                    |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `X`, `y`              | Training features and target. The search trains on these.                                                                  |
| `X_val`, `y_val`      | Validation split. The objective is measured here. Always provide it.                                                       |
| `time`                | Number of optimization trials. **Must be at least 10** (the surrogate needs initial points before it can model the space). |
| `max_epochs`          | Maximum epochs per trial. Combined with early stopping and pruning, most trials finish sooner.                             |
| `prune_by_epoch`      | When `True`, prune by the loss at `prune_epoch`; when `False`, prune by the best validation loss so far.                   |
| `prune_epoch`         | The epoch at which a trial is judged for pruning.                                                                          |
| `fixed_params`        | A `{field: value}` dict of config fields to hold constant and exclude from the search.                                     |
| `custom_search_space` | A `{field: skopt.space.Dimension}` dict that overrides or adds ranges for specific fields.                                 |

```{important}
`time` is the single biggest cost lever. Each trial trains a full model, so a
search with `time=20` trains up to twenty models. Keep it small while
prototyping, raise it for a final search, and always run the search on the
training and validation splits only. The test set must never be visible to it.
```

---

## Setup

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from skopt.space import Categorical, Real

from deeptab.configs import MLPConfig, PreprocessingConfig, TrainerConfig
from deeptab.core.reproducibility import set_seed
from deeptab.hpo import get_search_space
from deeptab.models import MLPClassifier, MLPLSS, MLPRegressor

RANDOM_STATE = 42
```

```{note}
For a quick demonstration these tutorials train with very low `max_epochs` and `patience` (5 and 2). Treat these as placeholders and choose values that match your own compute budget and problem. As a starting point, at least `max_epochs=100` and `patience=10` are recommended for meaningful results.
```

We use the MLP estimators throughout. They train quickly, which keeps the search
affordable, and they expose a compact, easy-to-read search space. Everything here
works identically for any other DeepTab estimator (FT-Transformer, ResNet, TabM,
NODE, and the rest); the only difference is that richer backbones expose more
fields to tune, so their searches cost more per trial.

A shared preprocessing and trainer configuration keeps the three examples
comparable:

```python
PREPROC = PreprocessingConfig(
    numerical_preprocessing="ple",   # piecewise-linear encoding of numericals
    n_bins=64,
    categorical_preprocessing="int",
)
TRAINER = TrainerConfig(max_epochs=5, batch_size=256, patience=2)
```

## Inspecting the Search Space First

Before spending compute, look at what will actually be searched.
`get_search_space()` returns the parameter names and their skopt ranges for a
given config. This is the exact call `optimize_hparams()` makes internally, so it
is a faithful preview.

```python
names, space = get_search_space(MLPConfig())
for name, dim in zip(names, space):
    print(f"{name:22s} {dim}")
```

```
embedding_activation   Categorical(categories=('ReLU', 'SELU', 'Identity', 'Tanh', 'LeakyReLU'), ...)
d_model                Categorical(categories=(32, 64, 128, 256, 512, 1024), ...)
layer_norm_eps         Real(low=1e-07, high=0.0001, ...)
activation             Categorical(categories=('ReLU', 'SELU', 'Identity', 'Tanh', 'LeakyReLU', 'SiLU'), ...)
dropout                Real(low=0.0, high=0.5, ...)
```

The search space is derived from the **model** config, so only fields that belong
to `MLPConfig` and have a known range appear. The five dimensions above mean
each trial chooses an embedding activation, a hidden width (`d_model`), a layer
norm epsilon, a block activation, and a dropout rate. Training settings such as
the learning rate live on `TrainerConfig`, not the model config, so they are not
part of this search by default. Reading this list first tells you precisely what
the optimizer can and cannot change.

---

## Regression

We start with a straightforward regression problem: twenty numerical features,
ten of them informative, with moderate noise.

```python
X_arr, y = make_regression(
    n_samples=4000, n_features=20, n_informative=10, noise=12.0, random_state=RANDOM_STATE
)
X = pd.DataFrame(X_arr, columns=[f"num_{i}" for i in range(X_arr.shape[1])])

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_STATE)
print(f"Train: {len(y_train)}  |  Val: {len(y_val)}  |  Test: {len(y_test)}")
```

First, a baseline with default hyperparameters. This is the number to beat.

```python
set_seed(RANDOM_STATE)
baseline = MLPRegressor(
    model_config=MLPConfig(),
    preprocessing_config=PREPROC,
    trainer_config=TRAINER,
    random_state=RANDOM_STATE,
)
baseline.fit(X_train, y_train, X_val=X_val, y_val=y_val, random_state=RANDOM_STATE)
base_r2 = r2_score(y_test, baseline.predict(X_test))
print(f"baseline R2: {base_r2:.4f}")
```

Now run the search. Note what is **not** here: there is no `regression=` argument.
The estimator already knows it is a regressor, so the task type is inferred for
you. The objective is the validation mean squared error.

```python
set_seed(RANDOM_STATE)
tuned = MLPRegressor(
    model_config=MLPConfig(),
    preprocessing_config=PREPROC,
    trainer_config=TRAINER,
    random_state=RANDOM_STATE,
)

best = tuned.optimize_hparams(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    time=15,              # 15 trials (must be at least 10)
    max_epochs=5,
    prune_by_epoch=True,  # judge each trial by its loss at prune_epoch
    prune_epoch=2,
)
print("Best vector:", best)
print("Tuned dropout:", tuned.config.dropout, "| d_model:", tuned.config.d_model)
```

`optimize_hparams()` has already written the winning values into `tuned.config`,
so a final clean fit trains on the selected configuration. Compare against the
baseline on the held-out test set:

```python
set_seed(RANDOM_STATE)
tuned.fit(X_train, y_train, X_val=X_val, y_val=y_val, random_state=RANDOM_STATE)
tuned_r2 = r2_score(y_test, tuned.predict(X_test))
print(f"baseline R2: {base_r2:.4f}   tuned R2: {tuned_r2:.4f}")
```

The tuned model is selected purely on validation loss, then scored once on the
untouched test set: the honest way to report the benefit of a search.

---

## Distributional Regression

Distributional regression (the `*LSS` family) predicts the parameters of a full
conditional distribution rather than a single point. The objective the search
minimizes here is the negative log-likelihood, not a point error. The API is the
same as regression with one addition: you choose a distribution `family`, which
is forwarded to the underlying `fit()` so every trial trains and is scored under
that family.

We reuse the regression data, which suits a `"normal"` family (real-valued,
roughly symmetric target).

```python
set_seed(RANDOM_STATE)
lss = MLPLSS(
    model_config=MLPConfig(),
    preprocessing_config=PREPROC,
    trainer_config=TRAINER,
    random_state=RANDOM_STATE,
)

best_lss = lss.optimize_hparams(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    family="normal",      # forwarded to fit(): trials train and score under this family
    time=15,
    max_epochs=5,
    prune_by_epoch=True,
    prune_epoch=2,
)
print("Best vector:", best_lss)
print("Selected family:", lss.family_name)
```

The search optimizes the validation negative log-likelihood, the same loss the
LSS model trains on. After the search, fit once more and evaluate with the
family's proper scoring rules:

```python
set_seed(RANDOM_STATE)
lss.fit(X_train, y_train, family="normal", X_val=X_val, y_val=y_val, random_state=RANDOM_STATE)
scores = lss.evaluate(X_test, y_test)
for name, value in scores.items():
    print(f"{name:20s} {value:.4f}")
```

`evaluate()` returns the default metrics for the chosen family (for the normal
family these are CRPS, RMSE, and MAE), letting you confirm the tuned distribution
is genuinely better calibrated. The search itself optimizes the negative
log-likelihood; these metrics are how you report the result afterwards. For a
deeper treatment of distributional models, see the
[Uncertainty Quantification](uncertainty_quantification) tutorial.

```{note}
The `family` you pass to `optimize_hparams()` must match the one you pass to the
final `fit()`. The search tunes architecture and regularization for that family;
switching families afterwards would discard the assumption the search optimized
under.
```

---

## Classification

Classification works exactly like regression. The estimator infers the task, and
the objective becomes the validation cross-entropy. We build a binary problem
with a few redundant and noise features to give the search something to do.

```python
Xc_arr, yc = make_classification(
    n_samples=4000, n_features=20, n_informative=10, n_redundant=4,
    n_classes=2, class_sep=0.8, random_state=RANDOM_STATE,
)
Xc = pd.DataFrame(Xc_arr, columns=[f"num_{i}" for i in range(Xc_arr.shape[1])])

Xc_train, Xc_tmp, yc_train, yc_tmp = train_test_split(Xc, yc, test_size=0.3, random_state=RANDOM_STATE)
Xc_val, Xc_test, yc_val, yc_test = train_test_split(Xc_tmp, yc_tmp, test_size=0.5, random_state=RANDOM_STATE)
```

Baseline first:

```python
set_seed(RANDOM_STATE)
clf_base = MLPClassifier(
    model_config=MLPConfig(),
    preprocessing_config=PREPROC,
    trainer_config=TRAINER,
    random_state=RANDOM_STATE,
)
clf_base.fit(Xc_train, yc_train, X_val=Xc_val, y_val=yc_val, random_state=RANDOM_STATE)
base_acc = accuracy_score(yc_test, clf_base.predict(Xc_test))
print(f"baseline accuracy: {base_acc:.4f}")
```

Then the search:

```python
set_seed(RANDOM_STATE)
clf = MLPClassifier(
    model_config=MLPConfig(),
    preprocessing_config=PREPROC,
    trainer_config=TRAINER,
    random_state=RANDOM_STATE,
)

best_clf = clf.optimize_hparams(
    Xc_train, yc_train,
    X_val=Xc_val, y_val=yc_val,
    time=15,
    max_epochs=5,
    prune_by_epoch=True,
    prune_epoch=2,
)

set_seed(RANDOM_STATE)
clf.fit(Xc_train, yc_train, X_val=Xc_val, y_val=yc_val, random_state=RANDOM_STATE)
tuned_acc = accuracy_score(yc_test, clf.predict(Xc_test))
print(f"baseline accuracy: {base_acc:.4f}   tuned accuracy: {tuned_acc:.4f}")
```

The search minimizes validation cross-entropy, a smoother and better-behaved
target than accuracy, while you report accuracy (or any metric you care about) on
the test set afterwards. Optimizing the loss and reporting the metric is the
standard, robust separation.

---

## Customizing the Search

The default search space is sensible, but you will often want to narrow it,
widen it, or pin certain choices. Two arguments give you full control, and both
are passed straight through to `get_search_space()`.

### Fixing parameters

`fixed_params` sets config fields to a constant and removes them from the search.
This shrinks the space so the optimizer spends its trial budget on the choices
that matter to you. Note that supplying your own `fixed_params` replaces the
default dict, so include any defaults you still want to keep.

```python
set_seed(RANDOM_STATE)
narrow = MLPRegressor(
    model_config=MLPConfig(),
    preprocessing_config=PREPROC,
    trainer_config=TRAINER,
    random_state=RANDOM_STATE,
)

best_narrow = narrow.optimize_hparams(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    time=12,
    max_epochs=5,
    fixed_params={
        "pooling_method": "avg",
        "head_skip_layers": False,
        "head_layer_size_length": 0,
        "cat_encoding": "int",
        "head_skip_layer": False,
        "use_cls": False,
        "activation": "ReLU",   # pin the activation; do not search it
    },
)
print("Tuned activation stays ReLU:", type(narrow.config.activation).__name__)
```

```{note}
You can pin any searchable field this way, including categorical choices and
activations. Activation names (such as `"ReLU"` or `"SELU"`) are mapped to their
`nn.Module` instances automatically, exactly as they are during the search.
```

### Overriding ranges

`custom_search_space` is a dict mapping a field name to a [skopt dimension](https://scikit-optimize.github.io/stable/modules/space.html)
(`Real`, `Integer`, or `Categorical`). It overrides the default range for that
field. Use it to restrict `d_model` to the sizes you can afford, or to widen a
dropout range:

```python
set_seed(RANDOM_STATE)
custom = MLPRegressor(
    model_config=MLPConfig(),
    preprocessing_config=PREPROC,
    trainer_config=TRAINER,
    random_state=RANDOM_STATE,
)

best_custom = custom.optimize_hparams(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    time=12,
    max_epochs=5,
    custom_search_space={
        "d_model": Categorical([64, 128, 256]),  # smaller, cheaper widths only
        "dropout": Real(0.1, 0.4),               # narrower dropout band
    },
)
print("Tuned d_model in {64,128,256}:", custom.config.d_model)
```

You can preview the effect of either argument before searching by passing the
same values to `get_search_space()` and printing the result, exactly as in the
[inspection step](#inspecting-the-search-space-first) above.

---

## Practical Guidance

- **Always pass a validation split.** The objective is measured on `X_val`/`y_val`. Without it the search cannot judge generalization.
- **Start small, then scale.** Use `time=10` to `time=15` while iterating on the space, then raise `time` for the final run.
- **Tune pruning to your patience.** Lowering `prune_epoch` prunes sooner and cheaper but risks discarding slow starters; raising it is safer but costs more.
- **Reproducibility.** The optimizer uses a fixed seed internally, so repeated searches on the same data and space explore the same sequence of trials. Call `set_seed()` before each `fit()` for fully deterministic training.
- **Keep the test set sacred.** Select on validation, report on test, once.

## Next Steps

- [Skewed-Target Regression](skewed_regression): a full regression pipeline that includes an HPO step in context.
- [Uncertainty Quantification](uncertainty_quantification): distributional models, families, and calibration in depth.
- [Imbalanced Classification](imbalance_classification): class weights, thresholds, and metrics for skewed labels.

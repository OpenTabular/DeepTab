# Experimental Models: Evaluating Research-Stage Architectures

<div style="display: flex; gap: 10px; margin-bottom: 20px;">
  <a href="https://colab.research.google.com/github/OpenTabular/DeepTab/blob/main/docs/tutorials/notebooks/experimental.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://github.com/OpenTabular/DeepTab/blob/main/docs/tutorials/notebooks/experimental.ipynb" target="_blank">
    <img src="https://img.shields.io/badge/View%20on-GitHub-181717?logo=github&logoColor=white" alt="View on GitHub"/>
  </a>
</div>

Experimental models live in `deeptab.models.experimental`. They share the exact same estimator workflow as the stable zoo: the same `fit`/`predict`/`save`/`load` surface, the same split-config system, and the same preprocessing pipeline. The difference is that they sit behind a separate import on purpose. Their constructors, defaults, and internals may change between releases without a deprecation cycle, so the explicit import is a deliberate speed bump that keeps surprise upgrades out of code review.

This tutorial goes beyond "import it and call `fit`". It explains what the experimental tier actually guarantees, introduces the three model families currently available, shows what makes each one architecturally distinctive, and walks through a defensible workflow for evaluating a research-stage model: benchmark it against a stable baseline, pin your environment, and persist results reproducibly.

```{note}
The notebook linked above mirrors this tutorial. Use the markdown page for reading; use the notebook when you want to execute cells directly.
```

## What You Will Learn

- What the **experimental tier** promises (and does not promise) compared with stable models.
- The three experimental families, **Trompt**, **ModernNCA**, and **Tangos**, and the idea behind each.
- How to configure each model with its own config class and read the parameters that matter.
- How to **benchmark** an experimental model against a stable baseline so results are trustworthy.
- How to keep experimental work reproducible with **exact version pinning** and the `.deeptab` model bundle.

## What "experimental" means in DeepTab

DeepTab sorts every model into one of two tiers. The tier is a contract about API stability, not a judgement about quality. Several experimental models are strong performers that simply have not finished the promotion process yet.

|                     | Experimental                                       | Stable                                                  |
| ------------------- | -------------------------------------------------- | ------------------------------------------------------- |
| **Import path**     | `deeptab.models.experimental`                      | `deeptab.models`                                        |
| **API stability**   | May change without a deprecation cycle             | Frozen under semantic versioning                        |
| **Recommended pin** | Exact version (`deeptab==2.0.0`)                   | Range (`deeptab>=2.0,<3.0`)                             |
| **Best for**        | Evaluating recent architectures, research feedback | Production, long-running baselines, reproducible suites |

Before an experimental model graduates to the stable zoo it has to clear a documented bar: a conventional public API, a model-zoo page with a limitations section, a runnable end-to-end example, working `save`/`load` with a prediction round-trip test, passing behavioural tests in CI, no open correctness bugs, and registration in the model registry. Until then, treat its defaults as provisional.

```{warning}
Pin the **exact** DeepTab version whenever experimental results go into a paper, a benchmark table, or anything you might need to reproduce later. A range such as `deeptab>=2.0` can silently pull a release that changes an experimental model's behaviour.
```

## The experimental lineup

Three model families are available today, each in `Classifier`, `Regressor`, and `LSS` (distributional) variants. They come from different corners of the tabular deep-learning literature, so they fail and succeed on different kinds of data, which is exactly why benchmarking matters.

| Model         | Core idea                                                                                                                                                                   | Config class      | Primary controls                                |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ----------------------------------------------- |
| **Trompt**    | Prompt-style aggregation: learnable prototype records repeatedly read column representations through feature-importance maps, emitting one prediction per cycle.            | `TromptConfig`    | `n_cycles`, `P`, `d_model`                      |
| **ModernNCA** | A differentiable nearest-neighbour model: rows are embedded, compared to candidate rows by distance, and predicted from a temperature-weighted average of candidate labels. | `ModernNCAConfig` | `dim`, `n_blocks`, `temperature`, `sample_rate` |
| **Tangos**    | An MLP with a gradient-attribution regularizer that pushes hidden units to specialise and decorrelate, aiming for better generalisation on small tabular data.              | `TangosConfig`    | `layer_sizes`, `lamda1`, `lamda2`               |

The following sections take each model in turn, explain the mechanism in a paragraph, and then train it on a small synthetic dataset.

## Setup

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from deeptab.configs import ModernNCAConfig, PreprocessingConfig, TangosConfig, TrainerConfig, TromptConfig
from deeptab.models import MambularClassifier
from deeptab.models.experimental import ModernNCARegressor, TangosClassifier, TromptClassifier
```

```{note}
For a quick demonstration these tutorials train with very low `max_epochs` and `patience` (5 and 2). Treat these as placeholders and choose values that match your own compute budget and problem. As a starting point, at least `max_epochs=100` and `patience=10` are recommended for meaningful results.
```

## Data

Two small synthetic datasets are reused throughout: a three-class classification problem (for Trompt and Tangos) and a regression problem (for ModernNCA). Building them once keeps the model sections comparable.

```python
# Shared classification dataset (3 classes), used by Trompt and Tangos.
Xc_num, yc = make_classification(
    n_samples=1000,
    n_features=8,
    n_informative=5,
    n_classes=3,
    random_state=101,
)
Xc = pd.DataFrame(Xc_num, columns=[f"num_{i}" for i in range(Xc_num.shape[1])])
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, stratify=yc, random_state=101
)

# Shared regression dataset, used by ModernNCA.
Xr_num, yr = make_regression(
    n_samples=1000,
    n_features=8,
    n_informative=6,
    noise=10.0,
    random_state=101,
)
Xr = pd.DataFrame(Xr_num, columns=[f"num_{i}" for i in range(Xr_num.shape[1])])
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.2, random_state=101
)

print("classification:", Xc_train.shape, "| regression:", Xr_train.shape)
```

## Trompt: prompt-style feature aggregation

Trompt is inspired by prompt learning. Instead of a single forward pass, it runs several **cycles**: a set of `P` learnable prototype records reads the embedded columns through a feature-importance map, aggregates them, and updates itself, producing one prediction per cycle. The cycle predictions are combined into the final output, which gives Trompt an ensemble-like character from a single model.

The parameters you will tune most are `n_cycles` (how many read-aggregate rounds) and `P` (how many prototype records). `d_model` sets the embedding width.

| Field      | Default | Meaning                                                   |
| ---------- | ------- | --------------------------------------------------------- |
| `d_model`  | `128`   | Embedding dimensionality.                                 |
| `n_cycles` | `6`     | Number of read-aggregate cycles; each emits a prediction. |
| `n_cells`  | `4`     | Declared cells per cycle (see the note below).            |
| `P`        | `128`   | Number of learnable prototype records.                    |

```python
trompt = TromptClassifier(
    model_config=TromptConfig(d_model=128, n_cycles=6, n_cells=4, P=128),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(max_epochs=5, batch_size=128, lr=3e-4, patience=2),
    random_state=101,
)
trompt.fit(Xc_train, yc_train)

trompt_pred = trompt.predict(Xc_test)
print("Trompt accuracy:", round(accuracy_score(yc_test, trompt_pred), 3))
```

```{important}
Trompt is configured with `TromptConfig`, never a stable config such as `MambularConfig`. Each experimental model has its own config class, and mixing them raises a validation error.
```

```{note}
The current DeepTab implementation builds one cell per cycle, so `n_cycles` and `P` are the primary practical controls; `n_cells` is accepted for forward compatibility. Trompt also does not use a standard multi-head self-attention stack, so there is no `n_heads` to tune.
```

## ModernNCA: a differentiable nearest-neighbour model

ModernNCA modernises Neighbourhood Component Analysis. It learns a neural representation of each row, then predicts a query row by comparing it to a set of **candidate** rows in that representation space: distances are turned into weights by a temperature-scaled softmax, and the prediction is the weighted average of the candidates' labels. It behaves like a learned, soft k-nearest-neighbours.

Two parameters deserve attention. `temperature` controls how sharply the softmax favours the closest candidates (lower is sharper). `sample_rate` is the fraction of training rows used as candidates on each forward pass, and it changes the stochastic training objective, so it should be reported alongside any benchmark numbers.

| Field         | Default | Meaning                                                |
| ------------- | ------- | ------------------------------------------------------ |
| `dim`         | `128`   | Per-feature embedding dimensionality.                  |
| `n_blocks`    | `4`     | Number of residual blocks in the encoder.              |
| `temperature` | `0.75`  | Softmax temperature over candidate distances.          |
| `sample_rate` | `0.5`   | Fraction of training rows used as candidates per step. |

```python
nca = ModernNCARegressor(
    model_config=ModernNCAConfig(dim=128, n_blocks=4, temperature=0.75, sample_rate=0.5),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(max_epochs=5, batch_size=128, lr=3e-4, patience=2),
    random_state=101,
)
nca.fit(Xr_train, yr_train)

nca_pred = nca.predict(Xr_test)
print("ModernNCA RMSE:", round(np.sqrt(mean_squared_error(yr_test, nca_pred)), 3))
```

```{important}
The pairwise distance computation is the dominant cost, roughly proportional to `batch_size x n_candidates x dim`. On large datasets, watch memory and step time, and tune `sample_rate` to trade accuracy for speed.
```

## Tangos: an MLP with a gradient-attribution regularizer

Tangos is a standard dense network with an unusual training objective. During training it computes the Jacobian of the latent representation with respect to the inputs and adds two penalties: a **specialisation** term that encourages each hidden unit to attribute to few inputs, and an **orthogonalisation** term that pushes different units to attend to different inputs. The total loss is

$$L_{\text{total}} = L_{\text{task}} + \lambda_1 L_{\text{spec}} + \lambda_2 L_{\text{orth}}$$

where `lamda1` and `lamda2` weight the two regularizers. The goal is better generalisation on small tabular datasets, at the cost of a more expensive backward pass.

| Field         | Default          | Meaning                                                  |
| ------------- | ---------------- | -------------------------------------------------------- |
| `layer_sizes` | `[256, 128, 32]` | Hidden layer widths of the MLP body.                     |
| `lamda1`      | `0.5`            | Weight of the specialisation penalty ($\lambda_1$).      |
| `lamda2`      | `0.1`            | Weight of the orthogonalisation penalty ($\lambda_2$).   |
| `subsample`   | `0.5`            | Fraction of features sampled when computing the penalty. |

```python
tangos = TangosClassifier(
    model_config=TangosConfig(layer_sizes=[256, 128, 32], lamda1=0.5, lamda2=0.1),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="standardization"),
    trainer_config=TrainerConfig(max_epochs=5, batch_size=128, lr=1e-3, patience=2),
    random_state=101,
)
tangos.fit(Xc_train, yc_train)

tangos_pred = tangos.predict(Xc_test)
print("Tangos accuracy:", round(accuracy_score(yc_test, tangos_pred), 3))
```

```{note}
The Jacobian-based penalty makes each training step noticeably heavier than a plain MLP. Start with the default `lamda1`/`lamda2` and only increase them if the model overfits; setting both to `0` recovers an ordinary MLP.
```

## Benchmark against a stable baseline

An experimental result is only meaningful next to a reference you trust. The most useful habit when evaluating any experimental model is to run it against a stable baseline under identical preprocessing and trainer settings, then compare on held-out data. Here we put both experimental classifiers next to stable Mambular on the shared classification task.

```python
PREPROC = PreprocessingConfig(numerical_preprocessing="quantile")
TRAINER = TrainerConfig(max_epochs=5, batch_size=128, patience=2)

candidates = {
    "Mambular (stable)": MambularClassifier(
        preprocessing_config=PREPROC, trainer_config=TRAINER, random_state=101
    ),
    "Trompt (experimental)": TromptClassifier(
        model_config=TromptConfig(d_model=128, n_cycles=4, n_cells=4, P=128),
        preprocessing_config=PREPROC, trainer_config=TRAINER, random_state=101,
    ),
    "Tangos (experimental)": TangosClassifier(
        model_config=TangosConfig(layer_sizes=[256, 128, 32]),
        preprocessing_config=PREPROC, trainer_config=TRAINER, random_state=101,
    ),
}

rows = []
for name, estimator in candidates.items():
    estimator.fit(Xc_train, yc_train)
    acc = accuracy_score(yc_test, estimator.predict(Xc_test))
    rows.append({"model": name, "accuracy": round(acc, 3)})

pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
```

```{tip}
Treat every experimental result as a hypothesis. With only five epochs on a synthetic dataset these numbers are illustrative, not verdicts. For a real comparison, train to convergence, average over several seeds, and keep the baseline and the candidate on the same preprocessing and trainer settings.
```

## Reproducibility: pinning and persistence

Because experimental APIs can shift, reproducibility rests on two habits: pin the exact package version, and save the fitted model as a self-contained bundle.

DeepTab's `.deeptab` bundle is the canonical artifact. It stores the architecture and config, the network weights, the fitted preprocessing state, the feature schema and column order, the task metadata and class labels, and the package versions used to create it. That is everything needed to reload and predict in another environment. (Saving with a `.pt` extension still works but emits a warning; prefer `.deeptab`.)

```python
import deeptab

print("Pin this exact version for experimental runs:")
print(f"    pip install deeptab=={deeptab.__version__}")

# Persist the fitted Trompt model and reload it.
path = trompt.save("trompt_model.deeptab")
reloaded = TromptClassifier.load(path)

assert (reloaded.predict(Xc_test) == trompt_pred).all()
print("Reloaded model reproduces the original predictions.")
```

## A checklist for experimental work

1. Import from `deeptab.models.experimental` so the dependency on a research-stage API is explicit.
2. Configure each model with its own config class (`TromptConfig`, `ModernNCAConfig`, `TangosConfig`).
3. Pin the exact DeepTab version in any environment whose results you need to reproduce.
4. Benchmark against at least one stable baseline (MLP, ResNet, TabM, or Mambular) before drawing conclusions.
5. Average over several seeds and report stochastic settings such as ModernNCA's `sample_rate`.
6. Save fitted models as `.deeptab` bundles, and read the model-zoo page for each model's known limitations.

## Next Steps

- [Experimental model zoo](../model_zoo/experimental/index): per-model pages with parameter tables and limitations.
- [Model tiers](../core_concepts/model_tiers): the full stability contract and promotion policy.
- [Stable model zoo](../model_zoo/stable/index): the baselines to benchmark against.
- [Advanced training](advanced_training): optimizers, schedulers, and production inference for any model.

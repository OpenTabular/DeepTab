# Migrating from v1 to v2

DeepTab v2 keeps the `fit` / `predict` / `evaluate` workflow you already know and
rebuilds the rest around it. The package layout, the configuration objects, and a
few import paths changed, so v1 code needs a handful of edits before it runs. This
page walks through each change with before and after examples.

```{warning}
**v2 is not backward compatible with v1, and v1 is no longer maintained.** The v1
branch receives no bug fixes or security updates. If you are not ready to upgrade,
pin `deeptab<2.0` and plan the move when you can.
```

## Before you upgrade

Pin the major version you test against so a future release never surprises a running
pipeline:

```bash
pip install "deeptab>=2.0,<3.0"
```

Most projects only touch two things: how the estimator is built, and a few import
lines. If you only ever called `Model().fit(...)` and `predict(...)` with defaults,
the change is small. If you imported from internal modules, read [Import
paths](import-paths) closely.

## Split configs

This is the one change almost every project needs. In v1, architecture,
preprocessing, and training options were flat keyword arguments on the estimator. In
v2 they live in three focused config objects, so you can tune one concern without
touching the others.

```python
# v1: every option flat on the estimator
from deeptab.models import MambularClassifier

model = MambularClassifier(
    d_model=128,
    n_layers=4,
    numerical_preprocessing="ple",
    lr=1e-3,
)
```

```python
# v2: options grouped by concern
from deeptab.models import MambularClassifier
from deeptab.configs import MambularConfig, PreprocessingConfig, TrainerConfig

model = MambularClassifier(
    model_config=MambularConfig(d_model=128, n_layers=4),                      # architecture
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="ple"),   # features
    trainer_config=TrainerConfig(lr=1e-3),                                     # training
)
```

Each option moves to the config that owns its concern:

| v1 argument was about  | Now lives on          | Typical fields                                                   |
| ---------------------- | --------------------- | ---------------------------------------------------------------- |
| Neural architecture    | `<Model>Config`       | `d_model`, `n_layers`, `n_heads`, `dropout`, `layer_sizes`       |
| Feature handling       | `PreprocessingConfig` | `numerical_preprocessing`, `categorical_preprocessing`, `n_bins` |
| Training and optimizer | `TrainerConfig`       | `max_epochs`, `batch_size`, `lr`, `patience`, `optimizer_type`   |

```{important}
Flat v1 keyword arguments are no longer accepted. A call like
`MambularClassifier(d_model=128)` now raises a `TypeError`. Move every setting into
its config object.
```

```{tip}
All three configs are optional. `MambularClassifier()` runs on sensible defaults, so
you only build the configs whose defaults you want to change.
```

The [Config System](../core_concepts/config_system) guide is the full field reference.

## Config class renames

Configuration classes dropped their `Default` prefix and moved to `deeptab.configs`.
A find and replace covers it.

| v1                           | v2                    |
| ---------------------------- | --------------------- |
| `DefaultMambularConfig`      | `MambularConfig`      |
| `DefaultFTTransformerConfig` | `FTTransformerConfig` |
| `Default<Arch>Config`        | `<Arch>Config`        |

```python
# v1
from deeptab.models import DefaultMambularConfig

# v2
from deeptab.configs import MambularConfig
```

(import-paths)=

## Import paths

The package was reorganised under the `deeptab` namespace. Estimators still come from
`deeptab.models`; supporting pieces moved to dedicated subpackages:

| What you import                                                  | v2 location             |
| ---------------------------------------------------------------- | ----------------------- |
| Estimators (`MambularClassifier`, ...)                           | `deeptab.models`        |
| Config objects                                                   | `deeptab.configs`       |
| Data layer                                                       | `deeptab.data`          |
| Distributions for `LSS`                                          | `deeptab.distributions` |
| Metrics                                                          | `deeptab.metrics`       |
| Top-level helpers (`InferenceModel`, `set_seed`, `seed_context`) | `deeptab`               |

```{tip}
If an import breaks after upgrading, it was probably reaching into an internal v1
module. Code that used only the public estimator and config imports needs the least
work.
```

## Data layer renames

The data classes were renamed to give the pipeline a clear, typed contract. The old
`Mambular*` aliases are deprecated.

| v1                              | v2                  |
| ------------------------------- | ------------------- |
| `Mambular*` data module aliases | `TabularDataModule` |
| `Mambular*` dataset aliases     | `TabularDataset`    |

v2 also adds `FeatureSchema`, an inspectable description of the columns a model
expects, their order, and their types. You rarely build one by hand: DeepTab derives
it from your data during `fit` and stores it inside saved artifacts.

## Saving, loading, and serving

Persistence goes through a single self-describing `.deeptab` artifact that bundles
the architecture, feature schema, preprocessing, task type, and package versions
alongside the weights. A saved model carries everything it needs to reload itself.

```python
# Save a fitted estimator
clf.save("churn_model.deeptab")

# Load it back as a full estimator
from deeptab.models import MambularClassifier
clf = MambularClassifier.load("churn_model.deeptab")

# Or load a read-only serving surface
from deeptab import InferenceModel
served = InferenceModel.from_path("churn_model.deeptab")
predictions = served.predict(X_new)
```

```{note}
`InferenceModel` exposes prediction and input validation only. Training is absent by
design, so a served model cannot be re-fitted by accident. See
[Inference](../core_concepts/inference) for the full serving surface.
```

## Experimental models

ModernNCA, Tangos, and Trompt import from a separate namespace that makes their less
stable status explicit:

```python
from deeptab.models.experimental import ModernNCAClassifier, TangosRegressor, TromptClassifier
```

```{warning}
Experimental models may change in minor releases. Pin an exact version such as
`deeptab==2.0.0` if you depend on one in production.
```

## What did not change

The day to day modeling surface is intentionally stable:

- `fit`, `predict`, `predict_proba`, and `evaluate` behave exactly as in v1.
- DeepTab is still scikit-learn compatible, including use inside `GridSearchCV`.
- Every architecture still ships as a classifier, a regressor, and a distributional
  (`LSS`) variant sharing one interface.
- Automatic preprocessing and feature-type detection still run for you.

## Upgrade checklist

1. Pin `deeptab>=2.0,<3.0` and install into a clean environment.
2. Move flat estimator arguments into `MambularConfig`, `PreprocessingConfig`, and
   `TrainerConfig` (or the configs for your model).
3. Rename `Default<Arch>Config` to `<Arch>Config` and import it from `deeptab.configs`.
4. Update imports: estimators from `deeptab.models`, configs from `deeptab.configs`,
   data classes from `deeptab.data`.
5. Replace `Mambular*` data module and dataset references with `TabularDataModule`
   and `TabularDataset`.
6. Run your training script. A `TypeError` from the constructor almost always means a
   setting still needs to move into a config.

```{seealso}
The [FAQ](faq) covers common upgrade questions and the v1 support policy. The [Config
System](../core_concepts/config_system) guide is the authoritative field reference.
```

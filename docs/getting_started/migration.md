# Migrating from v1 to v2

DeepTab v2 keeps the part of v1 you use most, the `fit` / `predict` / `evaluate`
workflow, and rebuilds everything around it. The package layout, the configuration
objects, and a handful of import paths changed, so existing v1 code needs a few
edits before it runs on v2. This page walks through each change and shows the
before and after side by side.

```{warning}
**v2 is not backward compatible with v1, and v1 is no longer maintained.** The v1
branch receives no bug fixes or security updates. If you are not ready to upgrade,
pin `deeptab<2.0` for now and plan the move when you can.
```

## Before you upgrade

Pin the major version you are testing against so an upgrade never surprises a
running pipeline:

```bash
pip install "deeptab>=2.0,<3.0"
```

Most projects only touch two things: how the estimator is constructed, and a few
import lines. If your code only ever called `Model().fit(...)` and `predict(...)`
with default settings, the change is small. If you reached into internal modules,
read the [Import paths](import-paths) section closely.

## The change almost every project needs: split configs

In v1, architecture, preprocessing, and training options were all flat keyword
arguments on the estimator. In v2 those same options live in three focused config
objects, so you can tune one concern without disturbing the others.

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
    model_config=MambularConfig(d_model=128, n_layers=4),      # architecture
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="ple"),  # features
    trainer_config=TrainerConfig(lr=1e-3),                     # training
)
```

Each option moves to the config that owns its concern:

| Where the v1 argument went | Lives in v2 on        | Typical fields                                                   |
| -------------------------- | --------------------- | ---------------------------------------------------------------- |
| Neural architecture        | `<Model>Config`       | `d_model`, `n_layers`, `n_heads`, `dropout`, `layer_sizes`       |
| Feature handling           | `PreprocessingConfig` | `numerical_preprocessing`, `categorical_preprocessing`, `n_bins` |
| Training and optimizer     | `TrainerConfig`       | `max_epochs`, `batch_size`, `lr`, `patience`, `optimizer_type`   |

```{important}
The flat keyword arguments from v1 are no longer accepted. Passing them, for
example `MambularClassifier(d_model=128)`, now raises a `TypeError`. Move every
setting into the matching config object.
```

```{tip}
All three configs are optional. `MambularClassifier()` with no arguments uses
sensible defaults for architecture, preprocessing, and training, so you only build
the configs whose defaults you actually want to change.
```

For the full field reference and which config owns each setting, see the
[Config System](../core_concepts/config_system) guide.

## Config class renames

The configuration classes dropped their `Default` prefix. The rename is mechanical,
so a find and replace across your codebase covers it.

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

```{note}
Config classes now live under `deeptab.configs`, not alongside the models. Importing
them from `deeptab.configs` is the supported path going forward.
```

(import-paths)=

## Import paths

The internal package layout was reorganised under the `deeptab` namespace. The
high-level estimators are still imported from `deeptab.models`, but supporting
pieces moved to dedicated subpackages:

| What you import                                                  | v2 location             |
| ---------------------------------------------------------------- | ----------------------- |
| Estimators (`MambularClassifier`, ...)                           | `deeptab.models`        |
| Config objects                                                   | `deeptab.configs`       |
| Data layer                                                       | `deeptab.data`          |
| Distributions for `LSS`                                          | `deeptab.distributions` |
| Metrics                                                          | `deeptab.metrics`       |
| Top-level helpers (`InferenceModel`, `set_seed`, `seed_context`) | `deeptab`               |

```{tip}
If an import fails after upgrading, check whether you were importing from an
internal v1 module. Code that only used the public estimator and config imports
above needs the least work.
```

## Data layer renames

The data modules were renamed to give the pipeline a clear, typed contract:

| v1                              | v2                  |
| ------------------------------- | ------------------- |
| `Mambular*` data module aliases | `TabularDataModule` |
| `Mambular*` dataset aliases     | `TabularDataset`    |

The v2 data layer also adds `FeatureSchema`, an inspectable description of the
columns a model expects, their order, and their types. You rarely build these by
hand; DeepTab constructs them from your data during `fit` and stores the schema
inside saved artifacts. The old `Mambular*` aliases are deprecated.

## Saving, loading, and serving

Persistence in v2 goes through a single self-describing `.deeptab` artifact that
bundles the architecture, feature schema, preprocessing, task type, and package
versions alongside the weights. A saved model therefore carries everything it needs
to reload itself.

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
`InferenceModel` exposes only prediction and input-validation methods. Training is
deliberately absent, so a served model cannot be re-fitted by accident. See
[Inference](../core_concepts/inference) for the full serving surface.
```

## Experimental models

The experimental models, ModernNCA, Tangos, and Trompt, import from a separate
namespace so their less stable status is explicit:

```python
from deeptab.models.experimental import ModernNCAClassifier, TangosRegressor, TromptClassifier
```

```{warning}
Experimental models may change in minor releases. Pin an exact version, for example
`deeptab==2.0.0`, if you depend on one in production.
```

## What did not change

The day-to-day modeling surface is intentionally stable:

- `fit`, `predict`, `predict_proba`, and `evaluate` behave exactly as in v1.
- DeepTab is still scikit-learn compatible, including use inside `GridSearchCV`.
- Every architecture still ships as a classifier, a regressor, and a distributional
  (`LSS`) variant sharing one interface.
- Automatic preprocessing and feature-type detection still run for you.

## Upgrade checklist

Use this as a quick pass over a v1 codebase:

1. Pin `deeptab>=2.0,<3.0` and install into a clean environment.
2. Move flat estimator arguments into `MambularConfig` / `PreprocessingConfig` /
   `TrainerConfig` (or the config for your model).
3. Rename `Default<Arch>Config` to `<Arch>Config` and import it from
   `deeptab.configs`.
4. Update imports: estimators from `deeptab.models`, configs from `deeptab.configs`,
   data classes from `deeptab.data`.
5. Replace any `Mambular*` data module or dataset references with
   `TabularDataModule` and `TabularDataset`.
6. Run your training script. A `TypeError` on the estimator constructor almost
   always means a setting still needs to move into a config object.

```{seealso}
The [FAQ](faq) answers common upgrade questions and the v1 support policy. The
[Config System](../core_concepts/config_system) guide is the authoritative field
reference for the three config objects.
```

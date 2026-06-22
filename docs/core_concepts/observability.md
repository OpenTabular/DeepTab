# Observability

DeepTab can record what happens during training without you writing a single callback. You attach an `ObservabilityConfig` to an estimator, and every `fit()` captures its hyperparameters, lifecycle events, and final metrics in one self-contained run directory. Optional experiment trackers (TensorBoard, MLflow) and structured logging build on the same configuration.

```{note}
Observability is entirely opt-in. Estimators created without an `ObservabilityConfig` train exactly as before and emit nothing, so notebooks stay quiet by default.
```

The default `pip install deeptab` does not include the observability backends. Install the extra you need:

```bash
pip install 'deeptab[logs]'         # structlog (structured logging)
pip install 'deeptab[tensorboard]'  # TensorBoard tracker
pip install 'deeptab[mlflow]'       # MLflow tracker
pip install 'deeptab[tracking]'     # TensorBoard + MLflow
pip install 'deeptab[all]'          # structlog + TensorBoard + MLflow
```

```{note}
Each backend is loaded lazily, so a missing package raises only when you enable the matching feature.
```

---

## Attaching observability

There are two equivalent ways to enable it. Pass the config at construction time:

```python
from deeptab.core.observability import ObservabilityConfig
from deeptab.models import MambularClassifier

obs = ObservabilityConfig(
    experiment_name="churn_baseline",
    structured_logging=True,          # console event log (add log_to_file=True for JSONL)
    experiment_trackers=["mlflow"],   # also supports "tensorboard"
)

model = MambularClassifier(observability_config=obs)
model.fit(X_train, y_train, max_epochs=50)
```

Or attach it to an already-constructed estimator. Changes take effect on the next `fit()` call:

```python
model = MambularClassifier()
model.configure_observability(obs)
model.fit(X_train, y_train, max_epochs=50)
```

```{important}
Structured logging relies on `structlog`, which is an optional dependency. Install it with `pip install 'deeptab[logs]'`. The experiment trackers need their own packages too: `tensorboard` for TensorBoard and `mlflow` for MLflow.
```

---

## The run directory

Every output path is derived from `root_dir`, producing a single organised tree per run:

```text
deeptab_runs/
  runs/churn_baseline/20260611_174830_8f3a2c1d/
    config.yaml       # estimator hyperparameters
    lifecycle.jsonl   # structured event log (when log_to_file=True)
    summary.json      # final metrics
    checkpoints/best_model.ckpt
  tensorboard/churn_baseline/20260611_174830_8f3a2c1d/
    events.out.tfevents...
  mlflow/
    backend/mlflow.db
    artifacts/
```

The run identifier combines a timestamp and a short hash, so concurrent or repeated runs never overwrite each other.

---

## Configuration reference

`ObservabilityConfig` is a dataclass. All fields are optional and resolve sensible defaults relative to `root_dir`.

| Field                      | Default          | Purpose                                                                               |
| -------------------------- | ---------------- | ------------------------------------------------------------------------------------- |
| `root_dir`                 | `"deeptab_runs"` | Base directory for all observability outputs.                                         |
| `experiment_name`          | `"default"`      | Logical label used to group related runs.                                             |
| `structured_logging`       | `False`          | Enable structured runtime logging via `structlog`.                                    |
| `log_to_console`           | `True`           | Stream compact human-readable output to stdout.                                       |
| `log_to_file`              | `False`          | Write a per-run `lifecycle.jsonl` inside the run directory.                           |
| `verbosity`                | `1`              | Which lifecycle events are emitted when `structured_logging=True` (see below).        |
| `experiment_trackers`      | `[]`             | Lightning loggers to activate: `"tensorboard"`, `"mlflow"`, or both.                  |
| `tensorboard_save_dir`     | `""`             | Resolved to `<root_dir>/tensorboard` when empty.                                      |
| `tensorboard_name`         | `"deeptab"`      | Reserved label field; the TensorBoard sub-directory currently uses `experiment_name`. |
| `mlflow_experiment_name`   | `"deeptab"`      | Name of the MLflow experiment.                                                        |
| `mlflow_tracking_uri`      | `""`             | Resolved to a local SQLite store under `<root_dir>/mlflow` when empty.                |
| `mlflow_artifact_location` | `""`             | Resolved to `<root_dir>/mlflow/artifacts` when empty.                                 |
| `mlflow_run_name`          | `None`           | Human-readable label for the MLflow run.                                              |
| `mlflow_log_model`         | `True`           | Upload model checkpoints as MLflow artifacts.                                         |
| `logger`                   | `None`           | A user-provided Lightning logger appended alongside any built-in trackers.            |

```{note}
`experiment_trackers` is a list, not a single string. Pass `["tensorboard"]`, `["mlflow"]`, or `["mlflow", "tensorboard"]` to activate one or both.
```

---

## Verbosity levels

When `structured_logging=True`, `verbosity` controls how much is emitted. Higher levels are supersets of lower ones.

| Level | Emits                                                                           |
| ----- | ------------------------------------------------------------------------------- |
| `0`   | Silent.                                                                         |
| `1`   | Milestones: `fit.started`, `model.created`, `train.completed`, `fit.completed`. |
| `2`   | Level 1 plus `data.created` and `train.started`.                                |
| `3`   | Debug: all events.                                                              |

The default of `1` keeps console output to a few meaningful milestones.

---

## Lifecycle events

Events are dot-namespaced and carry structured metadata, which makes them easy to filter, parse, and compare across runs. For example, `fit.started` records sample counts, `model.created` records the parameter count, and `train.completed` records the best validation loss.

```{tip}
For experiment sweeps, set `log_to_file=True` and read each run's `lifecycle.jsonl`. Because every record is a JSON object tagged with the same `run_id`, you can load many runs into a DataFrame and compare them programmatically.
```

---

## Bring your own framework

If you already have a logging and experiment-tracking stack (your own callbacks, a managed tracking service, or an in-house framework), you do not need DeepTab observability at all. Construct estimators without an `ObservabilityConfig` and they stay silent, leaving your existing setup in full control.

```python
# No ObservabilityConfig: DeepTab emits nothing and your own stack runs as-is.
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)
```

When you do want DeepTab to coexist with an existing setup, you have two integration points.

**Plug in your own Lightning logger.** DeepTab trains through PyTorch Lightning, so any Lightning logger works. Pass it via the `logger` field and DeepTab appends it alongside any built-in trackers rather than replacing them:

```python
from lightning.pytorch.loggers import WandbLogger
from deeptab.core.observability import ObservabilityConfig

obs = ObservabilityConfig(
    logger=WandbLogger(project="churn"),   # your existing tracker
    experiment_trackers=["tensorboard"],   # needed for the custom logger to attach
)

model = MambularClassifier(observability_config=obs)
model.fit(X_train, y_train, max_epochs=50)
```

```{warning}
A custom `logger` is only attached when `experiment_trackers` has at least one entry. With an empty `experiment_trackers`, DeepTab suppresses all Lightning loggers (so no stray `lightning_logs/` directory is created) and the `logger` is dropped. Keep at least one tracker active, such as `["tensorboard"]` above, for your logger to be picked up.
```

```{note}
The `logger` field accepts a single Lightning logger instance. To attach several at once, wire them through the trackers you control or compose them in your own framework, then hand DeepTab the one entry point.
```

**Consume the lifecycle events yourself.** With `structured_logging=True`, events are emitted through `structlog`. You can route them into your own sinks by configuring `structlog` processors at the application level, or by reading each run's `lifecycle.jsonl` and forwarding the records to your tracking system. This keeps DeepTab's run metadata available without committing to its built-in trackers.

```{tip}
A common pattern is to let your framework own the experiment dashboard while DeepTab owns the per-run artifact directory. Point `root_dir` at a path your pipeline already archives, and the `config.yaml` plus `summary.json` become a portable record your tooling can ingest.
```

---

## Next Steps

- [Training and Evaluation](training_and_evaluation): the fit pipeline, configs, and callbacks that observability wraps around
- [Model Operations](model_operations): saving, loading, and inspecting fitted estimators
- [Config System](config_system): how `ObservabilityConfig` fits alongside the model, preprocessing, and trainer configs

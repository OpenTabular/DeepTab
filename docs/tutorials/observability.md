# Observability: Logging, Tracking, and Run Directories

<div style="display: flex; gap: 10px; margin-bottom: 20px;">
  <a href="https://colab.research.google.com/github/OpenTabular/DeepTab/blob/main/docs/tutorials/notebooks/observability.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://github.com/OpenTabular/DeepTab/blob/main/docs/tutorials/notebooks/observability.ipynb" target="_blank">
    <img src="https://img.shields.io/badge/View%20on-GitHub-181717?logo=github&logoColor=white" alt="View on GitHub"/>
  </a>
</div>

DeepTab can record everything that happens during training without you writing a single
callback. You attach an `ObservabilityConfig` to an estimator and every `fit()` captures its
hyperparameters, lifecycle events, and final metrics in one self-contained run directory.
Optional experiment trackers (TensorBoard, MLflow) and your own Lightning loggers build on the
same configuration.

This tutorial is deliberately exhaustive. We train the **same model** many times, changing **one
observability setting at a time**, and after every run we show the resulting **directory tree** so
you can see exactly what each setting produces on disk and on the console.

```{note}
The notebook linked above mirrors this tutorial. Use the markdown page for reading; use the
notebook when you want to execute cells directly.
```

## What you will learn

- What a run with **no observability** does (and does not) leave behind.
- How a minimal `ObservabilityConfig` creates an organised per-run directory: `config.yaml`, `summary.json`, `checkpoints/`.
- How `structured_logging` streams lifecycle events to the console, and how `verbosity` (0-3) changes what you see.
- How `log_to_file` writes a machine-readable `lifecycle.jsonl` you can load into a DataFrame.
- The exact folder trees produced by the **TensorBoard** and **MLflow** experiment trackers.
- Three ways to **bring your own logger**: a Lightning logger through `ObservabilityConfig.logger`, a direct `fit(logger=...)` hand-off, and an in-process lifecycle-event sink.
- A side-by-side comparison of every case so you can pick the right settings for your workflow.

```{note}
For a quick demonstration these tutorials train with very low `max_epochs` and `patience` (5 and 2). Treat these as placeholders and choose values that match your own compute budget and problem. As a starting point, at least `max_epochs=100` and `patience=10` are recommended for meaningful results.
```

```{important}
Structured logging relies on `structlog`, and the experiment trackers need their own packages.
Install the optional extras you intend to use:

- `pip install 'deeptab[logs]'` for structured logging (`structlog`).
- `pip install 'deeptab[tensorboard]'` for the TensorBoard tracker.
- `pip install 'deeptab[mlflow]'` for the MLflow tracker.
```

## Setup

DeepTab and Lightning print a few framework banners on every fit (a device summary, a
parameter-count table) that are useful in isolation but drown out the observability messages this
tutorial is about. Raising those loggers to `ERROR` keeps the output focused; DeepTab's own
events are emitted separately and are unaffected.

```python
import contextlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from deeptab.configs import TrainerConfig
from deeptab.core.observability import ObservabilityConfig
from deeptab.models import MLPClassifier
```

Every run in this tutorial writes under a single scratch directory so the examples stay isolated
and easy to clean up. We recreate it from scratch on each execution so the trees you see below are
reproducible.

```python
WORKDIR = Path("obs_runs").resolve()
if WORKDIR.exists():
    shutil.rmtree(WORKDIR)
WORKDIR.mkdir(parents=True)
print("Scratch directory:", WORKDIR)
```

A small synthetic binary-classification dataset is all we need. Observability behaves identically
for regressors and distributional (LSS) models.

```python
X, y = make_classification(
    n_samples=800, n_features=8, n_informative=6, n_classes=2, random_state=42
)
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### A few small helpers

`show_tree` prints a directory as an indented tree so we can inspect what each run produced.
`latest_run` returns the newest per-run directory. `focused_output` hides DeepTab's per-feature
preprocessing summary (a plain `print` from the preprocessing layer) so that, when we look at
structured logging, the cell output stays on the observability messages. None of these helpers are
required to use observability; they only keep this tutorial readable.

```python
def show_tree(root, title=None):
    """Print *root* as an indented directory tree."""
    root = os.path.abspath(root)
    if title:
        print(title)
    if not os.path.exists(root):
        print("    (nothing was created here)")
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        depth = dirpath[len(root):].count(os.sep)
        print("    " * depth + os.path.basename(dirpath) + "/")
        for name in sorted(filenames):
            print("    " * (depth + 1) + name)


def latest_run(root_dir, experiment_name):
    """Return the newest per-run directory under <root_dir>/runs/<experiment_name>/."""
    runs = Path(root_dir) / "runs" / experiment_name
    return sorted(runs.iterdir())[-1]


_NOISE = re.compile(r"^(Numerical Feature:|Categorical Feature:|Embedding Feature:|-{5,}\s*$)")


class _LineFilter:
    """A thin stdout wrapper that drops the preprocessor's per-feature summary lines."""

    def __init__(self, target):
        self._target = target
        self._buf = ""

    def write(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if not _NOISE.match(line):
                self._target.write(line + "\n")

    def flush(self):
        self._target.flush()


@contextlib.contextmanager
def focused_output():
    real = sys.stdout
    sys.stdout = _LineFilter(real)
    try:
        yield
    finally:
        sys.stdout = real
```

We reuse one tiny `TrainerConfig` and a single `train` helper everywhere. The only thing that
changes between sections is the `observability_config` we hand to the estimator.

```python
TRAINER = TrainerConfig(max_epochs=5, patience=2, batch_size=128)


def train(observability_config=None, **fit_kwargs):
    """Fit a fresh MLPClassifier, optionally with observability attached."""
    model = MLPClassifier(
        trainer_config=TRAINER,
        random_state=42,
        observability_config=observability_config,
    )
    with focused_output():
        model.fit(X_train, y_train, enable_progress_bar=False, **fit_kwargs)
    return model
```

## 1. The baseline: no observability

Observability is entirely opt-in. An estimator created **without** an `ObservabilityConfig` trains
exactly as before and emits no events. There is no run directory, no `config.yaml`, and no event
log. This is why notebooks stay quiet by default.

The only artifact a plain `fit()` leaves behind is the Lightning checkpoint that restores the best
weights. We point its `default_root_dir` at our scratch folder so it does not clutter the working
directory.

```python
baseline = train(default_root_dir=str(WORKDIR / "01_no_observability"))
print("Fitted:", type(baseline).__name__)
print("Test accuracy:", round((baseline.predict(X_test) == y_test).mean(), 3))

show_tree(WORKDIR / "01_no_observability", "01_no_observability/")
```

```text
01_no_observability/
    checkpoints/
        best_model.ckpt
```

Only a `checkpoints/` directory with the best-epoch weights. Nothing was logged, nothing was
tracked. If you already run your own logging stack, this is the mode to use: DeepTab stays out of
the way.

## 2. A minimal `ObservabilityConfig`

The moment you attach an `ObservabilityConfig` (even an empty one), DeepTab creates a single
organised directory for the run. Every output path is derived from `root_dir`. With nothing else
enabled you already get the run's hyperparameters (`config.yaml`), its final metrics
(`summary.json`), and the best checkpoint, all under a timestamped run folder.

```python
obs_min = ObservabilityConfig(
    root_dir=str(WORKDIR / "02_minimal"),
    experiment_name="demo",
)
model = train(obs_min)
show_tree(WORKDIR / "02_minimal", "02_minimal/")
```

```text
02_minimal/
    runs/
        demo/
            20260613_092809_712e4b18/
                config.yaml
                summary.json
                artifacts/
                checkpoints/
                    best_model.ckpt
```

The run directory name combines a timestamp and a short random id
(`<YYYYMMDD_HHMMSS>_<run_id>`), so concurrent or repeated runs never overwrite each other. Reading
the two metadata files it wrote:

```python
run = latest_run(WORKDIR / "02_minimal", "demo")
print("=== config.yaml ===")
print((run / "config.yaml").read_text())

print("=== summary.json ===")
print((run / "summary.json").read_text())
```

```text
=== summary.json ===
{
  "run_id": "712e4b18",
  "model_class": "MLPClassifier",
  "n_params": 78273,
  "n_samples": 640,
  "best_val_loss": 0.6822827458381653,
  "best_epoch": null,
  "n_epochs_run": 5,
  "duration_min": 0.0058
}
```

`config.yaml` is the full, reloadable configuration of the estimator (model, preprocessing, and
trainer configs plus the random state). `summary.json` is the compact result: parameter count,
best validation loss, best epoch, epochs actually run, and wall-clock duration. Together they make
every run self-describing.

## 3. Structured logging and verbosity

Set `structured_logging=True` to stream named lifecycle events. By default they go to the console
as compact, column-aligned lines prefixed with the run id. `verbosity` controls **which** events
you see; higher levels are supersets of lower ones:

| Level | Emits                                                                           |
| ----- | ------------------------------------------------------------------------------- |
| `0`   | Silent.                                                                         |
| `1`   | Milestones: `fit.started`, `model.created`, `train.completed`, `fit.completed`. |
| `2`   | Level 1 plus `data.created` and `train.started`.                                |
| `3`   | Debug: every event.                                                             |

Watch how the same run prints progressively more as we raise `verbosity` from 1 to 3.

```python
for level in (1, 2, 3):
    print(f"\n===================== verbosity = {level} =====================")
    obs = ObservabilityConfig(
        root_dir=str(WORKDIR / f"03_verbosity_{level}"),
        experiment_name="demo",
        structured_logging=True,
        verbosity=level,
    )
    train(obs)
```

```text
===================== verbosity = 1 =====================
2026-06-13 09:46:39 [info] run=f67d60c0  fit.started       model=MLPClassifier  samples=640  features=8  seed=42
2026-06-13 09:46:39 [info] run=f67d60c0  model.created     backbone=MLP  params=78_273  num=8  cat=0  duration_min=0.0000
2026-06-13 09:46:39 [info] run=f67d60c0  train.completed   best_epoch=null  best_val_loss=0.6823  epochs_run=5  duration_min=0.0061
2026-06-13 09:46:39 [info] run=f67d60c0  fit.completed     status=success  model=MLPClassifier  params=78_273  best_val_loss=0.6823  duration_min=0.0069

===================== verbosity = 2 =====================
2026-06-13 09:46:39 [info] run=d5d96374  fit.started       model=MLPClassifier  samples=640  features=8  seed=42
2026-06-13 09:46:39 [info] run=d5d96374  data.created      train=512  val=128  num=8  cat=0  val_size=0.2000  duration_min=0.0004
2026-06-13 09:46:39 [info] run=d5d96374  model.created     backbone=MLP  params=78_273  num=8  cat=0  duration_min=0.0000
2026-06-13 09:46:39 [info] run=d5d96374  train.started     epochs=5  batch=128  lr=null  optimizer=Adam  patience=2  val_size=0.2000
2026-06-13 09:46:40 [info] run=d5d96374  train.completed   best_epoch=null  best_val_loss=0.6823  epochs_run=5  duration_min=0.0051
2026-06-13 09:46:40 [info] run=d5d96374  fit.completed     status=success  model=MLPClassifier  params=78_273  best_val_loss=0.6823  duration_min=0.0057
```

Each event carries structured context: `fit.started` records the sample and feature counts,
`model.created` the backbone and parameter count, `train.completed` the best validation loss and
epoch, and `fit.completed` the total duration. `verbosity=2` adds the data-split and
training-setup events; `verbosity=3` would add any finer-grained events such as save/load and
predict.

```{tip}
`verbosity=0` keeps the run directory and metadata files but emits nothing to the console: useful
for sweeps where you want artifacts on disk without log spam.
```

## 4. Persisting events to `lifecycle.jsonl`

Console output is convenient for a single run, but for sweeps you want machine-readable records.
Set `log_to_file=True` and DeepTab writes one JSON object per event to `lifecycle.jsonl` inside the
run directory. Here we also set `log_to_console=False` so this run writes only to the file.

```python
obs_file = ObservabilityConfig(
    root_dir=str(WORKDIR / "04_with_file"),
    experiment_name="demo",
    structured_logging=True,
    log_to_console=False,
    log_to_file=True,
    verbosity=3,
)
train(obs_file)
show_tree(WORKDIR / "04_with_file", "04_with_file/")
```

```text
04_with_file/
    runs/
        demo/
            20260613_092810_058d84e7/
                config.yaml
                lifecycle.jsonl
                summary.json
                artifacts/
                checkpoints/
                    best_model.ckpt
```

The run folder now also contains `lifecycle.jsonl`. Because every record is a flat JSON object,
you can load a run straight into a DataFrame:

```python
run = latest_run(WORKDIR / "04_with_file", "demo")
events = [json.loads(line) for line in (run / "lifecycle.jsonl").read_text().splitlines()]
pd.DataFrame(events)[["timestamp", "event", "run_id"]]
```

```text
             timestamp            event    run_id
0  2026-06-13T09:46:40      fit.started  29bef1c6
1  2026-06-13T09:46:40     data.created  29bef1c6
2  2026-06-13T09:46:40    model.created  29bef1c6
3  2026-06-13T09:46:40    train.started  29bef1c6
4  2026-06-13T09:46:40  train.completed  29bef1c6
5  2026-06-13T09:46:40    fit.completed  29bef1c6
```

Every record is tagged with the same `run_id`, so you can concatenate `lifecycle.jsonl` files from
many runs and compare them programmatically: training duration per configuration, best validation
loss per seed, and so on.

## 5. What each setting controls

The runtime-logging fields combine independently. This table summarises their effect; the sections
above and below show each one in action.

| Field                 | Default          | Effect                                                                              |
| --------------------- | ---------------- | ----------------------------------------------------------------------------------- |
| `root_dir`            | `"deeptab_runs"` | Base of the whole output tree. Point it at a path your pipeline already archives.   |
| `experiment_name`     | `"default"`      | Groups related runs under `runs/<experiment_name>/`.                                |
| `structured_logging`  | `False`          | Master switch for lifecycle event emission (needs `structlog`).                     |
| `log_to_console`      | `True`           | Stream compact event lines to stdout (only when `structured_logging=True`).         |
| `log_to_file`         | `False`          | Write `lifecycle.jsonl` in the run directory (only when `structured_logging=True`). |
| `verbosity`           | `1`              | Which events are emitted: `0` silent, `1` milestones, `2` detailed, `3` debug.      |
| `experiment_trackers` | `[]`             | Activate Lightning trackers: `"tensorboard"`, `"mlflow"`, or both.                  |
| `logger`              | `None`           | A user-provided Lightning logger appended alongside the trackers.                   |

```{note}
The run directory (`config.yaml`, `summary.json`, `checkpoints/`) is created whenever **any**
`ObservabilityConfig` is attached, regardless of the logging flags. The flags only add console
output, the event file, and trackers.
```

## 6. Experiment trackers

`experiment_trackers` turns on Lightning loggers that record metrics during training. DeepTab
resolves all of their paths under `root_dir` by default, so a tracker adds a sibling folder next to
`runs/` rather than scattering files across your project.

### TensorBoard

```python
obs_tb = ObservabilityConfig(
    root_dir=str(WORKDIR / "06_tensorboard"),
    experiment_name="demo",
    experiment_trackers=["tensorboard"],
)
train(obs_tb)
show_tree(WORKDIR / "06_tensorboard", "06_tensorboard/")
```

```text
06_tensorboard/
    runs/
        demo/
            20260613_094640_70f476cd/
                config.yaml
                summary.json
                artifacts/
                checkpoints/
                    best_model.ckpt
    tensorboard/
        demo/
            20260613_094640_70f476cd/
                events.out.tfevents...
                hparams.yaml
```

Alongside the usual `runs/` tree you now get a `tensorboard/<experiment_name>/<run>/` folder with
the event file and `hparams.yaml`. Point TensorBoard at the `tensorboard/` directory to explore the
curves:

```bash
tensorboard --logdir obs_runs/06_tensorboard/tensorboard
```

### MLflow

The MLflow tracker defaults to a self-contained local store: a SQLite backend plus a file-based
artifact directory, both under `root_dir`.

```python
obs_mlflow = ObservabilityConfig(
    root_dir=str(WORKDIR / "07_mlflow"),
    experiment_name="demo",
    experiment_trackers=["mlflow"],
    mlflow_experiment_name="deeptab-demo",
)
train(obs_mlflow)
show_tree(WORKDIR / "07_mlflow", "07_mlflow/")
```

```text
07_mlflow/
    mlflow/
        artifacts/
            950a0173cd2d4f799fa3267b07e77bf3/
                artifacts/
                    config.yaml
                    summary.json
                    best_model/
                        aliases.txt
                        best_model.ckpt
                        metadata.yaml
                    checkpoints/
                        best_model.ckpt
        backend/
            mlflow.db
    runs/
        demo/
            20260613_094641_259bbfef/
                config.yaml
                summary.json
                artifacts/
                checkpoints/
                    best_model.ckpt
```

MLflow stores run metadata in `mlflow/backend/mlflow.db` and uploads the run's `config.yaml`,
`summary.json`, and the best checkpoint into `mlflow/artifacts/<mlflow_run_id>/`. DeepTab also logs
the flattened hyperparameters, dataset statistics, and final metrics to the MLflow run. Launch the
UI against the same SQLite file:

```bash
mlflow ui --backend-store-uri sqlite:///obs_runs/07_mlflow/mlflow/backend/mlflow.db
```

Set both trackers at once with `experiment_trackers=["tensorboard", "mlflow"]` to get both trees
from a single run.

## 7. Bring your own logger

If you already have a logging or experiment-tracking stack, DeepTab can hand off to it instead of
(or alongside) its built-in trackers. There are three integration points, from most to least
integrated.

### 7a. A Lightning logger through `ObservabilityConfig.logger`

Because DeepTab trains through PyTorch Lightning, any Lightning logger works. Pass an instance via
the `logger` field and DeepTab appends it to the trainer's logger list. We use `CSVLogger` here
because it writes a folder you can see; the same pattern applies to `WandbLogger`, `CometLogger`,
`NeptuneLogger`, and friends.

```python
from lightning.pytorch.loggers import CSVLogger

obs_byo = ObservabilityConfig(
    root_dir=str(WORKDIR / "08_byo_logger"),
    experiment_name="demo",
    experiment_trackers=["tensorboard"],   # see the note below: at least one tracker is required
    logger=CSVLogger(save_dir=str(WORKDIR / "08_byo_logger" / "csv"), name="mlp"),
)
train(obs_byo)
show_tree(WORKDIR / "08_byo_logger", "08_byo_logger/")
```

```text
08_byo_logger/
    csv/
        mlp/
            version_0/
                hparams.yaml
                metrics.csv
    runs/
        demo/
            20260613_094641_.../
                config.yaml
                summary.json
                artifacts/
                checkpoints/best_model.ckpt
    tensorboard/
        demo/
            20260613_094641_.../
                events.out.tfevents...
                hparams.yaml
```

Your `CSVLogger` wrote `csv/mlp/version_0/` (with `metrics.csv` and `hparams.yaml`) right next to
DeepTab's own `runs/` and `tensorboard/` trees. A real tracker such as
`WandbLogger(project="churn")` would instead stream to your hosted dashboard while DeepTab keeps
owning the per-run artifact directory.

```{important}
The `logger` field is honoured **only when `experiment_trackers` is non-empty**. With an empty
`experiment_trackers` list DeepTab suppresses Lightning's logger entirely (to avoid a stray
`lightning_logs/` folder), and a `logger` you passed would be silently ignored. Pair your logger
with at least one tracker, or use the direct hand-off below.
```

To prove the point, here is the same custom logger with **no** tracker. Notice the run directory is
still created, but there is no `csv/` folder: the logger was not attached.

```python
obs_logger_only = ObservabilityConfig(
    root_dir=str(WORKDIR / "09_logger_only"),
    experiment_name="demo",
    logger=CSVLogger(save_dir=str(WORKDIR / "09_logger_only" / "csv"), name="mlp"),
)
train(obs_logger_only)
show_tree(WORKDIR / "09_logger_only", "09_logger_only/   (no csv/, logger was ignored without a tracker)")
```

```text
09_logger_only/   (no csv/, logger was ignored without a tracker)
    runs/
        demo/
            20260613_094641_.../
                config.yaml
                summary.json
                artifacts/
                checkpoints/best_model.ckpt
```

### 7b. Hand a logger straight to `fit()`

Any keyword argument `fit()` does not recognise is forwarded to `pl.Trainer`, and an explicit
`logger=` overrides DeepTab's default. This is the lightest-weight option: no `ObservabilityConfig`
at all, just your logger driving training. There is no DeepTab run directory in this mode, only
whatever your logger writes.

```python
direct = MLPClassifier(trainer_config=TRAINER, random_state=42)
with focused_output():
    direct.fit(
        X_train, y_train,
        enable_progress_bar=False,
        logger=CSVLogger(save_dir=str(WORKDIR / "10_direct_logger"), name="mlp"),
    )
show_tree(WORKDIR / "10_direct_logger", "10_direct_logger/")
```

```text
10_direct_logger/
    mlp/
        version_0/
            hparams.yaml
            metrics.csv
```

### 7c. Consume the lifecycle events in-process

If you want DeepTab's **events** (not just Lightning metrics) routed into your own system, attach
any object that exposes `info(event: str, **kwargs)`. DeepTab dispatches every lifecycle event to
it. This is the same interface the built-in `structlog` backend implements, so a test double or an
adapter to your telemetry pipeline drops in cleanly.

```{note}
This attaches to the `_event_logger` hook directly, which is a lower-level integration point than
the `ObservabilityConfig` fields above. Use it when you need the structured events inside your own
process; use `log_to_file=True` and read `lifecycle.jsonl` when a file-based hand-off is enough.
```

```python
class CollectingSink:
    """Minimal event sink: captures every lifecycle event in memory."""

    def __init__(self):
        self.events = []

    def info(self, event, **kwargs):
        self.events.append({"event": event, **kwargs})


sink = CollectingSink()
model = MLPClassifier(trainer_config=TRAINER, random_state=42)
model._event_logger = sink   # attach a custom in-process event consumer
with focused_output():
    model.fit(X_train, y_train, enable_progress_bar=False, default_root_dir=str(WORKDIR / "11_custom_sink"))

print("Captured events:")
for record in sink.events:
    print(" ", record["event"], "->", {k: v for k, v in record.items() if k != "event"})
```

```text
Captured events:
  fit.started -> {'run_id': '0f1c8c6a', 'model_class': 'MLPClassifier', 'n_samples': 640, 'n_features': 8, 'random_state': 42}
  data.created -> {'run_id': '0f1c8c6a', 'n_train': 512, 'n_val': 128, 'n_num_features': 8, 'n_cat_features': 0, 'val_size': 0.2, 'duration_min': 0.0004}
  model.created -> {'run_id': '0f1c8c6a', 'backbone': 'MLP', 'n_params': 78273, 'n_num_features': 8, 'n_cat_features': 0, 'duration_min': 0.0}
  train.started -> {'run_id': '0f1c8c6a', 'max_epochs': 5, 'batch_size': 128, 'lr': None, 'optimizer': 'Adam', 'patience': 2, 'val_size': 0.2}
  train.completed -> {'run_id': '0f1c8c6a', 'best_epoch': None, 'best_val_loss': 0.6822827458381653, 'n_epochs_run': 5, 'duration_min': 0.0051}
  fit.completed -> {'run_id': '0f1c8c6a', 'status': 'success', 'model_class': 'MLPClassifier', 'n_params': 78273, 'best_val_loss': 0.6822827458381653, 'duration_min': 0.0056}
```

Your sink received the full event stream with its structured payloads, ready to forward to whatever
backend you use. Because no `ObservabilityConfig` was attached, DeepTab created no run directory of
its own: your code is in full control.

## 8. Side-by-side: what each configuration leaves on disk

The trees below are the canonical shapes you can expect. Timestamps and ids vary per run; the
structure does not.

**No observability**: only the best-weights checkpoint:

```text
01_no_observability/
    checkpoints/
        best_model.ckpt
```

**Minimal `ObservabilityConfig`**: self-describing run directory:

```text
02_minimal/
    runs/demo/<timestamp>_<run_id>/
        config.yaml       # full estimator configuration
        summary.json      # final metrics
        artifacts/        # reserved for run artifacts
        checkpoints/
            best_model.ckpt
```

**`structured_logging=True, log_to_file=True`**: adds the event log:

```text
04_with_file/
    runs/demo/<timestamp>_<run_id>/
        config.yaml
        lifecycle.jsonl   # one JSON event per line
        summary.json
        artifacts/
        checkpoints/
            best_model.ckpt
```

**`experiment_trackers=["tensorboard"]`**: adds a TensorBoard tree:

```text
06_tensorboard/
    runs/demo/<timestamp>_<run_id>/
        config.yaml
        summary.json
        artifacts/
        checkpoints/best_model.ckpt
    tensorboard/demo/<timestamp>_<run_id>/
        events.out.tfevents...
        hparams.yaml
```

**`experiment_trackers=["mlflow"]`**: adds a local MLflow store:

```text
07_mlflow/
    runs/demo/<timestamp>_<run_id>/
        config.yaml
        summary.json
        artifacts/
        checkpoints/best_model.ckpt
    mlflow/
        backend/mlflow.db                       # run metadata (SQLite)
        artifacts/<mlflow_run_id>/artifacts/
            config.yaml
            summary.json
            best_model/...                      # logged model checkpoint
            checkpoints/best_model.ckpt
```

**`logger=...` + a tracker**: your Lightning logger sits beside DeepTab's trees:

```text
08_byo_logger/
    csv/mlp/version_0/
        hparams.yaml
        metrics.csv
    runs/demo/<timestamp>_<run_id>/...
    tensorboard/demo/<timestamp>_<run_id>/...
```

## When to use which

- **Quick experiments / notebooks:** no observability, or `verbosity=1` for a few milestone lines.
- **Reproducible runs you may revisit:** minimal `ObservabilityConfig` so every run keeps its `config.yaml` and `summary.json`.
- **Sweeps and comparisons:** `structured_logging=True, log_to_file=True, verbosity=2`, then load each `lifecycle.jsonl` into a DataFrame.
- **Dashboards:** add `experiment_trackers=["tensorboard"]` or `["mlflow"]`.
- **Existing stack:** pass your Lightning logger via `logger=` (with a tracker), hand it to `fit(logger=...)`, or attach an in-process event sink.

## Cleanup

The scratch directory is disposable. Remove it so re-running the examples starts clean (it is also
git-ignored).

```python
shutil.rmtree(WORKDIR, ignore_errors=True)
print("Removed", WORKDIR)
```

## Next steps

- [Observability (core concept)](../core_concepts/observability): the configuration reference and design notes.
- [Advanced training](advanced_training): optimizers, schedulers, callbacks, and `InferenceModel` in production.
- [Hyperparameter optimization](hpo): run sweeps whose results you can track with the tools above.

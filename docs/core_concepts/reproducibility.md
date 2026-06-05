# Reproducibility Guide

Getting the same result every time you run a training script is essential for
debugging, comparison studies, and publication. DeepTab provides layered
controls that let you pin every source of randomness from data splitting all
the way through weight initialisation and batch ordering.

---

## Platform and device support

`set_seed` is designed to work identically on **Windows, macOS, and Linux**,
and across all PyTorch compute backends:

| Backend                 | Condition                                                         | What is seeded                                         |
| ----------------------- | ----------------------------------------------------------------- | ------------------------------------------------------ |
| **CPU**                 | always                                                            | `torch.manual_seed`                                    |
| **CUDA** (NVIDIA)       | `torch.cuda.is_available()`                                       | `torch.cuda.manual_seed_all` + cuDNN determinism flags |
| **MPS** (Apple Silicon) | `torch.backends.mps.is_available()` — PyTorch ≥ 1.12, macOS 12.3+ | `torch.mps.manual_seed`                                |

All three backends can be active simultaneously; `set_seed` applies every
relevant call automatically.

---

## Layers of randomness in a training run

| Layer                     | What it controls                       | Seeded by                     |
| ------------------------- | -------------------------------------- | ----------------------------- |
| Data split                | `train_test_split` into train/val      | `random_state`                |
| DataLoader shuffle        | Batch order within each epoch          | `random_state` → `set_seed`   |
| Weight initialisation     | PyTorch `nn.Module` `reset_parameters` | `set_seed` (torch)            |
| Dropout masks             | `nn.Dropout` stochastic zeros          | `set_seed` (torch)            |
| NumPy preprocessing       | Binning, encoding helpers              | `set_seed` (numpy)            |
| Python hash randomisation | Dict/set ordering in child processes   | `set_seed` (`PYTHONHASHSEED`) |

---

## The `random_state` parameter

Every estimator constructor accepts a `random_state` integer. When set,
DeepTab calls `set_seed(random_state)` **at the start of every `fit` call**,
before `_build_model` and before `trainer.fit`. This means the full training
pipeline — data split, weight init, dropout, and DataLoader shuffling — all
derive from the same seed, regardless of the active device.

```python
from deeptab.configs import TrainerConfig
from deeptab.models import MambularRegressor

model = MambularRegressor(
    trainer_config=TrainerConfig(max_epochs=50),
    random_state=42,          # fixes ALL randomness inside fit()
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Running the same script twice produces bit-identical predictions.

---

## `set_seed` — standalone utility

Use `set_seed` when you need to seed the environment before code that lives
_outside_ an estimator call (e.g. custom data augmentation, manual tensor
operations, or experiment setup code).

```python
from deeptab import set_seed   # top-level convenience import
# or: from deeptab.core.reproducibility import set_seed

set_seed(42)

import torch
t = torch.randn(10)   # reproducible on CPU, CUDA, and MPS
```

`set_seed` seeds the following layers in order. Only the guards marked
_conditional_ skip calls on hosts where that backend is absent — no errors
are raised on CPU-only or MPS-only machines.

| Call                                        | Condition                              |
| ------------------------------------------- | -------------------------------------- |
| `random.seed(seed)`                         | always                                 |
| `os.environ["PYTHONHASHSEED"] = str(seed)`  | always (propagated to child processes) |
| `numpy.random.seed(seed)`                   | always                                 |
| `torch.manual_seed(seed)`                   | always                                 |
| `torch.cuda.manual_seed_all(seed)`          | only when CUDA is available            |
| `torch.backends.cudnn.deterministic = True` | only when CUDA is available            |
| `torch.backends.cudnn.benchmark = False`    | only when CUDA is available            |
| `torch.mps.manual_seed(seed)`               | only when MPS is available             |

> **Note on `PYTHONHASHSEED`**: writing to `os.environ` affects child
> processes (DataLoader workers, subprocesses) but has no effect on hash
> values already computed in the _current_ process. If you need
> hash-determinism from the very first import, set `PYTHONHASHSEED` in your
> shell before launching the interpreter.

### Deterministic kernels (optional)

For strict reproducibility on any accelerator, pass `deterministic=True`.
This calls `torch.use_deterministic_algorithms(True)`, which forces every
backend (CUDA, MPS, CPU) to choose a deterministic implementation.

```python
set_seed(42, deterministic=True)
```

> **Trade-off**: some operations have no deterministic kernel and will raise
> a `RuntimeError`. Only enable this when you need publication-grade
> reproducibility and are willing to accept a possible throughput reduction.

---

## `seed_context` — scoped seeding

When you want seeding to be lexically scoped to a block of code, use the
`seed_context` context manager:

```python
from deeptab import seed_context

with seed_context(42):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
```

`seed_context` calls `set_seed` on entry and applies the same per-device
guards. It does _not_ restore the previous RNG state on exit — restoring
global multi-framework RNG state across multiple backends is fragile. The
seed remains active for the rest of the process unless overridden.

---

## Confirming reproducibility at each step

The test suite in `tests/test_reproducibility.py` verifies each layer
independently. Run it to confirm reproducibility in your environment:

```bash
pytest tests/test_reproducibility.py -v
```

The tests are organised in six steps:

| Step | Test class                               | What is verified                                                        |
| ---- | ---------------------------------------- | ----------------------------------------------------------------------- |
| 1    | `TestSetSeedPrimitives`                  | `set_seed` seeds torch / numpy / python RNGs; invalid seed raises       |
| 2    | `TestSeedContext`                        | `seed_context` is equivalent to `set_seed`                              |
| 3    | `TestSameSeedSamePredictions`            | Two independent fits with same seed → identical predictions             |
| 4    | `TestDifferentSeedsDifferentPredictions` | Different seeds → different predictions (seed has real effect)          |
| 5    | `TestNoLeakageOnRefit`                   | Fresh instances + cross-instance contamination → no leakage             |
| 6    | `TestPlatformAndDeviceSeeding`           | CPU, CUDA, MPS, `PYTHONHASHSEED`, boundary values, `deterministic=True` |

Steps 6's CUDA and MPS sub-tests are automatically skipped when the
corresponding hardware is absent — the suite always passes on CPU-only hosts.

---

## Recommended workflow

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deeptab import set_seed
from deeptab.configs import MLPConfig, TrainerConfig
from deeptab.models import MLPRegressor

SEED = 42

# 1. Seed the global environment before any data preparation.
#    set_seed activates the right device guards automatically.
set_seed(SEED)

# 2. Split your dataset with the same seed
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# 3. Pass the seed to the estimator — fit() will re-apply it before training
model = MLPRegressor(
    model_config=MLPConfig(layer_sizes=[128, 64]),
    trainer_config=TrainerConfig(max_epochs=100, lr=1e-3),
    random_state=SEED,
)
model.fit(X_train, y_train)
```

> **Tip**: always pass the same integer to _both_ `train_test_split` (or your
> CV splitter) and the estimator's `random_state`. This guarantees that the
> data partition and the model initialisation are both pinned to one value you
> can record in a config file or experiment log.

---

## Known sources of non-determinism

Even with all seeds set, the following situations can still produce run-to-run
variation:

| Source                                  | When it occurs                                    | Mitigation                                                                        |
| --------------------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------- |
| Non-deterministic CUDA ops              | GPU training without `deterministic=True`         | Pass `deterministic=True` to `set_seed`                                           |
| Non-deterministic MPS ops               | MPS training without `deterministic=True`         | Pass `deterministic=True` to `set_seed`                                           |
| Multi-worker DataLoaders                | `num_workers > 0` without `worker_init_fn`        | Keep `num_workers=0` (default) or supply a `worker_init_fn` that calls `set_seed` |
| Floating-point accumulation order       | Parallel reductions on GPU/MPS                    | Use `deterministic=True`; accept small numerical differences                      |
| `PYTHONHASHSEED` in the current process | Hash values computed before `set_seed` was called | Set `PYTHONHASHSEED` in the shell before launching Python                         |
| Third-party library internals           | Some preprocessing libraries ignore seeds         | File a bug with the upstream library                                              |

---

## Cross-validation and hyperparameter search

When running `sklearn` cross-validation or DeepTab's HPO utilities, pass the
same `random_state` to both the splitter and the estimator:

```python
from sklearn.model_selection import cross_val_score, KFold

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
scores = cross_val_score(model, X, y, cv=cv)
```

Each fold receives a fresh `fit` call, which reseeds all RNG layers via
`random_state`, so fold-level reproducibility is maintained automatically
on every supported platform and device.

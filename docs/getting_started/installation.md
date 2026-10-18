# Installation

```{important}
**Requirements:** Python 3.11-3.13 | PyTorch >=2.13,<2.15 (auto-installed)
```

## Quick Install

```bash
pip install deeptab
```

This installs DeepTab with its core dependencies, including PyTorch, Lightning, and preprocessing tools. Logging, tracking, and native TabR retrieval use separate extras.

**Verify installation:**

```python
import deeptab
print(deeptab.__version__)  # e.g., "2.0.0"
```

## Supported Platforms

CI configures Linux, Windows, and macOS (Apple Silicon) across Python 3.11-3.13.
Locked tests and independent dependency endpoints cover different combinations;
neither a configured job nor a successful resolver proves that a candidate passed.
See the [Support Matrix](../developer_guide/support_matrix) for validation limits.

```{warning}
**Intel Mac (macOS x86_64) is not supported.** Upstream PyTorch no longer publishes
x86_64 macOS wheels for `torch >= 2.3` or Python 3.13. This restriction does not
apply to Apple Silicon: Python 3.13 wheels exist for the supported Torch range.

Use Apple Silicon, Linux, or Windows. The older Intel build `torch==2.2.2`
does not satisfy the current dependency range and is not a supported workaround.
```

```{warning}
**Native TabR is unsupported on Apple Silicon.** Patched Torch and the PyPI
`faiss-cpu` build load conflicting OpenMP runtimes. DeepTab rejects native TabR
construction before importing them together. Use Linux or Windows with the
`tabr` extra; alternative FAISS builds are outside the CI matrix.
```

## Optional Dependencies

Structured logging, experiment tracking, and TabR retrieval ship as optional
extras. The default installation does not install FAISS or delu. The `all`
bundle includes observability backends only, not native retrieval.

```bash
pip install 'deeptab[logs]'         # structlog (structured logging)
pip install 'deeptab[tensorboard]'  # TensorBoard tracker
pip install 'deeptab[mlflow]'       # MLflow tracker
pip install 'deeptab[tracking]'     # TensorBoard + MLflow
pip install 'deeptab[all]'          # structlog + TensorBoard + MLflow
pip install 'deeptab[tabr]'         # FAISS + delu, on Linux or Windows
pip install 'deeptab[all,tabr]'     # observability and native retrieval
```

```{note}
These backends are loaded lazily. Without the matching extra, DeepTab trains normally and raises only when you enable that specific feature. See the [Observability](../core_concepts/observability) guide for usage.
```

## Dependency Compatibility

DeepTab requires NumPy 2, pandas >=2.2.2, SciPy >=1.16, Torch >=2.13,<2.15,
Lightning >=2.6.6,<3, Accelerate >=1.15,<2, and setuptools >=83,<84. The patched
dependency floors replace older compatibility pins with reported vulnerabilities.
Do not force older versions with `--no-deps`.

TensorBoard extras require >=2.21,<3. Earlier versions can fail hyperparameter
event writing with NumPy 2 or import the removed `pkg_resources` module with
patched setuptools.

```{important}
DeepTab 2.1 no longer supports Python 3.10. Recreate your environment with Python
3.11, 3.12 or 3.13 before upgrading; an older DeepTab release is not a security
guarantee. See the [Migration Guide](migration) for upgrade guidance. Python 3.14
and newer remain outside the declared range.
```

For fresh core and logging/tracking comparisons, follow the
{ref}`dependency endpoint procedure <dependency-endpoints>`. Keep runtime pins,
interpreter/platform details and wheel origins with the results. `pip check`
detects metadata inconsistencies, not every native binary incompatibility.

## Hardware Support

DeepTab automatically detects and uses your accelerator, with no configuration
needed. It supports NVIDIA GPUs through CUDA and Apple Silicon through the MPS
backend, and falls back to the CPU when neither is present.

```{note}
Backend detection reports availability, not validation of every architecture or
operation on that device. The local dependency and width studies use CPU only;
actual CUDA/MPS contracts and native retrieval platform checks remain separate
release requirements. See the [Support Matrix](../developer_guide/support_matrix).
```

**Inspect what DeepTab can see:**

```python
from deeptab import print_hardware_info

print_hardware_info()
```

```text
DeepTab hardware report
-----------------------
Platform: Darwin (arm64), Python 3.12.9, PyTorch 2.14.1
CPU: 14 logical cores
CUDA: not available
MPS (Apple Silicon): available
Recommended accelerator: mps
```

The report covers the CPU core count, CUDA GPUs, the Apple Silicon MPS backend,
and the `accelerator` value DeepTab would pick by default.

DeepTab already selects the best available accelerator, so you usually set
nothing. For portable scripts you can make this explicit with `accelerator="auto"`,
which uses a GPU when present and falls back to the CPU otherwise:

```python
# Uses CUDA or MPS when available, otherwise CPU
model.fit(X_train, y_train, accelerator="auto", max_epochs=100)
```

```{tip}
Pin a specific backend (`"gpu"`, `"cuda"`, `"mps"`, or `"cpu"`) only to force
one. Unlike `"auto"`, those raise an error if the device is not present.
```

### NVIDIA GPUs (CUDA)

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

```{warning}
If you have a GPU but CUDA isn't detected, install PyTorch with CUDA support first:
```

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install deeptab
```

See [PyTorch installation guide](https://pytorch.org/get-started/locally/) for your CUDA version.

**Multiple GPUs:**

```bash
export CUDA_VISIBLE_DEVICES=0,1  # Use specific GPUs
```

### Apple Silicon (MPS)

On M-series Macs, DeepTab uses the Metal Performance Shaders (MPS) backend.
MPS ships with the standard PyTorch build, so no extra install step is needed.

```python
import torch

print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

```{note}
`is_available()` reports `True` only on macOS 12.3+ with Apple Silicon and a
PyTorch build that includes MPS. When it returns `False` but `is_built()` is
`True`, the backend is present but the OS or hardware does not support it, and
DeepTab runs on the CPU.
```

## Development Installation

For contributing or using unreleased features:

```bash
git clone https://github.com/OpenTabular/DeepTab.git
cd DeepTab
pip install -e .
```

```{note}
DeepTab uses Poetry for development. Install with `poetry install` to get dev tools (pytest, ruff, pyright). Documentation dependencies require Python 3.12 or 3.13; core development also supports 3.11. See the [Contributing guide](../developer_guide/contributing) for details.
```

## Optional: Mamba CUDA Kernels

For 20-30% faster Mamba models, install optimized CUDA kernels:

```bash
pip install mamba-ssm
```

```{important}
**Requirements:** NVIDIA GPU (compute capability ≥7.0) | CUDA 11.6+ | C++ compiler

If installation fails, DeepTab automatically falls back to the default implementation. This only affects Mamba-based models.
```

## Quick Troubleshooting

**CUDA out of memory?** Reduce batch size:

```python
from deeptab.configs import TrainerConfig
model = FTTransformerClassifier(
    trainer_config=TrainerConfig(batch_size=64)
)
```

**Training slow?** Check which accelerator DeepTab detected:

```python
from deeptab import print_hardware_info

print_hardware_info()  # "Recommended accelerator" should not be cpu
```

**Module not found?** Verify correct environment:

```bash
which python
pip list | grep deeptab
```

## Next Steps

- [Quickstart](quickstart): Train your first model in 5 minutes
- [FAQ](faq): Common questions and solutions

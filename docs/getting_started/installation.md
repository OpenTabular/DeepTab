# Installation

```{important}
**Requirements:** Python 3.10+ | PyTorch 2.2+ (auto-installed)
```

## Quick Install

```bash
pip install deeptab
```

This installs DeepTab with all dependencies including PyTorch, Lightning, and preprocessing tools.

**Verify installation:**

```python
import deeptab
print(deeptab.__version__)  # e.g., "2.0.0"
```

## Optional Dependencies

The default install keeps the core small. Observability backends (structured logging and experiment trackers) ship as extras, so install only what you need:

```bash
pip install 'deeptab[logs]'         # structlog (structured logging)
pip install 'deeptab[tensorboard]'  # TensorBoard tracker
pip install 'deeptab[mlflow]'       # MLflow tracker
pip install 'deeptab[tracking]'     # TensorBoard + MLflow
pip install 'deeptab[all]'          # structlog + TensorBoard + MLflow
```

```{note}
These backends are loaded lazily. Without the matching extra, DeepTab trains normally and raises only when you enable that specific feature. See the [Observability](../core_concepts/observability) guide for usage.
```

## Hardware Support

DeepTab automatically detects and uses your accelerator, with no configuration
needed. It supports NVIDIA GPUs through CUDA and Apple Silicon through the MPS
backend, and falls back to the CPU when neither is present.

**Inspect what DeepTab can see:**

```python
from deeptab import print_hardware_info

print_hardware_info()
```

```text
DeepTab hardware report
-----------------------
Platform: Darwin (arm64), Python 3.11.8, PyTorch 2.2.0
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
DeepTab uses Poetry for development. Install with `poetry install` to get dev tools (pytest, ruff, pyright). See the [Contributing guide](../developer_guide/contributing) for details.
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

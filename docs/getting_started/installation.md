# Installation

```{important}
**Requirements:** Python 3.10+ | PyTorch 2.0+ (auto-installed)
**Installation time:** ~2 minutes
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

## GPU Support

DeepTab automatically detects and uses your GPU, with no configuration needed.

**Verify GPU:**

```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
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

**Training slow?** Check GPU is being used:

```python
import torch
assert torch.cuda.is_available(), "GPU not detected"
```

**Module not found?** Verify correct environment:

```bash
which python
pip list | grep deeptab
```

## Next Steps

- [Quickstart](quickstart): Train your first model in 5 minutes
- [FAQ](faq): Common questions and solutions

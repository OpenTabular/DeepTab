# Installation

This guide covers installing DeepTab in different environments and verifying the setup.

## Requirements

- **Python**: 3.10, 3.11, 3.12, 3.13, or 3.14
- **pip** or **poetry** for package management
- **PyTorch**: Version 2.0 or later (installed automatically)

See the [support matrix](../developer_guide/support_matrix) for tested version combinations.

## Install from PyPI

The simplest way to get started:

```bash
pip install deeptab
```

This installs DeepTab along with all required dependencies:

- PyTorch (CPU or CUDA, depending on your system)
- PyTorch Lightning (training framework)
- pretab (preprocessing library)
- scikit-learn, pandas, numpy (data utilities)

### Verify installation

After installing, verify that DeepTab is available:

```python
import deeptab
print(deeptab.__version__)
```

You should see the version number (e.g., `2.0.0`).

### Test with a simple model

Run a quick smoke test to ensure everything works:

```python
from deeptab.models import MambularClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=5, random_state=42)
model = MambularClassifier()
model.fit(X, y, max_epochs=5)
print("Installation verified!")
```

## Install from source

For development or to use unreleased features:

### Clone the repository

```bash
git clone https://github.com/OpenTabular/DeepTab.git
cd DeepTab
```

### Install with Poetry

DeepTab uses Poetry for dependency management:

```bash
# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Install DeepTab in editable mode
poetry install
```

This creates a virtual environment and installs all dependencies, including dev tools (pytest, ruff, pyright).

### Install with pip

If you prefer pip:

```bash
pip install -e .
```

This installs DeepTab in editable mode, so changes to the source code are immediately reflected.

### Run tests

Verify the development installation:

```bash
# With Poetry
poetry run pytest

# With pip
pytest
```

See the [Contributing guide](../developer_guide/contributing) for the full development setup.

## GPU support

DeepTab will automatically use your GPU if PyTorch detects one. No additional configuration is needed.

### Check GPU availability

Verify that PyTorch can see your GPU:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

If CUDA is available, DeepTab will use it automatically during training.

### Install specific CUDA version

If you need a specific CUDA version, install PyTorch manually first, then install DeepTab:

```bash
# Example: CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install DeepTab
pip install deeptab
```

### CUDA version compatibility

Check the [PyTorch installation page](https://pytorch.org/get-started/locally/) for supported CUDA versions. Common options:

| CUDA version | PyTorch index URL                        |
| ------------ | ---------------------------------------- |
| 11.8         | `https://download.pytorch.org/whl/cu118` |
| 12.1         | `https://download.pytorch.org/whl/cu121` |
| CPU only     | `https://download.pytorch.org/whl/cpu`   |

### Multiple GPUs

DeepTab uses the first available GPU by default. To use a specific GPU:

```bash
# Set before importing PyTorch
export CUDA_VISIBLE_DEVICES=1
python your_script.py
```

Or in Python:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from deeptab.models import MambularClassifier
```

For multi-GPU training, see the Lightning documentation on distributed training.

## Optional: Mamba CUDA kernels

The default Mamba implementation in DeepTab runs on any hardware (CPU or GPU). If you have a compatible NVIDIA GPU and want the optimized CUDA kernels from the original Mamba paper:

```bash
pip install mamba-ssm
```

### Requirements for mamba-ssm

- NVIDIA GPU with compute capability 7.0 or higher (Volta, Turing, Ampere, Ada, Hopper)
- CUDA 11.6 or later
- Compatible C++ compiler

If installation fails, DeepTab will fall back to the default implementation automatically.

### Verify Mamba kernels

Check which Mamba implementation is being used:

```python
from deeptab.architectures import MambularArch

# If mamba-ssm is installed and working, you'll see a message
# about using optimized kernels when instantiating the model
```

This is optional and only affects Mamba-based models (`Mambular`, `MambaTab`, `MambAttention`). Other models are unaffected.

## Platform-specific notes

### macOS (Apple Silicon)

PyTorch has native support for Apple Silicon (M1/M2/M3):

```bash
pip install deeptab
```

DeepTab will use the Metal Performance Shaders (MPS) backend automatically. Verify:

```python
import torch

print(f"MPS available: {torch.backends.mps.is_available()}")
```

Note: Some operations may fall back to CPU on MPS. This is a PyTorch limitation, not specific to DeepTab.

### Windows

Install from PyPI as usual:

```bash
pip install deeptab
```

For GPU support on Windows, ensure you have:

- NVIDIA GPU with recent drivers
- CUDA Toolkit (if using CUDA-enabled PyTorch)

### Linux

DeepTab works on all major Linux distributions. For GPU support:

```bash
# Ubuntu/Debian
sudo apt-get install nvidia-cuda-toolkit

# Then install DeepTab
pip install deeptab
```

## Virtual environments

We recommend using a virtual environment to avoid dependency conflicts.

### Using venv

```bash
python -m venv deeptab-env
source deeptab-env/bin/activate  # On Windows: deeptab-env\Scripts\activate
pip install deeptab
```

### Using conda

```bash
conda create -n deeptab python=3.11
conda activate deeptab
pip install deeptab
```

### Using Poetry

```bash
poetry new my-project
cd my-project
poetry add deeptab
poetry shell
```

## Troubleshooting

### ImportError: No module named 'deeptab'

Ensure you've activated the correct virtual environment:

```bash
which python  # Should point to your venv
pip list | grep deeptab  # Should show the installed version
```

### CUDA out of memory

Reduce batch size in `TrainerConfig`:

```python
from deeptab.configs import TrainerConfig
from deeptab.models import MambularClassifier

model = MambularClassifier(
    trainer_config=TrainerConfig(batch_size=64)  # Smaller batch size
)
```

### Slow training on CPU

Ensure PyTorch is using GPU:

```python
import torch
assert torch.cuda.is_available(), "CUDA not available"
```

If CUDA is not available and you have a GPU, reinstall PyTorch with CUDA support.

### mamba-ssm installation fails

This is optional. DeepTab works fine without it. If you still want to install:

1. Ensure you have a compatible CUDA version
2. Install with verbose output: `pip install -v mamba-ssm`
3. Check the error message for missing dependencies (usually a C++ compiler or CUDA toolkit)

If it continues to fail, you can skip this step—DeepTab will use the default Mamba implementation.

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade deeptab
```

Check the [changelog](../../CHANGELOG.md) for breaking changes when upgrading across major versions.

## Uninstalling

To remove DeepTab:

```bash
pip uninstall deeptab
```

This removes DeepTab but leaves PyTorch and other dependencies installed. To remove everything:

```bash
pip uninstall deeptab torch torchvision lightning pretab
```

## Next steps

- **[Quickstart](quickstart)** — Run your first model
- **[FAQ](faq)** — Common questions and solutions
- **[Key Concepts](../key_concepts)** — Understand the API before diving in

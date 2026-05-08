# Installation

## Prerequisites

- Python 3.10 – 3.14
- [pip](https://pip.pypa.io/) or [poetry](https://python-poetry.org/)
- A working PyTorch installation (CPU or CUDA). See the [support matrix](developer_guide/support_matrix) for tested versions.

## Install from PyPI

```bash
pip install deeptab
```

Verify the installation:

```python
import deeptab
print(deeptab.__version__)
```

## Install from source

For development or to use unreleased features:

```bash
git clone https://github.com/OpenTabular/DeepTab
cd DeepTab
poetry install
```

See [Contributing](developer_guide/contributing) for the full development setup.

## Optional: Mamba CUDA kernels

The default DeepTab Mamba implementation runs on any hardware. If you want the original optimised CUDA kernels (requires a compatible GPU and CUDA toolkit):

```bash
pip install mamba-ssm
```

## GPU setup

If you need a specific PyTorch + CUDA combination, install PyTorch first following the [official selector](https://pytorch.org/get-started/locally/), then install DeepTab:

```bash
# Example: CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install deeptab
```

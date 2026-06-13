# Model Efficiency Benchmarking Tutorial

<div style="display: flex; gap: 10px; margin-bottom: 20px;">
  <a href="https://colab.research.google.com/github/OpenTabular/DeepTab/blob/main/docs/tutorials/notebooks/model_efficiency.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://github.com/OpenTabular/DeepTab/blob/main/docs/tutorials/notebooks/model_efficiency.ipynb" target="_blank">
    <img src="https://img.shields.io/badge/View%20on-GitHub-181717?logo=github&logoColor=white" alt="View on GitHub"/>
  </a>
</div>

This tutorial shows how to benchmark DeepTab model families under controlled synthetic workloads. It focuses on forward-pass latency, peak device memory, and parameter count so researchers and developers can decide which architectures are practical before running full training experiments.

```{note}
The notebook linked above is generated from this same tutorial content. Use the markdown page to understand the protocol, and use the notebook when you want to run or modify the benchmark cells.
```

## What You Will Learn

- How to isolate architecture cost from preprocessing and trainer overhead.
- How feature count, depth, and batch size affect different model families.
- How to report efficiency results without implying an accuracy ranking.
- How to connect runtime measurements back to model selection.

```{important}
Efficiency numbers are hardware-specific. Report the device, CUDA version, PyTorch version, DeepTab commit, dtype, feature schema, batch size, warmup count, and repeat count whenever you share results.
```

## Benchmark Scope

The cells below profile low-level architecture classes directly. This isolates the model body and avoids estimator-level preprocessing, Lightning training, validation, checkpointing, and data-loading overhead.

Use this tutorial for architecture screening. For end-to-end claims, add a second benchmark around the sklearn-style estimator workflow: `fit`, `predict`, and `evaluate`.

## Setup

```python
import platform
import time
from dataclasses import dataclass

import pandas as pd
import torch

from deeptab.architectures import (
    FTTransformer,
    MLP,
    MambAttention,
    MambaTab,
    Mambular,
    ResNet,
    TabulaRNN,
)
from deeptab.configs import (
    FTTransformerConfig,
    MLPConfig,
    MambAttentionConfig,
    MambaTabConfig,
    MambularConfig,
    ResNetConfig,
    TabulaRNNConfig,
)

print({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
})
```

## Synthetic Feature Schema

The helper below creates a controlled half-numerical, half-categorical schema. Keeping the schema synthetic makes it easier to isolate architecture scaling. It does not replace real-dataset benchmarking.

```python
@dataclass(frozen=True)
class BenchmarkSpec:
    n_features: int
    batch_size: int = 256
    n_layers: int = 4
    repeats: int = 50
    warmup: int = 10
    n_categories: int = 10


def make_feature_information(n_features: int, n_categories: int = 10):
    """Create a half-numerical, half-categorical synthetic feature schema."""
    n_num = n_features // 2
    n_cat = n_features - n_num

    num_info = {
        f"num_{i}": {
            "preprocessing": "standard",
            "dimension": 1,
            "categories": None,
        }
        for i in range(n_num)
    }
    cat_info = {
        f"cat_{i}": {
            "preprocessing": "int",
            "dimension": 1,
            "categories": n_categories,
        }
        for i in range(n_cat)
    }
    return num_info, cat_info, {}


def make_batch(feature_information, batch_size: int, device: torch.device):
    num_info, cat_info, _ = feature_information
    num_features = [
        torch.randn(batch_size, info["dimension"], device=device)
        for info in num_info.values()
    ]
    cat_features = [
        torch.randint(
            low=0,
            high=info["categories"],
            size=(batch_size, info["dimension"]),
            device=device,
        )
        for info in cat_info.values()
    ]
    return num_features, cat_features, []


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

```{tip}
Start with synthetic sweeps to understand scaling, then repeat the benchmark using the actual feature schema and preprocessing from your target dataset.
```

## Model Factories

The factory function keeps model construction consistent across sweeps. The configs are intentionally simple: they are not tuned for accuracy.

```python
def model_factories(n_layers: int):
    """Return comparable default-ish architecture configs for profiling."""
    return {
        "Mambular": (
            Mambular,
            MambularConfig(d_model=64, n_layers=n_layers),
        ),
        "MambaTab": (
            MambaTab,
            MambaTabConfig(d_model=64, n_layers=max(1, min(n_layers, 4))),
        ),
        "MambAttention": (
            MambAttention,
            MambAttentionConfig(d_model=64, n_layers=n_layers, n_heads=8),
        ),
        "FTTransformer": (
            FTTransformer,
            FTTransformerConfig(d_model=128, n_layers=n_layers, n_heads=8),
        ),
        "TabulaRNN": (
            TabulaRNN,
            TabulaRNNConfig(d_model=128, n_layers=n_layers),
        ),
        "MLP": (
            MLP,
            MLPConfig(layer_sizes=[512, 256, 128, 32], use_embeddings=True, d_model=64),
        ),
        "ResNet": (
            ResNet,
            ResNetConfig(layer_sizes=[512, 256, 64], use_embeddings=True, d_model=64),
        ),
    }
```

## Forward Benchmark Runner

This runner uses `model.eval()` and `torch.inference_mode()` because it measures inference-style forward cost. CUDA synchronization is required for meaningful GPU timing.

```python
def benchmark_forward(model: torch.nn.Module, batch, repeats: int = 50, warmup: int = 10):
    model.eval()
    device = next(model.parameters()).device

    with torch.inference_mode():
        for _ in range(warmup):
            model(*batch)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()
        for _ in range(repeats):
            model(*batch)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        else:
            memory_mb = None

    latency_ms = (time.perf_counter() - start) * 1000 / repeats
    return latency_ms, memory_mb


def run_benchmark(spec: BenchmarkSpec, selected_models=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_information = make_feature_information(spec.n_features, spec.n_categories)
    batch = make_batch(feature_information, spec.batch_size, device)
    factories = model_factories(spec.n_layers)

    if selected_models is not None:
        factories = {name: factories[name] for name in selected_models}

    rows = []
    for name, (model_cls, config) in factories.items():
        model = model_cls(
            feature_information=feature_information,
            num_classes=1,
            config=config,
        ).to(device)
        latency_ms, memory_mb = benchmark_forward(
            model,
            batch,
            repeats=spec.repeats,
            warmup=spec.warmup,
        )
        rows.append({
            "model": name,
            "n_features": spec.n_features,
            "batch_size": spec.batch_size,
            "n_layers": spec.n_layers,
            "latency_ms": latency_ms,
            "peak_memory_mb": memory_mb,
            "parameters": count_parameters(model),
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)
```

```{warning}
Forward-only inference timing does not include backward pass, optimizer state, data loading, validation, early stopping, or hyperparameter search. Use it as an architecture-screening signal, not as a full training-cost claim.
```

## Feature-Count Sweep

This sweep is most relevant when deciding whether feature attention is affordable for wide tables. Keep batch size and depth fixed while increasing the number of synthetic feature tokens.

```python
feature_sweep_results = []
for n_features in [10, 20, 40, 80, 160, 320]:
    spec = BenchmarkSpec(n_features=n_features, batch_size=128, n_layers=4, repeats=20, warmup=5)
    feature_sweep_results.append(run_benchmark(spec))

feature_sweep = pd.concat(feature_sweep_results, ignore_index=True)
feature_sweep
```

Interpret this sweep together with the architecture. Transformer-style feature attention becomes more expensive as feature-token count grows, while dense and state-space paths usually avoid explicit full attention maps.

## Depth Sweep

This sweep is most relevant when choosing `n_layers`. It keeps the synthetic feature schema fixed while changing model depth for sequence and attention families.

```python
depth_sweep_results = []
for n_layers in [1, 2, 4, 8, 12]:
    spec = BenchmarkSpec(n_features=64, batch_size=128, n_layers=n_layers, repeats=20, warmup=5)
    depth_sweep_results.append(
        run_benchmark(
            spec,
            selected_models=["Mambular", "MambaTab", "MambAttention", "FTTransformer", "TabulaRNN"],
        )
    )

depth_sweep = pd.concat(depth_sweep_results, ignore_index=True)
depth_sweep
```

Depth affects more than latency. It also changes activation memory during training and often changes the amount of regularization needed.

## Batch-Size Sweep

This sweep is most relevant for GPU utilization and memory planning. Larger batches can improve throughput but may hide latency problems for online inference.

```python
batch_sweep_results = []
for batch_size in [32, 64, 128, 256, 512]:
    spec = BenchmarkSpec(n_features=64, batch_size=batch_size, n_layers=4, repeats=20, warmup=5)
    batch_sweep_results.append(run_benchmark(spec))

batch_sweep = pd.concat(batch_sweep_results, ignore_index=True)
batch_sweep
```

```{important}
For SAINT-style row attention or retrieval-style models, batch size can change the effective algorithmic cost. Do not report efficiency results without the batch size.
```

## Reporting Results

Report benchmark results with enough context that another researcher can reproduce the workload.

| Field | What to record |
| ----- | -------------- |
| Hardware | CPU/GPU model, GPU memory, CUDA version |
| Software | DeepTab version or commit, PyTorch version, Python version |
| Workload | Number of rows if applicable, feature count, categorical cardinalities |
| Config | Model config, preprocessing config, trainer config if training is measured |
| Measurement | Forward-only, training step, epoch, or full fit |
| Runtime settings | Batch size, dtype, warmup count, repeat count |
| Results | Latency, peak memory, parameter count, throughput if useful |

```{tip}
If efficiency is part of a research claim, report accuracy or validation loss separately. A faster model is not automatically a better model.
```

## Next Steps

- [Model efficiency guide](../model_zoo/efficiency)
- [Model comparison](../model_zoo/comparison_tables)
- [Recommended configs](../model_zoo/recommended_configs)

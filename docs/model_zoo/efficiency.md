# Model Efficiency & Benchmarking

This page explains where efficiency analysis belongs in DeepTab and how to use it when selecting models. It complements the architectural complexity table in [Model Comparison](comparison_tables) with a practical benchmarking protocol.

```{important}
Efficiency results are hardware- and workload-dependent. Use them to compare candidate models under the same feature schema, batch size, preprocessing, dtype, and device. Do not treat synthetic timing results as an accuracy benchmark or as a universal ranking.
```

## Where This Applies

Efficiency analysis is most useful when researchers or developers need to choose a model under runtime constraints.

| Decision                 | Why efficiency matters                                                                                               | Where to use it                              |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| Model selection          | Attention, state-space, dense, tree-style, and retrieval models scale differently with feature tokens and batch size | Model Zoo comparison and recommended configs |
| Experiment planning      | Search budget, number of seeds, and architecture grid size depend on training cost                                   | Research protocol and benchmark reports      |
| Production screening     | Memory use and inference latency can rule out otherwise accurate models                                              | Deployment and low-latency model choice      |
| Architecture development | New blocks should be compared against strong baselines at controlled feature counts and depths                       | Developer benchmarking                       |

It is less appropriate for the API reference. The API pages should document classes, signatures, and methods. Efficiency belongs in the Model Zoo because it helps users decide which architecture to try before they write code.

## What to Measure

For tabular deep learning, the most informative efficiency variables are usually:

| Variable                 | Why it matters                                                                                                                                                 |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Feature-token count      | Transformer-style feature attention grows roughly quadratically in the number of tokens, while Mamba/RNN/dense paths usually avoid full feature-attention maps |
| Batch size               | Larger batches improve accelerator utilization, but SAINT-style row attention and activation memory can grow quickly                                           |
| Hidden width             | Dense projections often scale with width squared; increasing `d_model` affects attention, Mamba blocks, heads, and embeddings                                  |
| Depth                    | More layers increase activation memory and forward/backward time; tree depth in differentiable tree models can be especially expensive                         |
| Categorical cardinality  | Embedding-table size depends on category counts, not just number of columns                                                                                    |
| Retrieval candidate size | TabR-style models add candidate encoding, nearest-neighbor search, and context-mixing costs                                                                    |

```{tip}
For model selection, measure forward latency, peak device memory, and parameter count. For training-budget planning, also measure one or more full training epochs because backward pass, optimizer state, data loading, and validation can change the ranking.
```

## Expected Scaling Patterns

These are practical expectations from the architecture, not measured leaderboard results.

| Family                 | Main cost driver                                           | Practical implication                                                           |
| ---------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------- |
| MLP, ResNet            | Dense layer widths                                         | Fast baselines; good first checks for latency-sensitive workflows               |
| TabM                   | Dense layer widths plus active ensemble outputs            | Strong ensemble-like baseline with better cost than many independent models     |
| Mambular, MambaTab     | Feature sequence length, `d_model`, number of Mamba layers | Attractive when feature-token count is high and full attention is expensive     |
| FTTransformer, AutoInt | Feature-token attention maps                               | Watch memory when many columns, numerical bins, or embedding tokens are present |
| TabTransformer         | Categorical-token attention                                | Most relevant when categorical features dominate                                |
| SAINT                  | Column attention plus row attention within each batch      | Batch size is part of the architecture cost, not just a loader setting          |
| NODE, ENODE, NDTF      | Number of trees, depth, and soft path/leaf evaluations     | Tree depth is a compute knob as well as a modeling knob                         |
| TabR                   | Candidate encoding/search and context size                 | Report candidate-pool construction and retrieval settings with results          |

## Benchmark Protocol

Use a controlled protocol when reporting efficiency numbers.

1. Fix the hardware, PyTorch version, DeepTab version, dtype, and device.
2. Use the same feature schema across models unless the research question is schema-specific.
3. Run warmup iterations before timing GPU code.
4. Use `torch.inference_mode()` and `model.eval()` for inference benchmarks.
5. Synchronize CUDA before and after timed regions.
6. Reset and report peak memory with `torch.cuda.reset_peak_memory_stats()` and `torch.cuda.max_memory_allocated()`.
7. Report median or mean over repeated runs, not a single pass.
8. Separate forward-only, training-step, and full-fit measurements.

```{warning}
Synthetic forward-pass benchmarks are useful for isolating architecture cost, but they do not include preprocessing, data loading, validation, early stopping, checkpointing, or hyperparameter search. For end-to-end claims, benchmark the sklearn-style estimator workflow too.
```

## Using the Efficiency Notebook

The runnable version lives in the [Model Efficiency Benchmarking tutorial](../tutorials/model_efficiency), with the notebook stored at `docs/tutorials/notebooks/model_efficiency.ipynb` ([open on GitHub](https://github.com/OpenTabular/DeepTab/blob/main/docs/tutorials/notebooks/model_efficiency.ipynb)). The notebook is stored with the tutorial notebooks so executable examples live in one place.

Use the notebook when you want to stress-test model families across:

- increasing feature counts,
- increasing model depth,
- fixed feature schemas with different architecture families,
- GPU memory and latency constraints.

The notebook should be run on the same machine and environment used for the reported results. If you publish or share benchmark numbers, include the notebook commit, hardware, CUDA version, PyTorch version, batch size, feature count, model configs, and whether the numbers are forward-only or full-training.

## Minimal Forward Benchmark Pattern

The low-level architecture classes are useful for isolating model-body cost because they avoid estimator-level preprocessing and Lightning trainer overhead.

```python
import time

import torch

from deeptab.architectures import FTTransformer, Mambular
from deeptab.configs import FTTransformerConfig, MambularConfig


def make_feature_information(n_features: int):
    n_num = n_features // 2
    n_cat = n_features - n_num

    num_info = {
        f"num_{i}": {"preprocessing": "standard", "dimension": 1, "categories": None}
        for i in range(n_num)
    }
    cat_info = {
        f"cat_{i}": {"preprocessing": "int", "dimension": 1, "categories": 10}
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
        torch.randint(0, info["categories"], (batch_size, info["dimension"]), device=device)
        for info in cat_info.values()
    ]
    return num_features, cat_features, []


def benchmark_forward(model, batch, repeats: int = 50, warmup: int = 10):
    model.eval()
    device = next(model.parameters()).device

    with torch.inference_mode():
        for _ in range(warmup):
            model(*batch)

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()
        for _ in range(repeats):
            model(*batch)

        if device.type == "cuda":
            torch.cuda.synchronize()
            memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        else:
            memory_mb = None

    latency_ms = (time.perf_counter() - start) * 1000 / repeats
    return latency_ms, memory_mb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_information = make_feature_information(n_features=64)
batch = make_batch(feature_information, batch_size=256, device=device)

models = {
    "Mambular": Mambular(
        feature_information=feature_information,
        config=MambularConfig(d_model=64, n_layers=4),
    ).to(device),
    "FTTransformer": FTTransformer(
        feature_information=feature_information,
        config=FTTransformerConfig(d_model=128, n_layers=4, n_heads=8),
    ).to(device),
}

for name, model in models.items():
    latency_ms, memory_mb = benchmark_forward(model, batch)
    print(name, {"latency_ms": latency_ms, "memory_mb": memory_mb})
```

## Reporting Template

Use this compact template in experiment notes or pull requests:

| Field                | Value                                                          |
| -------------------- | -------------------------------------------------------------- |
| Hardware             | GPU/CPU model, memory, CUDA version                            |
| Software             | DeepTab commit/version, PyTorch version, Python version        |
| Workload             | Task, number of rows, feature count, categorical cardinalities |
| Config               | Model config, preprocessing config, trainer config             |
| Measurement          | Forward-only, train-step, epoch, or full fit                   |
| Batch size and dtype | Example: `batch_size=256`, `float32`                           |
| Repeats              | Warmup count and measured repeats                              |
| Results              | Latency, peak memory, parameter count, optional throughput     |

## References

- Gu, A., & Dao, T. (2024). _Mamba: Linear-Time Sequence Modeling with Selective State Spaces_. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). _Revisiting Deep Learning Models for Tabular Data_. NeurIPS 2021. [arXiv:2106.11959](https://arxiv.org/abs/2106.11959)
- Thielmann, A. F., & Samiee, S. (2024). _On the Efficiency of NLP-Inspired Methods for Tabular Deep Learning_. [arXiv:2411.17207](https://arxiv.org/abs/2411.17207)

## See Also

- [Model Comparison](comparison_tables): Architecture-level complexity and model selection tables
- [Recommended Configs](recommended_configs): Hyperparameter and reporting guidance
- [Model Efficiency Benchmarking tutorial](../tutorials/model_efficiency): Runnable benchmarking workflow

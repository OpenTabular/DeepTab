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

(numerical-width-study)=

## Numerical Width Study

This small end-to-end study compares requested preprocessing widths, not model
architectures. It was measured on 2026-10-15 at development revision `c68656f`
on Apple Silicon macOS 27.0.1, using CPU and one Torch thread. The candidate's
package metadata still reads `2.0.0`; these are not measurements of the
historical `v2.0.0` release or certification of a published 2.1 artifact.

The environment used Python 3.13.15, PyTorch 2.9.1, Lightning 2.6.6, NumPy 2.5.4,
pandas 2.3.3, scikit-learn 1.9.1 and PreTab 1.0.0. Each fresh-process run used an
MLP with hidden widths `[32, 16]`, zero dropout, batch size 64, learning rate
`0.001`, no scheduler and a 20-epoch budget. All runs completed 20 epochs.

The bundled sklearn breast-cancer and diabetes datasets each gained one
deterministic categorical feature. Seeds 42, 101 and 202 produce identical
splits across widths: 341/114/114 train/validation/test rows for classification
and 264/89/89 for regression. Classification uses stratified splits and
target-aware CART PLE. Regression uses uniform B-splines, with targets centered
and scaled using training statistics only; reported scores use the original
target scale. The categorical transform is integer encoding in both tasks.

### Held-Out Quality

Values are the mean and sample standard deviation across three seeds, rounded
once from full-precision measurements. Effective width is per numerical
feature; the categorical column retains width 1.

| Classification Request | Effective Width | Accuracy, Mean (SD) | AUROC, Mean (SD) |
| ---------------------- | --------------- | ------------------- | ---------------- |
| 7                      | 7               | 0.9708 (0.0051)     | 0.9960 (0.0025)  |
| 15                     | 15              | 0.9766 (0.0134)     | 0.9969 (0.0020)  |
| 30                     | 30              | 0.9795 (0.0183)     | 0.9969 (0.0028)  |
| 64                     | 64              | 0.9766 (0.0051)     | 0.9959 (0.0033)  |

| Regression Request | Effective Width | RMSE, Mean (SD)  | R2, Mean (SD)   |
| ------------------ | --------------- | ---------------- | --------------- |
| 7                  | 7               | 59.4150 (8.6235) | 0.3908 (0.1208) |
| 15                 | 15              | 56.5717 (3.1065) | 0.4462 (0.0577) |
| 30                 | 30              | 60.8712 (4.7937) | 0.3597 (0.0747) |
| 64                 | 50              | 62.4998 (5.1062) | 0.3260 (0.0656) |

### Runtime and Memory

The following are means across the same three seeds. Preprocessing time includes
fitting and train/validation transformations. Training time spans Lightning's
train-start/train-end callbacks; full fit also includes setup and orchestration,
so it is not simply the sum of the other columns. Peak RSS is the fresh process's
high-water mark through test inference, including imports and preprocessing but
excluding artifact reload. It is neither incremental model memory nor GPU memory.

| Task           | Request | Preprocessing (s) | Training (s) | Full Fit (s) | Peak RSS (MiB) |
| -------------- | ------- | ----------------- | ------------ | ------------ | -------------- |
| Classification | 7       | 0.0616            | 0.3775       | 0.5465       | 416.1          |
| Classification | 15      | 0.0622            | 0.3723       | 0.5429       | 418.0          |
| Classification | 30      | 0.0628            | 0.3754       | 0.5527       | 419.2          |
| Classification | 64      | 0.0709            | 0.3774       | 0.5827       | 432.3          |
| Regression     | 7       | 0.0237            | 0.1563       | 0.2401       | 413.3          |
| Regression     | 15      | 0.0261            | 0.1553       | 0.2425       | 413.7          |
| Regression     | 30      | 0.0298            | 0.1523       | 0.2539       | 414.0          |
| Regression     | 64      | 0.0355            | 0.1585       | 0.2822       | 415.9          |

Every case checked fitted and transformed widths, finite outputs and exact CPU
save/load predictions. The {download}`raw measurements <../tutorials/examples/numerical_widths/results.json>`
retain per-seed quality, timing, parameter counts, effective widths and split
hashes. The {download}`runner <../tutorials/examples/numerical_widths/numerical_widths.py>` reproduces the
24-case design on macOS or Linux with compatible dependencies. Set `TMPDIR` to
a writable location; for example, from the downloaded runner's directory:

```bash
TMPDIR="$PWD" python numerical_widths.py --epochs 20 --output results-rerun.json
```

```{important}
The default requested width remains 7. Width 30 had the highest mean accuracy
on this classifier task, while 15 had the highest mean regression R2. Three
seeds on two datasets do not establish a universal accuracy winner. Timing and
process RSS depend on the environment; do not pool these measurements with
older runs or use them as native retrieval, accelerator or security evidence.
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

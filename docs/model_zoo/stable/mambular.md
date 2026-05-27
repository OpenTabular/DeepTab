# Mambular

**Stacked Mamba State Space Model for tabular data.** DeepTab's flagship architecture combining efficient sequence modeling with strong empirical performance.

```{tip}
**Quick verdict:** Best general-purpose model. Strong performance across tasks with linear complexity. Recommended starting point for most applications.
```

## Architecture Overview

**Core mechanism:** Selective state space models (SSMs) with data-dependent state transitions  
**Complexity:** O(n·d) time, O(d) space per layer  
**Inductive bias:** Sequential feature processing with long-range dependencies

### Key Components

1. **Feature embedding:** Projects numerical and categorical features to d_model dimensions
2. **Mamba blocks (×N):** Selective SSM layers with residual connections
3. **Output head:** Task-specific projection (classification/regression/LSS)

**Architecture diagram:**

```
Input (mixed types) → Embedding → Mamba₁ → ... → MambaₙAcquire → Head → Output
                                     ↓ residual ↓          ↓
```

```{note}
**Selective mechanism:** Unlike traditional SSMs with fixed state transitions, Mamba uses input-dependent parameters, allowing adaptive processing based on feature importance.
```

## When to Use

### Recommended For

✅ **General-purpose modeling** — No specific data requirements  
✅ **Large datasets (>10K samples)** — Scales efficiently  
✅ **Training time constraints** — Faster than Transformers  
✅ **Production deployments** — Linear inference complexity

### Consider Alternatives When

❌ **Dataset <1K samples** → [MambaTab](mambatab) (lighter) or [TabM](tabm) (ensemble)  
❌ **Maximum interpretability needed** → [NODE](node) or [NDTF](ndtf) (tree-based)  
❌ **Extremely limited compute** → [MLP](mlp) or [ResNet](resnet) (simpler)  
❌ **Primarily categorical features** → [TabTransformer](tabtransformer) (specialized)

## Performance Overview

```{note}
**Qualitative assessment:** Mambular consistently performs well across classification, regression, and LSS tasks. Performance is competitive with or exceeds transformer-based models while maintaining faster training and linear complexity.
```

**Relative strengths:**

- **vs FTTransformer:** Similar accuracy, ~40% faster training, lower memory
- **vs MambaTab:** Higher capacity model, better on complex/large datasets
- **vs ResNet:** More expressive, better on datasets with complex interactions
- **vs NODE:** Typically higher accuracy, less interpretable

```{tip}
**When to expect best results:** Medium to large datasets (>5K samples), mixed categorical/numerical features, production deployments where inference speed matters.
```

## Computational Characteristics

```{note}
**Complexity advantage:** O(n·d) scaling makes Mambular efficient on large datasets and feature counts compared to O(n·f²·d) transformer models.
```

### Training Efficiency

**Relative training speed:**

- **Faster than:** FTTransformer (~40% faster), SAINT, TabR
- **Comparable to:** MambAttention, NODE, TabM
- **Slower than:** MLP, ResNet, MambaTab

**Training scales linearly** with dataset size due to O(n·d) complexity (no quadratic attention bottleneck).

### Inference Performance

**Latency:** Low latency due to sequential SSM processing (no attention matrix computation)

**Throughput:** High throughput on both CPU and GPU

**Scalability:** Linear O(n) complexity maintains performance on large batches

### Memory Requirements

**Memory scaling:** Linear with dataset size O(n) and features O(d)

**Typical footprint:** Low to medium compared to transformer models (no O(f²) attention matrices)

**GPU friendly:** Efficient CUDA kernels for Mamba operations enable good GPU utilization

## Configuration Guidelines

### Model Config (MambularConfig)

```{note}
**Parameter tuning:** Start with defaults and adjust based on dataset size. `d_model` and `n_layers` have the largest impact on model capacity and training time.
```

| Parameter        | Default | Typical Range | Description                 |
| ---------------- | ------- | ------------- | --------------------------- |
| `d_model`        | 64      | 64-512        | Hidden dimension            |
| `n_layers`       | 8       | 4-12          | Number of Mamba blocks      |
| `expand_factor`  | 2       | 1-4           | SSM state expansion         |
| `d_conv`         | 4       | 2-8           | Local convolution width     |
| `dropout`        | 0.0     | 0.0-0.5       | Dropout rate                |
| `bias`           | False   | True/False    | Use bias in linear layers   |
| `layer_norm_eps` | 1e-5    | 1e-6-1e-4     | Layer normalization epsilon |

### Recommended Settings by Dataset Size

**Small datasets (<5K samples):**

```python
from deeptab.configs import MambularConfig, TrainerConfig

cfg = MambularConfig(
    d_model=64,        # Lower capacity to prevent overfitting
    n_layers=4,        # Shallower network
    dropout=0.2,       # High dropout for regularization
)

trainer = TrainerConfig(
    lr=1e-3,           # Higher learning rate acceptable
    batch_size=128,    # Smaller batches for better generalization
    max_epochs=100,
    patience=15,
)
```

**Medium datasets (5K-50K samples):**

```python
cfg = MambularConfig(
    d_model=128,       # Sweet spot for capacity
    n_layers=6,        # Moderate depth
    dropout=0.1,       # Light regularization
)

trainer = TrainerConfig(
    lr=5e-4,           # Conservative learning rate
    batch_size=256,
    max_epochs=150,
    patience=20,
)
```

**Large datasets (>50K samples):**

```python
cfg = MambularConfig(
    d_model=256,       # High capacity
    n_layers=8,        # Full depth
    dropout=0.0,       # No dropout needed
)

trainer = TrainerConfig(
    lr=1e-4,           # Lower learning rate for stability
    batch_size=512,    # Larger batches for efficiency
    max_epochs=200,
    patience=25,
)
```

## Quick Start

```python
from deeptab.models import MambularClassifier, MambularRegressor, MambularLSS
from deeptab.configs import MambularConfig

# Classification (default config often works well)
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)
predictions = model.predict(X_test)

# Regression with custom config
cfg = MambularConfig(d_model=128, n_layers=6)
model = MambularRegressor(model_config=cfg)
model.fit(X_train, y_train, max_epochs=50)

# LSS (distributional regression)
model = MambularLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)
params = model.predict(X_test)  # Returns (mean, std) for each sample
```

## Architecture Details

### Selective State Space Mechanism

Unlike fixed SSMs, Mamba's selectivity allows input-dependent state transitions:

**Traditional SSM:**

```
h_t = A·h_{t-1} + B·x_t  (A, B fixed)
```

**Mamba (Selective SSM):**

```
h_t = A(x_t)·h_{t-1} + B(x_t)·x_t  (A, B depend on input)
```

This selectivity enables:

- **Adaptive forgetting** — Discard irrelevant past states
- **Input-aware filtering** — Emphasize important features
- **Long-range dependencies** — Maintain relevant information across sequences

### Computational Efficiency

**Why Mamba is faster than Transformers:**

| Operation     | Transformer | Mamba      |
| ------------- | ----------- | ---------- |
| Attention     | O(n²·d)     | Not needed |
| State update  | -           | O(n·d)     |
| Total forward | O(n²·d)     | O(n·d)     |
| Memory        | O(n²)       | O(n)       |

**Practical implications:**

- Transformer: Quadratic scaling limits to ~50-100 features efficiently
- Mamba: Linear scaling handles hundreds of features with ease

## Comparison with Alternatives

```{note}
**Trade-off analysis:** Architectural characteristics and relative strengths across model families.
```

| Model         | Relative Performance | Training Speed | Inference  | Memory     | Interpretability |
| ------------- | -------------------- | -------------- | ---------- | ---------- | ---------------- |
| **Mambular**  | Strong               | Moderate       | O(n)       | Low        | Low              |
| FTTransformer | Strong               | Slow           | O(n·f²)    | High       | Low              |
| MambaTab      | Good                 | **Fast**       | O(n)       | **Lowest** | Low              |
| MambAttention | Strong               | Moderate       | O(n·f²)    | Medium     | Low              |
| ResNet        | Good                 | Very Fast      | O(n)       | Low        | Medium           |
| NODE          | Good                 | Moderate       | O(n·log n) | Medium     | **High**         |

**When to choose each:**

- **Mambular:** Best general-purpose (recommended default)
- **FTTransformer:** If you have GPU memory and prioritize accuracy
- **MambaTab:** Need fastest training or working with small datasets
- **ResNet:** Extremely limited compute, need simplicity
- **NODE:** Interpretability required (e.g., regulated domains)

## Known Limitations

```{warning}
**Current limitations:**
- **Very small datasets (<1K):** Simpler models may outperform due to overfitting risk
- **Interpretability:** Black-box nature makes feature importance hard to extract
- **Categorical-only data:** Slight disadvantage vs TabTransformer on >80% categorical features
```

## References

**Original Mamba paper:**

- Gu, A., & Dao, T. (2024). _Mamba: Linear-Time Sequence Modeling with Selective State Spaces_. arXiv:2312.00752

**Related work:**

- Gu, A., et al. (2022). _Efficiently Modeling Long Sequences with Structured State Spaces_. ICLR 2022
- Fu, D., et al. (2023). _Hungry Hungry Hippos: Towards Language Modeling with State Space Models_. ICLR 2023

**Implementation:**

- DeepTab adaptation includes tabular-specific modifications to the original Mamba architecture

## See Also

- [MambaTab](mambatab) — Lightweight variant with single Mamba block
- [MambAttention](mambattention) — Hybrid combining Mamba + Transformer attention
- [Model Comparison](../comparison_tables) — Performance across all models
- [Hyperparameter Guide](../recommended_configs) — Configuration recommendations

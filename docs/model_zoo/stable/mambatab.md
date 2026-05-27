# MambaTab

**Single-block Mamba architecture** — Lightweight SSM variant optimized for speed and small datasets.

```{tip}
**Architecture highlight:** Single Mamba block trades depth for speed. Maintains O(n·d) linear complexity with ~50% faster training than Mambular. Excellent for prototyping and resource-constrained environments.
```

## Architecture Overview

**Core mechanism:** Single selective state space model block  
**Complexity:** O(n·d) time per forward pass (same as Mambular but single layer)  
**Memory:** O(d) minimal (no multi-layer stacking)  
**Inductive bias:** Sequential feature processing with selective attention

### Key Components

1. **Feature embedding:** Projects features to d_model dimensions
2. **Single Mamba block:** One selective SSM layer
3. **Output head:** Task-specific projection

**Architecture comparison:**

| Model | Mamba Blocks | Typical Params | Training Speed |
| ----- | ------------ | -------------- | -------------- |
| **MambaTab** | 1 | 50K-200K | Baseline (fastest SSM) |
| Mambular | 4-12 | 100K-500K | ~1.5x slower |
| MambAttention | Hybrid | 200K-1M | ~2x slower |

```{note}
**Design trade-off:** MambaTab sacrifices capacity (single block) for speed. Best when data is limited or compute budget is tight. For datasets >10K with sufficient compute, Mambular's additional depth typically worth the cost.
```

## When to Use

| Scenario | Recommendation | Reasoning |
| -------- | -------------- | --------- |
| **Small datasets (<5K samples)** | ✅ Use MambaTab | Lower capacity reduces overfitting risk |
| **Fast training needed** | ✅ Use MambaTab | Fastest SSM variant, 1.5x faster than Mambular |
| **Limited compute/memory** | ✅ Use MambaTab | Minimal parameters, low memory footprint |
| **Quick prototyping** | ✅ Use MambaTab | Fast iteration cycles for experimentation |
| **Production with strict latency** | ✅ Use MambaTab | Lower inference time than multi-block |
| **Large datasets (>10K)** | ❌ Use [Mambular](mambular) | Additional capacity worth the cost |
| **Maximum accuracy needed** | ❌ Use [Mambular](mambular) or [FTTransformer](fttransformer) | 5-10% better typical |
| **Complex feature interactions** | ❌ Use [Mambular](mambular) | Multiple blocks capture hierarchical patterns |

## Computational Characteristics

### Complexity Analysis

| Model | Time Complexity | Layers | Parameters | Memory |
| ----- | --------------- | ------ | ---------- | ------ |
| **MambaTab** | O(n·d) | 1 | ~100K | Minimal |
| Mambular | O(n·L·d) | 4-12 | ~300K | Low |
| MLP | O(n·d²) | 4-16 | ~100K | Minimal |
| ResNet | O(n·L·d²) | 4-16 | ~200K | Low |

### Training Efficiency

| Model | Relative Training Speed | GPU Memory | Best Use Case |
| ----- | ----------------------- | ---------- | ------------- |
| **MambaTab** | Baseline (fastest SSM) | Low | Fast SSM baseline |
| MLP | ~1.2x faster | Minimal | Absolute fastest |
| ResNet | ~1.3x faster | Low | Fast traditional |
| Mambular | ~1.5x slower | Low-Medium | Accuracy > speed |
| FTTransformer | ~2.5x slower | High | Maximum accuracy |

```{tip}
**Speed advantage:** MambaTab trains ~50% faster than Mambular while retaining ~95% of its accuracy on small-medium datasets.
```

### Accuracy vs Speed Trade-off

| Model | Typical Accuracy (relative) | Training Time (relative) | Sweet Spot |
| ----- | --------------------------- | ------------------------ | ---------- |
| **MambaTab** | 95% of Mambular | 1.0x (baseline) | <10K samples, speed matters |
| Mambular | 100% (reference) | 1.5x | >10K samples, general use |
| MLP | 85-90% | 0.8x | Absolute speed |
| ResNet | 90-95% | 0.9x | Fast traditional |

## Configuration Guidelines

### Model Config (MambaTabConfig)

```{note}
**Simplicity:** Fewer parameters than multi-block models. Primary tuning: d_model and dropout. Expand_factor affects SSM state space dimension.
```

| Parameter | Default | Typical Range | Description | Impact |
| --------- | ------- | ------------- | ----------- | ------ |
| `d_model` | 64 | 32-256 | Embedding dimension | High - capacity control |
| `expand_factor` | 2 | 1-4 | SSM state expansion | Moderate - state richness |
| `d_conv` | 4 | 2-8 | Local convolution kernel | Low - local context |
| `dropout` | 0.0 | 0.0-0.3 | Dropout rate | Dataset-dependent |
| `bias` | False | True/False | Use bias | Low |

### Recommended Settings by Dataset Size

| Dataset Size | d_model | expand_factor | dropout | batch_size | Reasoning |
| ------------ | ------- | ------------- | ------- | ---------- | --------- |
| **<1K samples** | 32-64 | 1-2 | 0.2-0.3 | 64 | Minimal capacity to prevent overfitting |
| **1K-5K samples** | 64-128 | 2 | 0.1-0.2 | 128 | Balanced capacity |
| **5K-10K samples** | 128 | 2-3 | 0.0-0.1 | 256 | Full capacity for single block |
| **>10K samples** | Consider Mambular | - | - | - | Multi-block worth the cost |

### Quick Start

```python
from deeptab.models import MambaTabClassifier, MambaTabRegressor, MambaTabLSS
from deeptab.configs import MambaTabConfig, TrainerConfig

# Fast baseline with defaults
model = MambaTabClassifier()
model.fit(X_train, y_train, max_epochs=50)
predictions = model.predict(X_test)

# Custom configuration for small dataset
cfg = MambaTabConfig(
    d_model=64,
    expand_factor=2,
    dropout=0.2,
)
trainer = TrainerConfig(
    lr=5e-4,
    batch_size=128,
    max_epochs=100,
)
model = MambaTabRegressor(model_config=cfg, trainer_config=trainer)
model.fit(X_train, y_train)

# LSS mode
model = MambaTabLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)
```

## Performance Characteristics

### Comparative Analysis

| vs Model | Accuracy Gap | Speed Advantage | Memory | When to Prefer MambaTab | When to Prefer Alternative |
| -------- | ------------ | --------------- | ------ | ----------------------- | -------------------------- |
| **Mambular** | -3 to -7% | 1.5x faster | Similar | Small data, speed critical | >10K samples, max accuracy |
| **ResNet** | Similar to +3% | Slightly slower | Similar | SSM inductive bias | Simplest baseline |
| **MLP** | +5 to +10% | Slightly slower | Similar | Better accuracy | Absolute speed |
| **FTTransformer** | -5 to -10% | 2.5x faster | Much lower | Limited memory, speed | Complex interactions |

```{note}
**Performance profile:** MambaTab performs best on small-to-medium datasets (<10K samples) where its efficiency shines. On larger datasets, Mambular's additional depth typically recovers the 3-7% accuracy gap.
```

### Use Case Suitability

| Use Case | Suitability | Reasoning |
| -------- | ----------- | --------- |
| Small datasets (<5K) | ⭐⭐⭐⭐⭐ | Optimal capacity for data size |
| Fast prototyping | ⭐⭐⭐⭐⭐ | Quick training iterations |
| Resource-constrained | ⭐⭐⭐⭐⭐ | Minimal compute requirements |
| Medium datasets (5-10K) | ⭐⭐⭐⭐ | Good speed-accuracy trade-off |
| Large datasets (>10K) | ⭐⭐⭐ | Consider Mambular |
| Maximum accuracy | ⭐⭐⭐ | Multi-block models better |

## Architecture Details

### Single Block Design Philosophy

**Multi-block (Mambular):**
```
Input → Mamba₁ → Mamba₂ → ... → Mambaₙ → Output
        ↓ features ↓        ↓ hierarchical ↓
        ↓ level 1   ↓       ↓ abstractions ↓
```

**Single block (MambaTab):**
```
Input → Mamba → Output
        ↓ single-pass ↓
        ↓ transformation ↓
```

**Trade-offs:**

| Aspect | Single Block (MambaTab) | Multi-Block (Mambular) |
| ------ | ----------------------- | ---------------------- |
| **Capacity** | Lower | Higher |
| **Speed** | Faster (~1.5x) | Slower |
| **Overfitting risk** | Lower (fewer params) | Higher (needs more data) |
| **Feature abstraction** | Single level | Hierarchical |
| **Best for** | Small data, speed | Large data, accuracy |

### Why Single Block Works

```{note}
**Sufficiency principle:** For many tabular datasets with <10K samples, single-pass transformation sufficient. Diminishing returns from additional depth when data limited.
```

**Advantages on small data:**
1. **Parameter efficiency:** Fewer parameters reduce overfitting
2. **Faster convergence:** Simpler optimization landscape
3. **Lower variance:** More stable training
4. **Adequate capacity:** Most tabular patterns not deeply hierarchical

## Known Limitations

```{warning}
**Capacity constraints:**
- **Large datasets:** Single block may underfit on >10K samples
- **Complex patterns:** Hierarchical features need multi-block depth
- **Accuracy ceiling:** Typically 3-7% below Mambular on large data
- **Feature interactions:** Limited depth constrains interaction modeling
```

**When limitations matter:**
- Dataset >10K samples → Use Mambular (additional capacity worth cost)
- Complex hierarchical patterns → Use Mambular or FTTransformer
- Maximum accuracy required → Multi-block or attention-based models

## Migration Path

```{tip}
**Start with MambaTab, scale to Mambular:** Common workflow is prototype with MambaTab for fast iteration, then migrate to Mambular if accuracy needs justify slower training.
```

**Migration is seamless:**
```python
# Start with MambaTab for fast experimentation
from deeptab.models import MambaTabClassifier
model = MambaTabClassifier()
model.fit(X_train, y_train, max_epochs=50)
# Accuracy: 0.85

# If need more accuracy, upgrade to Mambular
from deeptab.models import MambularClassifier
model = MambularClassifier()  # Same API!
model.fit(X_train, y_train, max_epochs=50)
# Accuracy: 0.88 (3% gain, 1.5x slower)
```

## References

**Mamba foundation:**
- Gu, A., & Dao, T. (2024). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752

**Architectural principle:**
- Single-layer effectiveness: Simpler models often sufficient for limited data (Occam's Razor)

## See Also

- [Mambular](mambular) — Multi-block variant for better accuracy
- [MambAttention](mambattention) — Hybrid with attention mechanism
- [ResNet](resnet) — Alternative fast baseline
- [Comparison Tables](../comparison_tables) — Performance across all models

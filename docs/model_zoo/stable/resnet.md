# ResNet

**Residual Network for tabular data** — Deep feedforward MLP with skip connections enabling stable gradient flow.

```{tip}
**Architecture highlight:** Residual connections allow training deep networks (8-16 layers) without degradation. Simple, fast, and remarkably effective baseline with O(n·d) complexity.
```

## Architecture Overview

**Core mechanism:** Stacked residual blocks with skip connections  
**Complexity:** O(n·d) time per forward pass (linear in features)  
**Memory:** O(d) per layer (minimal, no attention matrices)  
**Inductive bias:** Hierarchical feature transformation with identity shortcuts

### Key Components

1. **Input projection:** Maps features to d_model dimensions
2. **Residual blocks (×N):** `output = activation(Linear(input)) + input`
3. **Batch normalization:** Stabilizes training in each block
4. **Output head:** Task-specific projection

**Architecture diagram:**

```
Input → Projection → [Block₁ → Block₂ → ... → Blockₙ] → Head → Output
                      ↓ +skip ↓     ↓ +skip ↓
```

```{note}
**Why skip connections matter:** Without residual connections, deep MLPs suffer from vanishing gradients. Skip connections provide direct gradient paths, enabling stable training of 8-16+ layer networks.
```

## When to Use

| Scenario                         | Recommendation                                                | Reasoning                                |
| -------------------------------- | ------------------------------------------------------------- | ---------------------------------------- |
| **Need fast baseline**           | ✅ Use ResNet                                                 | 3-5x faster than transformers            |
| **Limited compute/memory**       | ✅ Use ResNet                                                 | O(n·d) linear complexity, minimal memory |
| **Quick experimentation**        | ✅ Use ResNet                                                 | Fast iteration cycles                    |
| **Simple feature relationships** | ✅ Use ResNet                                                 | Effective without complex modeling       |
| **Production speed constraints** | ✅ Use ResNet                                                 | Low latency inference                    |
| **Need maximum accuracy**        | ❌ Use [Mambular](mambular) or [FTTransformer](fttransformer) | 5-10% better typical                     |
| **Complex feature interactions** | ❌ Use transformers or [Mambular](mambular)                   | Attention/SSM better at interactions     |
| **Want interpretability**        | ❌ Use [NODE](node) or [NDTF](ndtf)                           | Tree-based models more interpretable     |

## Computational Characteristics

### Complexity Analysis

| Operation              | Per Layer | Total (L layers) | Scaling                     |
| ---------------------- | --------- | ---------------- | --------------------------- |
| Linear transformation  | O(n·d²)   | O(n·L·d²)        | Linear in samples, features |
| Activation + skip      | O(n·d)    | O(n·L·d)         | Negligible                  |
| Batch norm             | O(n·d)    | O(n·L·d)         | Negligible                  |
| **Total forward pass** | -         | **O(n·L·d²)**    | **Linear in all dims**      |

**Comparison with other architectures:**

| Model         | Time Complexity | Memory per Layer | Bottleneck          |
| ------------- | --------------- | ---------------- | ------------------- |
| **ResNet**    | O(n·d²)         | O(d)             | Simple linear ops   |
| FTTransformer | O(n·f²·d)       | O(f²)            | Quadratic attention |
| Mambular      | O(n·d²)         | O(d)             | SSM convolution     |
| NODE          | O(n·d·log d)    | O(d·2^depth)     | Tree routing        |

### Training Efficiency

| Model         | Relative Training Speed | GPU Memory | CPU Viable |
| ------------- | ----------------------- | ---------- | ---------- |
| **ResNet**    | Baseline (fastest)      | Low        | ✅ Yes     |
| MLP           | ~1.2x faster            | Minimal    | ✅ Yes     |
| MambaTab      | ~1.3x slower            | Low        | ✅ Yes     |
| Mambular      | ~2x slower              | Low-Medium | Partial    |
| FTTransformer | ~3x slower              | High       | ❌ No      |
| SAINT         | ~4-5x slower            | Very High  | ❌ No      |

## Configuration Guidelines

### Model Config (ResNetConfig)

```{note}
**Robustness:** ResNet remarkably stable across hyperparameter ranges. Default settings often sufficient. `n_layers` has more impact than `d_model` after certain threshold.
```

| Parameter  | Default | Typical Range   | Description                     | Impact                              |
| ---------- | ------- | --------------- | ------------------------------- | ----------------------------------- |
| `d_model`  | 64      | 32-256          | Hidden dimension                | Moderate - diminishing returns >128 |
| `n_layers` | 8       | 4-16            | Number of residual blocks       | High - depth = capacity             |
| `dropout`  | 0.0     | 0.0-0.5         | Dropout rate                    | Dataset-dependent regularization    |
| `d_block`  | None    | Same as d_model | Block hidden dim (if different) | Rarely tuned                        |

### Recommended Settings by Dataset Size

| Dataset Size       | d_model | n_layers | dropout | batch_size | lr           | Reasoning                           |
| ------------------ | ------- | -------- | ------- | ---------- | ------------ | ----------------------------------- |
| **<5K samples**    | 64-128  | 4-6      | 0.2-0.3 | 128        | 1e-3         | Lower capacity, high regularization |
| **5K-50K samples** | 128     | 6-8      | 0.1-0.2 | 256        | 5e-4 to 1e-3 | Balanced setup                      |
| **>50K samples**   | 128-256 | 8-12     | 0.0-0.1 | 512        | 5e-4         | Full capacity, large batches        |

### Quick Start

```python
from deeptab.models import ResNetClassifier, ResNetRegressor, ResNetLSS
from deeptab.configs import ResNetConfig, TrainerConfig

# Fast baseline with defaults
model = ResNetClassifier()
model.fit(X_train, y_train, max_epochs=50)
predictions = model.predict(X_test)

# Custom configuration
cfg = ResNetConfig(
    d_model=128,
    n_layers=8,
    dropout=0.1,
)
trainer = TrainerConfig(
    lr=1e-3,           # Can use higher lr than transformers
    batch_size=512,    # Larger batches work well
    max_epochs=100,
)
model = ResNetRegressor(model_config=cfg, trainer_config=trainer)
model.fit(X_train, y_train)

# LSS (distributional regression)
model = ResNetLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)
```

## Performance Characteristics

### Comparative Analysis

| vs Model          | Accuracy Gap | Speed Advantage | Memory Advantage      | When to Prefer ResNet         | When to Prefer Alternative   |
| ----------------- | ------------ | --------------- | --------------------- | ----------------------------- | ---------------------------- |
| **Mambular**      | -5 to -10%   | 2x faster       | Similar               | Speed critical, fast baseline | Maximum accuracy             |
| **FTTransformer** | -5 to -10%   | 3x faster       | Much lower (no O(f²)) | Limited compute/memory        | Complex feature interactions |
| **MLP**           | +3 to +5%    | Slightly slower | Similar               | Better accuracy, still fast   | Absolute fastest             |
| **NODE**          | -2 to +2%    | Similar         | Similar               | Speed, simplicity             | Interpretability             |
| **TabM**          | -2 to +5%    | Similar         | Similar               | Single model simplicity       | Ensemble benefits            |

```{note}
**Accuracy-speed trade-off:** ResNet typically achieves 80-90% of best model's accuracy with 2-5x faster training. Excellent choice for fast iteration and baselines.
```

### Use Case Suitability

| Use Case                            | Suitability | Reasoning                        |
| ----------------------------------- | ----------- | -------------------------------- |
| Fast baseline/prototyping           | ⭐⭐⭐⭐⭐  | Fastest among competitive models |
| Production with latency constraints | ⭐⭐⭐⭐⭐  | Low inference time, small memory |
| Limited GPU/CPU-only deployment     | ⭐⭐⭐⭐⭐  | Works well on CPU                |
| General-purpose modeling            | ⭐⭐⭐⭐    | Good default, robust             |
| Maximum accuracy                    | ⭐⭐⭐      | Consider Mambular/FTTransformer  |
| Interpretability                    | ⭐⭐        | Tree models better               |

## Architecture Details

### Residual Block Mechanism

**Standard MLP problem:**

```
Deep MLP: x → f₁(x) → f₂(f₁(x)) → ... → fₙ(...)  ← vanishing gradients
```

**ResNet solution:**

```
Residual: x → x + f₁(x) → x + f₂(x) + f₁(x) → ...  ← direct gradient path
```

**Benefits:**

- **Gradient flow:** Skip connections provide direct backpropagation path
- **Identity initialization:** Network can learn to do nothing (x + 0), then add complexity
- **Depth without degradation:** Can stack many layers (8-16+) without performance collapse

### Why Effective for Tabular Data

| Property            | Benefit for Tabular                               |
| ------------------- | ------------------------------------------------- |
| Linear complexity   | Scales to hundreds of features efficiently        |
| No attention        | No assumptions about feature relationships        |
| Skip connections    | Can learn both simple and complex transformations |
| Batch normalization | Handles varied feature scales naturally           |
| Simplicity          | Fewer failure modes, easier debugging             |

## Known Limitations

```{warning}
**Architectural constraints:**
- **Limited feature interactions:** No explicit mechanism for modeling complex interactions (unlike attention)
- **Lower accuracy ceiling:** Typically 5-10% below state-of-the-art on complex datasets
- **Black box:** No interpretability (consider NODE if needed)
- **Feature engineering:** May need good preprocessing to excel
```

**When limitations matter:**

- Complex feature interactions crucial → FTTransformer or Mambular
- Maximum accuracy required → Mambular or ensemble
- Interpretability needed → NODE, ENODE, NDTF
- Structured/sequential data → Consider specialized architectures

## References

**Original ResNet paper:**

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). _Deep Residual Learning for Image Recognition_. CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

**Tabular adaptation:**

- Gorishniy et al. (2021). _Revisiting Deep Learning Models for Tabular Data_. NeurIPS 2021 (comparative study including ResNet baselines)

**Related work:**

- Batch normalization: Ioffe & Szegedy (2015). _Batch Normalization: Accelerating Deep Network Training_

## See Also

- [MLP](mlp) — Even simpler baseline without skip connections
- [Mambular](mambular) — Better accuracy, similar complexity
- [FTTransformer](fttransformer) — Feature interactions via attention
- [Comparison Tables](../comparison_tables) — Performance across all models

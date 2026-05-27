# FTTransformer

**Feature Tokenizer Transformer** — Applies self-attention over tokenized features for tabular learning.

```{tip}
**Architecture highlight:** Unified tokenization of numerical and categorical features enables attention-based feature interaction modeling. Strong general-purpose model with O(n·f²·d) complexity.
```

## Architecture Overview

**Core mechanism:** Feature-wise tokenization + multi-head self-attention  
**Complexity:** O(n·f²·d) time per forward pass  
**Memory:** O(f²) for attention matrices (quadratic in feature count)  
**Inductive bias:** Feature interactions through attention

### Key Components

1. **Feature tokenization:** Each feature (numerical or categorical) → d_model-dimensional token
2. **Transformer encoder (×N layers):** Multi-head self-attention + feedforward blocks
3. **CLS token:** Special learnable token aggregates information for prediction
4. **Output head:** Task-specific projection from CLS token

**Architecture diagram:**

```
Features → Tokenize → [CLS | f₁ | f₂ | ... | fₙ] → Transformer Encoder → CLS token → Output
                                                    ↓ Self-Attention ↓
```

```{note}
**Tokenization strategy:** All features treated uniformly as tokens, unlike TabTransformer which only tokenizes categoricals. This enables attention to capture numerical-categorical and numerical-numerical interactions.
```

## When to Use

| Scenario                           | Recommendation                                  | Reasoning                                                    |
| ---------------------------------- | ----------------------------------------------- | ------------------------------------------------------------ |
| **Feature interactions important** | ✅ Use FTTransformer                            | Attention mechanism excels at modeling feature relationships |
| **Medium feature count (<100)**    | ✅ Use FTTransformer                            | O(f²) quadratic complexity manageable                        |
| **Sufficient GPU memory (>8GB)**   | ✅ Use FTTransformer                            | Attention matrices require O(f²) space per sample            |
| **General-purpose modeling**       | ✅ Use FTTransformer                            | No assumptions about data structure                          |
| **Many features (>100)**           | ❌ Use [Mambular](mambular)                     | Linear complexity more efficient                             |
| **Limited memory (<8GB GPU)**      | ❌ Use [ResNet](resnet) or [MLP](mlp)           | Quadratic attention too memory-intensive                     |
| **Need fastest training**          | ❌ Use [ResNet](resnet) or [MambaTab](mambatab) | 3-5x faster training time                                    |
| **Primarily categorical (>80%)**   | ❌ Use [TabTransformer](tabtransformer)         | Specialized for categorical-only attention                   |

## Computational Characteristics

```{note}
**Scaling analysis:** Attention over f features costs O(f²) per sample. For datasets with 50-100 features, this becomes the bottleneck compared to O(f) models like ResNet or Mambular.
```

### Complexity

**Per forward pass:**

- Tokenization: O(n·f·d)
- Attention (per layer): O(n·f²·d)
- Feedforward (per layer): O(n·f·d)
- **Total:** O(n·f²·d) dominated by attention

### Memory Requirements

**GPU memory scales with:**

- Model parameters: O(L·d²) where L = n_layers
- Attention matrices: O(f²) per sample per layer (quadratic in features!)
- Batch processing: O(batch_size · f²)

**Practical impact:** 100 features → 10K attention weights per sample per layer

### Training Efficiency

| Model             | Relative Training Speed | Reasoning                         |
| ----------------- | ----------------------- | --------------------------------- |
| **FTTransformer** | Baseline (1.0x)         | Reference point                   |
| SAINT             | ~1.5x slower            | Intersample attention O(n²)       |
| TabR              | ~1.2x slower            | Retrieval overhead at each step   |
| MambAttention     | Similar (~1.0x)         | Comparable hybrid architecture    |
| Mambular          | ~1.4x faster            | Linear SSM vs quadratic attention |
| ResNet            | ~3x faster              | Simple feedforward, no attention  |
| MLP               | ~5x faster              | Minimal architecture              |

## Configuration Guidelines

### Model Config (FTTransformerConfig)

```{note}
**Attention heads:** Use n_heads = d_model / 16 as rule of thumb. More heads allow diverse attention patterns but increase computation.
```

| Parameter      | Default | Typical Range | Description                       |
| -------------- | ------- | ------------- | --------------------------------- |
| `d_model`      | 64      | 64-512        | Token embedding dimension         |
| `n_heads`      | 8       | 4-16          | Number of attention heads         |
| `n_layers`     | 6       | 3-12          | Transformer encoder blocks        |
| `attn_dropout` | 0.0     | 0.0-0.3       | Dropout in attention layer        |
| `ffn_dropout`  | 0.0     | 0.0-0.5       | Dropout in feedforward layer      |
| `d_ffn_factor` | 4       | 2-8           | FFN hidden dim = d_model × factor |

### Recommended Settings

**Small datasets (<5K samples):**

```python
from deeptab.configs import FTTransformerConfig, TrainerConfig

cfg = FTTransformerConfig(
    d_model=64,        # Lower capacity
    n_heads=4,         # Fewer heads
    n_layers=4,        # Shallow
    attn_dropout=0.2,  # High regularization
    ffn_dropout=0.2,
)
trainer = TrainerConfig(
    lr=1e-4,           # Conservative for Transformer
    batch_size=128,
)
```

**Medium-large datasets (>5K samples):**

```python
cfg = FTTransformerConfig(
    d_model=128,       # Standard capacity
    n_heads=8,         # d_model / 16
    n_layers=6,        # Full depth
    attn_dropout=0.1,  # Light regularization
    ffn_dropout=0.1,
)
trainer = TrainerConfig(
    lr=1e-4,
    batch_size=256,
)
```

## Quick Start

```python
from deeptab.models import FTTransformerClassifier, FTTransformerRegressor, FTTransformerLSS

# Classification
model = FTTransformerClassifier()
model.fit(X_train, y_train, max_epochs=50)
predictions = model.predict(X_test)

# Regression with custom config
cfg = FTTransformerConfig(d_model=128, n_layers=6)
model = FTTransformerRegressor(model_config=cfg)
model.fit(X_train, y_train, max_epochs=50)

# LSS (distributional regression)
model = FTTransformerLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)
params = model.predict(X_test)  # Returns distribution parameters
```

## Architecture Details

### Self-Attention Mechanism

**Multi-head attention over feature tokens:**

```
Query, Key, Value = Linear(tokens)
Attention(Q, K, V) = softmax(QKᵀ/√d_k)V
```

**Why it works for tabular:**

- **Feature interactions:** Attention weights capture which features are relevant for prediction
- **Contextual embeddings:** Each feature's representation depends on all other features
- **Flexible patterns:** Different heads can learn different interaction types

### Comparison with TabTransformer

| Aspect               | FTTransformer          | TabTransformer                  |
| -------------------- | ---------------------- | ------------------------------- |
| Numerical features   | Tokenized + attended   | Pass-through (no attention)     |
| Categorical features | Tokenized + attended   | Tokenized + attended            |
| Feature interactions | All pairs              | Only categorical pairs          |
| Complexity           | O(f²) for all features | O(f_cat²) for categoricals only |

**When to prefer which:**

- **FTTransformer:** Balanced or numerical-heavy data (default choice)
- **TabTransformer:** >80% categorical features (specialized optimization)

## Performance Characteristics

### Comparative Analysis

| vs Model           | Accuracy                 | Training Speed | Memory                                    | When to Prefer FTTransformer         | When to Prefer Alternative    |
| ------------------ | ------------------------ | -------------- | ----------------------------------------- | ------------------------------------ | ----------------------------- |
| **Mambular**       | Similar                  | ~1.4x slower   | Higher (O(f²) vs O(f))                    | Strong feature interactions          | >100 features, limited memory |
| **TabTransformer** | Better (numerical-heavy) | Similar        | Higher (all features vs categorical-only) | Mixed or numerical-heavy data        | >80% categorical features     |
| **ResNet**         | Better (~5-10%)          | ~3x slower     | Similar                                   | Complex patterns, sufficient compute | Fast baseline, limited budget |
| **NODE**           | Better                   | Similar        | Similar                                   | Maximum accuracy                     | Interpretability required     |
| **MLP**            | Much better              | ~5x slower     | Similar                                   | General modeling                     | Extreme speed constraints     |

### Recommended Use Cases

| Scenario                       | Suitability | Reasoning                           |
| ------------------------------ | ----------- | ----------------------------------- | ---------------- |
| General-purpose modeling       | High        | No assumptions about data structure |
| Feature count <100             | High        | O(f²) scaling manageable            |
| Feature interactions important | High        | Attention excels at this            |
| Sufficient GPU memory (>8GB)   | High        | Can handle attention matrices       |
| Many features (>100)           | Low         | Consider Mambular (linear)          |
| Very limited compute           | Low         | ResNet or MLP faster                | ss interpretable |

**Recommended use cases:**

- General-purpose modeling when compute is available
- Feature count <100 (quadratic scaling manageable)
- When feature interactions likely important

## Known Limitations

```{warning}
**Architectural constraints:**
- **Quadratic complexity:** O(f²) attention becomes prohibitive with >100 features
- **Memory intensive:** Large attention matrices require substantial GPU RAM
- **High feature counts:** Consider Mambular (linear) for >100 features
- **Interpretability:** Attention weights don't directly indicate feature importance
```

## References

**Original paper:**

- Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). _Revisiting Deep Learning Models for Tabular Data_. NeurIPS 2021. [arXiv:2106.11959](https://arxiv.org/abs/2106.11959)

**Related work:**

- Vaswani et al. (2017). _Attention Is All You Need_. NeurIPS 2017 (original Transformer)
- Huang et al. (2020). _TabTransformer_. arXiv:2012.06678 (categorical-only variant)

**Implementation:**

- Based on the original implementation with DeepTab-specific optimizations

## See Also

- [TabTransformer](tabtransformer) — Categorical-only variant
- [Mambular](mambular) — Linear complexity alternative with similar performance
- [MambAttention](mambattention) — Hybrid Mamba + Attention architecture
- [Comparison Tables](../comparison_tables) — Performance across all models

# AutoInt

**Automatic Feature Interaction Learning via Multi-Head Self-Attention** — Explicitly models feature interactions through attention mechanism.

```{tip}
**Architecture highlight:** Multi-head self-attention automatically learns feature interactions. O(n·f²·d) complexity scales quadratically with feature count. Excels when feature crosses are critical but manual engineering infeasible. Best for datasets with 10-100 features where interactions drive predictions.
```

## Architecture Overview

**Core mechanism:** Multi-head self-attention over feature embeddings  
**Complexity:** O(n·f²·d) per forward pass where f = number of features  
**Memory:** O(f²) for attention matrices  
**Inductive bias:** All feature pairs can interact

### Key Components

1. **Feature embedding:** Projects each feature to d_model dimensions
2. **Multi-head self-attention:** Learns pairwise feature interactions
3. **Residual connections:** Preserves original feature information
4. **Feed-forward layers:** Non-linear transformations

**Architecture comparison:**

| Model         | Interaction Method | Complexity | Feature Scaling | Best For                             |
| ------------- | ------------------ | ---------- | --------------- | ------------------------------------ |
| **AutoInt**   | Explicit attention | O(n·f²·d)  | Quadratic       | Moderate features, rich interactions |
| FTTransformer | Row-wise attention | O(n·f·d)   | Linear          | Many features, simpler patterns      |
| Mambular      | Sequential SSM     | O(n·f·d)   | Linear          | General purpose                      |
| ResNet        | Implicit MLP       | O(n·f·d²)  | Linear          | Fast baseline                        |

```{note}
**Design trade-off:** AutoInt explicitly models all feature pairs via attention, making interactions interpretable but computationally expensive. With 100 features, attention requires 10,000 pairwise computations per sample.
```

## When to Use

| Scenario                            | Recommendation                        | Reasoning                                    |
| ----------------------------------- | ------------------------------------- | -------------------------------------------- |
| **Feature interactions crucial**    | ✅ Use AutoInt                        | Explicitly learns all pairwise interactions  |
| **10-100 features**                 | ✅ Use AutoInt                        | Optimal range for quadratic scaling          |
| **Need interpretability**           | ✅ Use AutoInt                        | Attention weights show interaction strengths |
| **Categorical + numerical mix**     | ✅ Use AutoInt                        | Handles both via embeddings                  |
| **Manual feature engineering hard** | ✅ Use AutoInt                        | Discovers interactions automatically         |
| **>200 features**                   | ❌ Use [FTTransformer](fttransformer) | Attention becomes expensive                  |
| **Simple additive patterns**        | ❌ Use [MLP](mlp)                     | Simpler models sufficient                    |
| **Maximum speed needed**            | ❌ Use [ResNet](resnet)               | Faster with linear feature scaling           |
| **Very small datasets (<1K)**       | ❌ Use simpler models                 | High capacity risks overfitting              |

## Computational Characteristics

### Complexity Analysis

| Model         | Time Complexity | Space (Attention) | Feature Scaling | Parameters |
| ------------- | --------------- | ----------------- | --------------- | ---------- |
| **AutoInt**   | O(n·f²·d)       | O(f²)             | Quadratic       | ~100K-500K |
| FTTransformer | O(n·f·d)        | O(f)              | Linear          | ~200K-1M   |
| Mambular      | O(n·f·d)        | O(d)              | Linear          | ~100K-500K |
| MLP           | O(n·f·d²)       | O(1)              | Linear          | ~100K-300K |

### Training Efficiency

| Model         | Training Speed | GPU Memory | Feature Count Impact | Best Use Case       |
| ------------- | -------------- | ---------- | -------------------- | ------------------- |
| **AutoInt**   | Moderate       | Medium     | High (f²)            | 10-100 features     |
| MLP           | Fast           | Low        | Low (f)              | <50 features, speed |
| ResNet        | Fast-Moderate  | Low-Medium | Low (f)              | Fast baseline       |
| FTTransformer | Slow           | High       | Low (f)              | >100 features       |
| Mambular      | Moderate       | Low-Medium | Low (f)              | General purpose     |

```{tip}
**Feature count guidelines:** AutoInt performs best with 10-100 features. Below 10, simpler models suffice. Above 100, FTTransformer's linear scaling more efficient.
```

### Scaling with Features

| Feature Count | AutoInt Feasibility | Alternative                                          |
| ------------- | ------------------- | ---------------------------------------------------- |
| <10           | Overkill            | [MLP](mlp), [ResNet](resnet)                         |
| 10-50         | ⭐⭐⭐⭐⭐ Optimal  | -                                                    |
| 50-100        | ⭐⭐⭐⭐ Good       | -                                                    |
| 100-200       | ⭐⭐⭐ Workable     | Consider [FTTransformer](fttransformer)              |
| >200          | ⭐⭐ Expensive      | [FTTransformer](fttransformer), [Mambular](mambular) |

## Configuration Guidelines

### Model Config (AutoIntConfig)

```{note}
**Key parameters:** `d_model` controls embedding richness, `n_heads` enables parallel interaction learning, `n_layers` stacks interaction blocks for hierarchical patterns. Attention dimension = d_model / n_heads must be integer.
```

| Parameter           | Default | Typical Range | Description                  | Impact                           |
| ------------------- | ------- | ------------- | ---------------------------- | -------------------------------- |
| `d_model`           | 128     | 64-256        | Embedding dimension          | High - capacity & memory         |
| `n_heads`           | 8       | 4-16          | Number of attention heads    | Moderate - parallel interactions |
| `n_layers`          | 4       | 2-8           | Number of interaction blocks | High - hierarchical modeling     |
| `dropout`           | 0.1     | 0.0-0.3       | Dropout rate                 | Dataset-dependent                |
| `attention_dropout` | 0.1     | 0.0-0.3       | Attention-specific dropout   | Regularization for interactions  |
| `use_residual`      | True    | True/False    | Residual connections         | Moderate - training stability    |

### Parameter Interactions

| d_model | n_heads | Valid? | Reasoning                    |
| ------- | ------- | ------ | ---------------------------- |
| 128     | 8       | ✅ Yes | 128/8 = 16 (head dimension)  |
| 128     | 12      | ❌ No  | 128/12 = 10.67 (not integer) |
| 256     | 16      | ✅ Yes | 256/16 = 16 (head dimension) |

### Recommended Settings by Dataset Size

| Dataset Size       | d_model | n_heads | n_layers | dropout | batch_size | Reasoning                             |
| ------------------ | ------- | ------- | -------- | ------- | ---------- | ------------------------------------- |
| **<1K samples**    | 64      | 4       | 2        | 0.2-0.3 | 64         | Minimal capacity prevents overfitting |
| **1K-5K samples**  | 128     | 8       | 3-4      | 0.1-0.2 | 128        | Balanced capacity                     |
| **5K-10K samples** | 128-192 | 8       | 4-6      | 0.1     | 256        | More interactions beneficial          |
| **>10K samples**   | 192-256 | 8-16    | 4-8      | 0.0-0.1 | 512        | Full capacity justified               |

### Quick Start

```python
from deeptab.models import AutoIntClassifier, AutoIntRegressor, AutoIntLSS
from deeptab.configs import AutoIntConfig, TrainerConfig

# Fast baseline with defaults
model = AutoIntClassifier()
model.fit(X_train, y_train, max_epochs=50)
predictions = model.predict(X_test)

# Custom configuration for interaction-rich dataset
cfg = AutoIntConfig(
    d_model=128,
    n_heads=8,
    n_layers=4,
    dropout=0.1,
    attention_dropout=0.1,
)
trainer = TrainerConfig(
    lr=1e-3,
    batch_size=256,
    max_epochs=100,
)
model = AutoIntRegressor(model_config=cfg, trainer_config=trainer)
model.fit(X_train, y_train)

# Examine learned interactions (attention weights)
# Attention weights in model.model.attention_layers show interaction strengths

# LSS mode for distributional regression
model = AutoIntLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)
```

## Performance Characteristics

### Comparative Analysis

| vs Model           | Accuracy Gap | Speed Comparison | Memory  | When to Prefer AutoInt                 | When to Prefer Alternative            |
| ------------------ | ------------ | ---------------- | ------- | -------------------------------------- | ------------------------------------- |
| **FTTransformer**  | -2 to +3%    | 1.5-2x faster    | Lower   | 10-100 features, explicit interactions | >100 features, memory constrained     |
| **Mambular**       | -3 to +5%    | Similar          | Similar | Interaction-dominated tasks            | General purpose, no interaction focus |
| **ResNet**         | +3 to +8%    | 1.3x slower      | Higher  | Feature crosses matter                 | Fast baseline, simple patterns        |
| **MLP**            | +5 to +15%   | 1.5x slower      | Higher  | Interactions essential                 | Minimal features, speed critical      |
| **GBDT (XGBoost)** | Varies       | Much faster      | Lower   | Neural approach needed                 | Traditional ML sufficient             |

```{note}
**Performance profile:** AutoInt shines on datasets where feature interactions dominate (e.g., recommendation systems, click prediction). On additive or simple patterns, overhead not justified. Typical improvement over non-interaction models: 3-10% when interactions matter.
```

### Interaction Discovery Quality

| Task Type            | AutoInt Advantage             | Best Alternative    |
| -------------------- | ----------------------------- | ------------------- |
| Click prediction     | High (interactions crucial)   | FTTransformer       |
| Recommendation       | High (user-item interactions) | FTTransformer       |
| Fraud detection      | Moderate (some interactions)  | Mambular, XGBoost   |
| Time series features | Low (temporal > interactions) | Mambular, TabularNN |
| Additive patterns    | Low (overkill)                | MLP, ResNet         |

### Use Case Suitability

| Use Case                 | Suitability | Reasoning                           |
| ------------------------ | ----------- | ----------------------------------- |
| Feature-interaction-rich | ⭐⭐⭐⭐⭐  | Designed for this scenario          |
| 10-100 features          | ⭐⭐⭐⭐⭐  | Optimal computational range         |
| Need interpretability    | ⭐⭐⭐⭐⭐  | Attention weights show interactions |
| Categorical + numerical  | ⭐⭐⭐⭐⭐  | Handles via embeddings              |
| Medium datasets (1-10K)  | ⭐⭐⭐⭐    | Good capacity balance               |
| Large datasets (>10K)    | ⭐⭐⭐⭐    | Scales well if features moderate    |
| Many features (>200)     | ⭐⭐        | Quadratic scaling expensive         |
| Simple patterns          | ⭐⭐        | Simpler models sufficient           |

## Architecture Details

### Multi-Head Self-Attention for Features

**Standard transformer attention (row-wise):**

```
Each sample attends to other samples
→ Captures sample relationships
```

**AutoInt attention (feature-wise):**

```
Each feature attends to other features
→ Captures feature interactions
```

**Mathematical formulation:**

Given feature embeddings $\mathbf{E} \in \mathbb{R}^{f \times d}$ for $f$ features:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

Where:

- $\mathbf{Q} = \mathbf{E}\mathbf{W}_Q$ (queries from features)
- $\mathbf{K} = \mathbf{E}\mathbf{W}_K$ (keys from features)
- $\mathbf{V} = \mathbf{E}\mathbf{W}_V$ (values from features)
- Output: Updated feature representations incorporating interactions

**Multi-head formulation:**

$$
\text{MultiHead}(\mathbf{E}) = \text{Concat}(\text{head}_1, ..., \text{head}_h)\mathbf{W}_O
$$

$$
\text{head}_i = \text{Attention}(\mathbf{E}\mathbf{W}_Q^i, \mathbf{E}\mathbf{W}_K^i, \mathbf{E}\mathbf{W}_V^i)
$$

Each head learns different interaction patterns in parallel.

### Interaction Example

**Concrete scenario:** Predicting house prices with features [bedrooms, bathrooms, sqft, location]

**Learned interactions might be:**

- Head 1: bedrooms ↔ bathrooms (size indicator)
- Head 2: sqft ↔ location (area importance)
- Head 3: bedrooms ↔ sqft (consistency check)
- Head 4: bathrooms ↔ location (luxury indicator)

**Attention weight interpretation:**

| Feature Pair         | Attention Weight | Interpretation                                          |
| -------------------- | ---------------- | ------------------------------------------------------- |
| sqft ↔ location      | 0.8              | Strong interaction (size matters more in certain areas) |
| bedrooms ↔ bathrooms | 0.6              | Moderate correlation                                    |
| sqft ↔ bedrooms      | 0.3              | Weaker (explained by other features)                    |

### Full Architecture Flow

```
Input features [f₁, f₂, ..., fₙ]
        ↓
Embedding layer: fᵢ → eᵢ ∈ ℝᵈ
        ↓
Embedding matrix E ∈ ℝ^(f×d)
        ↓
╔═══════════════════════════════╗
║ AutoInt Layer (repeated L times) ║
╠═══════════════════════════════╣
║ Multi-Head Self-Attention    ║
║   E' = Attention(E, E, E)     ║
║ Residual: E = E + E'          ║
║ LayerNorm(E)                  ║
║ Feed-Forward                  ║
║ Residual + LayerNorm          ║
╚═══════════════════════════════╝
        ↓
Flatten or pool: ℝ^(f×d) → ℝᵈ
        ↓
Output head (task-specific)
        ↓
Predictions
```

### Computational Bottleneck

**Per layer:**

1. **Attention:** O(f²·d) — quadratic in features
2. **Feed-forward:** O(f·d²) — quadratic in d_model
3. **Total:** O(f²·d + f·d²)

**For typical settings (f=50, d=128):**

- Attention: 50² × 128 = 320K operations
- Feed-forward: 50 × 128² = 819K operations
- Attention dominates when f > d

## Known Limitations

```{warning}
**Computational and applicability constraints:**
- **Feature count scaling:** Quadratic complexity makes >200 features expensive
- **Memory requirements:** O(f²) attention matrices for each head
- **Training time:** Slower than linear-scaling models (ResNet, Mambular)
- **Small datasets:** High capacity risks overfitting with <1K samples
- **Simple patterns:** Overhead not justified when interactions weak
- **Inference latency:** Attention computation adds overhead vs simple MLPs
```

**When limitations matter:**

- Many features (>200) → Use FTTransformer (linear scaling) or Mambular
- Speed critical → Use ResNet or MLP
- Simple additive patterns → Use MLP or linear models
- Very limited data (<1K) → Use simpler models (MLP, small ResNet)

## Interaction Analysis

```{tip}
**Interpreting learned interactions:** AutoInt's attention weights provide insights into feature importance and interactions. Higher attention weight between features indicates stronger learned interaction.
```

**Extracting attention patterns:**

```python
# After training
model = AutoIntClassifier()
model.fit(X_train, y_train, max_epochs=50)

# Access attention weights (requires model internals)
# Attention weights show which feature pairs interact strongly
# Shape: [n_layers, n_heads, n_features, n_features]

# High attention[i,j] → features i and j strongly interact
```

## Migration from Manual Feature Engineering

**Traditional approach:**

```python
# Manual interaction features
X['bed_bath_interaction'] = X['bedrooms'] * X['bathrooms']
X['sqft_per_room'] = X['sqft'] / X['bedrooms']
X['price_per_sqft_location'] = X['sqft'] * X['location_encoded']
# ... many manual crosses
```

**AutoInt approach:**

```python
# Automatic discovery
model = AutoIntRegressor()
model.fit(X, y)  # Learns optimal interactions
```

**Advantages:**

- No domain expertise needed for feature engineering
- Discovers non-obvious interactions
- Adapts to different datasets automatically

## References

**Original AutoInt paper:**

- Song, W., Shi, C., Xiao, Z., Duan, Z., Xu, Y., Zhang, M., & Tang, J. (2019). _AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks_. CIKM 2019. arXiv:1810.11921

**Related attention mechanisms:**

- Vaswani, A., et al. (2017). _Attention Is All You Need_. NeurIPS 2017. (Foundation for self-attention)

**Feature interaction learning:**

- Rendle, S. (2010). _Factorization Machines_. ICDM 2010. (Classical interaction modeling)
- Guo, H., et al. (2017). _DeepFM: A Factorization-Machine based Neural Network_. IJCAI 2017.

## See Also

- [FTTransformer](fttransformer) — Row-wise attention, better for many features
- [Mambular](mambular) — General-purpose model with linear scaling
- [ResNet](resnet) — Fast baseline without explicit interactions
- [MLP](mlp) — Simplest baseline for comparison
- [Comparison Tables](../comparison_tables) — Performance across all models

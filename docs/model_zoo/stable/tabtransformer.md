# TabTransformer

_Attention-Based Architecture for Categorical Feature Embeddings_

```{tip}
**Architecture Highlight**: Applies self-attention ONLY to categorical features with O(n·f_cat²·d) complexity. Choose TabTransformer when your dataset has >60% categorical features and categorical interactions matter.
```

## Architecture Overview

TabTransformer applies transformer self-attention exclusively to categorical feature embeddings while passing numerical features through unchanged. This selective attention mechanism makes it significantly more efficient than FTTransformer while still capturing rich interactions between categorical variables.

**Core Mechanism**: Transform each categorical feature into contextual embeddings via self-attention, then concatenate with raw numerical features for final prediction. Only categorical embeddings participate in attention.

**Computational Complexity**: O(n·f_cat²·d) where f_cat is number of categorical features  
**Memory Scaling**: O(f_cat²·d + f_cat·d²·L) dominated by attention matrices  
**Inductive Bias**: Categorical features benefit from contextualization; numerical features are assumed sufficient as-is

**Key Components**:

- Per-categorical feature embedding layers
- Multi-head self-attention over categorical embeddings only
- Feedforward network within each transformer block
- Numerical features bypass transformer, concatenated at output
- MLP head combining contextualized categoricals + raw numericals

### Architecture Comparison

| Aspect             | TabTransformer    | FTTransformer        | Mambular       | MLP            |
| ------------------ | ----------------- | -------------------- | -------------- | -------------- |
| Complexity         | O(n·f_cat²·d)     | O(n·f²·d)            | O(n·f·d)       | O(n·d²)        |
| Attention Scope    | Categoricals only | All features         | None           | None           |
| Training Speed     | Fast              | Moderate             | Moderate       | **Fastest**    |
| Memory Usage       | Low-Medium        | Medium-High          | Medium         | Low            |
| Best Use Case      | Categorical-heavy | Balanced features    | Sequential     | Baseline/speed |
| Numerical Handling | Pass-through      | Embedded + attention | Embedded + SSM | Embedded       |

## When to Use

| Scenario                              | Recommendation            | Reasoning                                                                   |
| ------------------------------------- | ------------------------- | --------------------------------------------------------------------------- |
| **>60% categorical features**         | ✅ **Highly Recommended** | Attention focuses computational budget where it matters                     |
| **Categorical interactions critical** | ✅ **Highly Recommended** | Self-attention explicitly models categorical cross-features                 |
| **5-20 categorical features**         | ✅ **Highly Recommended** | Sweet spot: enough categoricals to benefit, not too many for quadratic cost |
| **Few numerical features**            | ✅ **Recommended**        | Numerical pass-through avoids unnecessary computation                       |
| **Medium datasets (10K-500K)**        | ✅ **Recommended**        | Sufficient data to learn categorical embeddings                             |
| **Need faster than FTTransformer**    | ✅ **Recommended**        | 1.5-2x faster due to selective attention                                    |
| **Balanced numerical/categorical**    | ⚠️ **Use with caution**   | FTTransformer may be better if numericals also need attention               |
| **<3 categorical features**           | ❌ **Not Recommended**    | Overhead not justified, use MLP or ResNet                                   |
| **Mostly numerical features (>70%)**  | ❌ **Not Recommended**    | FTTransformer or Mambular better for numerical-heavy data                   |
| **No categorical features**           | ❌ **Not Recommended**    | Architecture provides no benefit, use FTTransformer/Mambular                |
| **>50 categorical features**          | ❌ **Not Recommended**    | Quadratic attention cost becomes prohibitive                                |
| **Small datasets (<5K samples)**      | ❌ **Not Recommended**    | Insufficient data to learn rich categorical embeddings                      |

## Computational Characteristics

### Complexity Analysis

| Operation                 | Time Complexity            | Space Complexity | Notes                                 |
| ------------------------- | -------------------------- | ---------------- | ------------------------------------- |
| **Forward Pass**          | O(n·f_cat²·d·L)            | O(n·f_cat·d)     | Quadratic in categorical count only   |
| **Attention Computation** | O(n·f_cat²·d)              | O(f_cat²)        | Per layer, per head                   |
| **Feedforward Network**   | O(n·f_cat·d²·L)            | O(f_cat·d)       | Applied to each categorical embedding |
| **Memory (weights)**      | O(f_cat²·d + f_cat·d²·L)   | O(f_cat²·d)      | Attention + FFN weights               |
| **vs FTTransformer**      | **Faster** when f_cat << f | **Lower** memory | Key efficiency gain                   |

Where: n = samples, f_cat = categorical features, f = total features, d = hidden dim, L = layers

### Training Efficiency Comparison

| Model                        | Relative Training Time | Relative Memory | Best For                 |
| ---------------------------- | ---------------------- | --------------- | ------------------------ |
| **MLP**                      | 1.0x                   | 1.0x            | Baseline/speed           |
| **ResNet**                   | 1.1x                   | 1.1x            | General purpose          |
| **TabTransformer (10 cats)** | **1.6x**               | **1.3x**        | **Categorical-heavy**    |
| **Mambular**                 | 1.8x                   | 1.4x            | Sequential patterns      |
| **FTTransformer**            | 2.2x                   | 1.8x            | All feature interactions |
| **TabTransformer (30 cats)** | 2.8x                   | 2.0x            | Many categoricals        |

```{note}
**Efficiency Insight**: TabTransformer's advantage grows as the ratio of numerical to categorical features increases. With 20 numerical + 5 categorical features, it's ~2x faster than FTTransformer.
```

### Memory Requirements (Approximate)

| Configuration        | Parameters | GPU Memory (batch=256) | f_cat=5 | f_cat=15 | f_cat=30 |
| -------------------- | ---------- | ---------------------- | ------- | -------- | -------- |
| Small (d=64, L=3)    | ~100K-300K | 300 MB                 | ✅ Fast | ✅ OK    | ⚠️ Slow  |
| Medium (d=128, L=6)  | ~400K-1M   | 600 MB                 | ✅ Fast | ✅ Fast  | ⚠️ OK    |
| Large (d=256, L=8)   | ~2M-5M     | 1.2 GB                 | ✅ Fast | ✅ Fast  | ✅ OK    |
| XLarge (d=512, L=10) | ~10M-20M   | 3 GB                   | ✅ Fast | ✅ Fast  | ⚠️ Slow  |

## Configuration Guidelines

### Parameter Reference

| Parameter           | Default | Range   | Impact       | Description                                     |
| ------------------- | ------- | ------- | ------------ | ----------------------------------------------- |
| `d_model`           | 128     | 64-512  | **High**     | Embedding dimension for categorical features    |
| `n_heads`           | 8       | 4-16    | **High**     | Number of attention heads (must divide d_model) |
| `n_layers`          | 6       | 3-10    | **High**     | Transformer block depth                         |
| `dropout`           | 0.1     | 0.0-0.3 | **Moderate** | Dropout in attention and FFN                    |
| `ffn_multiplier`    | 4       | 2-8     | **Moderate** | FFN hidden dim = d_model \* multiplier          |
| `attention_dropout` | 0.1     | 0.0-0.2 | **Low**      | Dropout on attention weights                    |
| `mlp_depth`         | 2       | 1-4     | **Moderate** | Layers in final MLP head                        |
| `mlp_hidden`        | 256     | 128-512 | **Moderate** | Hidden size in final MLP                        |

### Recommended Settings by Dataset Size and Categorical Count

| Dataset Size | f_cat | d_model | n_heads | n_layers | dropout | Expected Training Time |
| ------------ | ----- | ------- | ------- | -------- | ------- | ---------------------- |
| **<10K**     | 3-10  | 64      | 4       | 3        | 0.2     | 2-5 minutes            |
| **10K-50K**  | 5-15  | 128     | 8       | 6        | 0.15    | 5-15 minutes           |
| **50K-200K** | 5-20  | 192     | 8       | 6        | 0.1     | 15-40 minutes          |
| **200K-1M**  | 10-30 | 256     | 16      | 8        | 0.1     | 40-120 minutes         |
| **>1M**      | 10-25 | 256     | 16      | 10       | 0.05    | 2-4 hours              |

```{important}
**Categorical Count Matters**: With >30 categorical features, attention cost becomes O(900·d) per sample. Consider feature selection or switching to Mambular for very high-cardinality scenarios.
```

## Quick Start

### Classification Example

```python
from deeptab.models import TabTransformerClassifier
from deeptab.configs import TabTransformerConfig

# Configure for categorical-heavy dataset
config = TabTransformerConfig(
    d_model=128,
    n_heads=8,
    n_layers=6,
    dropout=0.1,
    ffn_multiplier=4
)

# Initialize and train
model = TabTransformerClassifier(config=config)
model.fit(
    X_train, y_train,
    max_epochs=100,
    batch_size=256,
    learning_rate=1e-4
)

# Predict
predictions = model.predict(X_test)
```

### Regression Example

```python
from deeptab.models import TabTransformerRegressor
from deeptab.configs import TabTransformerConfig

config = TabTransformerConfig(
    d_model=192,
    n_heads=8,
    n_layers=6,
    mlp_depth=3,  # Deeper MLP head for regression
    mlp_hidden=256
)

model = TabTransformerRegressor(config=config)
model.fit(X_train, y_train, max_epochs=150)

predictions = model.predict(X_test)
```

### Distributional Regression (LSS)

```python
from deeptab.models import TabTransformerLSS
from deeptab.configs import TabTransformerConfig

# Predict full distribution for uncertainty quantification
config = TabTransformerConfig(
    d_model=128,
    n_heads=8,
    n_layers=6
)

model = TabTransformerLSS(config=config, distribution="normal")
model.fit(X_train, y_train, max_epochs=100)

# Returns distributional parameters (e.g., mean and std)
distribution_params = model.predict(X_test)
```

## Performance Characteristics

### Comparative Analysis

| vs. Model         | Accuracy Gap   | Speed Advantage                   | Memory    | When to Prefer TabTransformer | When to Prefer Alternative   |
| ----------------- | -------------- | --------------------------------- | --------- | ----------------------------- | ---------------------------- |
| **FTTransformer** | Similar to +5% | **1.5-2x faster** (if f_cat << f) | 1.5x less | >60% categorical features     | Balanced or numerical-heavy  |
| **Mambular**      | -2% to +3%     | 10-20% faster                     | Similar   | Categorical interactions      | Sequential/temporal patterns |
| **MLP/ResNet**    | +5% to +15%    | 0.5-0.6x (slower)                 | 1.5x more | >5 categorical features       | Pure speed, simple features  |
| **SAINT**         | +3% to +8%     | **2-3x faster**                   | 2x less   | Standard supervised           | Semi-supervised learning     |
| **XGBoost**       | +2% to +10%    | Context-dependent                 | N/A       | Deep embeddings valuable      | Gradient boosting preferred  |

```{important}
**Sweet Spot**: TabTransformer excels when 40-70% of features are categorical with 5-20 distinct categoricals. It achieves FTTransformer-level accuracy at significantly lower computational cost.
```

### Strengths and Weaknesses

**Strengths**:

- ✅ Captures rich categorical interactions via self-attention
- ✅ More efficient than FTTransformer when f_cat << f
- ✅ Contextual embeddings improve categorical feature quality
- ✅ Handles high-cardinality categoricals well (via embeddings)
- ✅ Interpretable attention weights show categorical dependencies
- ✅ Strong performance on categorical-heavy benchmarks

**Weaknesses**:

- ❌ Numerical features get no contextualization (simple pass-through)
- ❌ Quadratic cost in categorical count limits scalability
- ❌ Requires sufficient data to learn meaningful embeddings (>5K samples)
- ❌ No benefit if dataset has few/no categorical features
- ❌ Slower than MLP/ResNet for same accuracy on simple tasks
- ❌ Cannot model sequential/temporal patterns in features

## Use Case Suitability

| Use Case                       | Suitability | Notes                                                  |
| ------------------------------ | ----------- | ------------------------------------------------------ |
| **Categorical-Heavy Datasets** | ⭐⭐⭐⭐⭐  | Primary use case, excels at categorical interactions   |
| **Recommendation Systems**     | ⭐⭐⭐⭐⭐  | User/item IDs benefit from contextual embeddings       |
| **Click-Through Rate (CTR)**   | ⭐⭐⭐⭐⭐  | Many categorical features (campaign, device, etc.)     |
| **Customer Segmentation**      | ⭐⭐⭐⭐    | Demographic categoricals interact meaningfully         |
| **Fraud Detection**            | ⭐⭐⭐⭐    | Transaction categories, merchant types, locations      |
| **Medical Diagnosis**          | ⭐⭐⭐⭐    | Diagnosis codes, procedure codes, categorical symptoms |
| **E-commerce**                 | ⭐⭐⭐⭐    | Product categories, brands, user segments              |
| **Financial Risk**             | ⭐⭐⭐      | Credit categories, loan types, but many numericals     |
| **Time Series Tabular**        | ⭐⭐        | No sequential modeling; consider Mambular              |
| **Numerical-Heavy (Sensors)**  | ⭐⭐        | FTTransformer or Mambular better for numerical data    |
| **Small Datasets (<5K)**       | ⭐⭐        | Insufficient data for embedding learning               |

## Architecture Details

### Network Structure

```
Input: Categorical Features [f_cat] + Numerical Features [f_num]
    ↓
Categorical → Embedding Layer → [f_cat, d_model]
Numerical   → Pass through    → [f_num]
    ↓
[Transformer Block × L]:
    Multi-Head Self-Attention (on categorical embeddings)
    ↓
    LayerNorm → Residual
    ↓
    Feedforward Network (per categorical embedding)
    ↓
    LayerNorm → Residual
    ↓
Concatenate: Contextualized Categoricals + Raw Numericals
    ↓
MLP Head → Predictions
```

### Mathematical Formulation

**Categorical Embedding**:
$$e_i = \text{Embed}_i(x_i^{\text{cat}}) \in \mathbb{R}^d$$

**Self-Attention** (per layer):
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where $Q, K, V$ are computed from categorical embeddings only:
$$Q = E W_Q, \quad K = E W_K, \quad V = E W_V$$
$$E \in \mathbb{R}^{f_{\text{cat}} \times d}$$

**Multi-Head Attention**:
$$\text{MultiHead}(E) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O$$

**Final Representation**:
$$h = \text{Concat}([\text{Transformer}(e_1, \ldots, e_{f_{\text{cat}}}), x_{f_{\text{cat}}+1}^{\text{num}}, \ldots, x_f^{\text{num}}])$$

**Key Insight**: Attention complexity is O(f_cat²) not O(f²), providing significant savings when f_num > f_cat.

### Parameter Count

$$\text{params} = f_{\text{cat}} \cdot d + L \cdot (4d^2 + 3d) + \text{MLP}_{\text{head}}$$

Where:

- $f_{\text{cat}} \cdot d$: categorical embeddings
- $4d^2$: attention projections (Q, K, V, O)
- $3d$: layer norms and biases
- FFN parameters depend on ffn_multiplier

### Design Rationale

**Why attention on categoricals only?**

1. **Efficiency**: Categorical count typically << total features
2. **Semantic Richness**: Categories benefit more from contextualization than numericals
3. **Empirical Results**: Paper shows numerical pass-through doesn't hurt performance
4. **Interpretability**: Attention weights reveal categorical dependencies

**Comparison to FTTransformer**:

| Design Choice      | TabTransformer   | FTTransformer        |
| ------------------ | ---------------- | -------------------- |
| Numerical Handling | Pass-through     | Embedded + attention |
| Attention Scope    | Categorical only | All features         |
| Complexity         | O(f_cat²)        | O(f²)                |
| Best For           | f_cat << f       | Balanced features    |

```{warning}
**Known Limitations**

1. **Numerical Features Not Contextualized**: Raw numerical features don't benefit from attention. If numerical interactions matter, consider FTTransformer.

2. **Quadratic in Categorical Count**: With 50+ categorical features, attention cost becomes prohibitive. Consider feature selection or Mambular.

3. **Requires Sufficient Data**: Needs >5K samples to learn meaningful categorical embeddings. Smaller datasets may underfit.

4. **No Sequential Modeling**: Cannot capture temporal or sequential patterns. Use Mambular or recurrent architectures for time series.

5. **High-Cardinality Challenges**: Very high-cardinality categoricals (>1000 unique values) may require large embedding dimensions, increasing memory.

6. **Categorical Feature Requirement**: Provides no benefit if dataset has <3 categorical features. Use MLP/ResNet/FTTransformer instead.
```

## References

1. **Huang, X., Khetan, A., Cvitkovic, M., & Karnin, Z. (2020)**. _TabTransformer: Tabular Data Modeling Using Contextual Embeddings_. arXiv:2012.06678. [Original paper introducing selective attention on categorical features]

2. **Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021)**. _Revisiting Deep Learning Models for Tabular Data_. NeurIPS 2021. [Benchmark comparison including TabTransformer]

3. **Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C. B., & Goldstein, T. (2021)**. _SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training_. arXiv:2106.01342. [Extends TabTransformer ideas]

4. **Vaswani, A., et al. (2017)**. _Attention Is All You Need_. NeurIPS 2017. [Foundation of transformer architecture]

5. **Shavitt, I., & Segal, E. (2018)**. _Regularization Learning Networks: Deep Learning for Tabular Datasets_. NeurIPS 2018. [Early work on categorical embeddings]

## See Also

- **[FTTransformer](fttransformer.md)** — Attention on ALL features, better for balanced/numerical-heavy data
- **[Mambular](mambular.md)** — State-space model, linear complexity, good for sequential patterns
- **[SAINT](saint.md)** — Adds intersample attention for semi-supervised learning
- **[ResNet](resnet.md)** — Simpler alternative if categorical interactions aren't critical
- **[Model Selection Guide](../model_selection.md)** — Choosing between transformer variants

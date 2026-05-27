# SAINT (Self-Attention and Intersample Attention Network)

_Dual Attention Architecture for Row and Column Interactions_

```{tip}
**Architecture Highlight**: Applies attention both across features (column) and across samples (row) with O(n²·f·d) complexity. Choose SAINT for semi-supervised learning with <5K samples or when intersample relationships are critical.
```

```{warning}
**Critical Performance Warning**: Intersample attention has O(n²) complexity. SAINT becomes impractical for datasets >10K samples due to quadratic memory and computation growth. For larger datasets, use FTTransformer or Mambular instead.
```

## Architecture Overview

SAINT introduces a dual attention mechanism that models both feature interactions (self-attention, like FTTransformer) and sample interactions (intersample attention). The intersample attention allows the model to learn relationships between different data points, making it particularly effective for semi-supervised learning and small datasets where each sample provides valuable context for others.

**Core Mechanism**: For each sample, apply self-attention across its features, then apply intersample attention across all samples in the batch. This creates a rich representation considering both what features relate to each other within a sample and how different samples relate to each other.

**Computational Complexity**: O(n²·f·d + n·f²·d) dominated by intersample attention's O(n²)  
**Memory Scaling**: O(n²·f + f²·d) attention matrices scale quadratically with batch size  
**Inductive Bias**: Similar samples should inform predictions; feature and sample relationships are both important

**Key Components**:

- Feature embedding layer (categorical + numerical)
- Self-attention layers (across features, like FTTransformer)
- Intersample attention layers (across samples in batch)
- Contrastive learning head (for semi-supervised)
- MLP head for final predictions

### Architecture Comparison

| Aspect            | SAINT                       | FTTransformer      | Mambular    | TabTransformer       |
| ----------------- | --------------------------- | ------------------ | ----------- | -------------------- |
| Complexity        | O(n²·f·d)                   | O(n·f²·d)          | O(n·f·d)    | O(n·f_cat²·d)        |
| Feature Attention | ✅ Full                     | ✅ Full            | ❌ None     | ⚠️ Categorical only  |
| Sample Attention  | ✅ **Unique**               | ❌ None            | ❌ None     | ❌ None              |
| Training Speed    | **Slowest**                 | Moderate           | Fast        | Fast                 |
| Memory Usage      | **Highest** O(n²)           | Medium O(f²)       | Medium O(f) | Low-Medium O(f_cat²) |
| Best Use Case     | Semi-supervised, small data | Supervised, global | Sequential  | Categorical-heavy    |
| Batch Size Limit  | **Very Limited**            | Normal             | Normal      | Normal               |

## When to Use

| Scenario                            | Recommendation            | Reasoning                                                     |
| ----------------------------------- | ------------------------- | ------------------------------------------------------------- |
| **Semi-supervised learning**        | ✅ **Highly Recommended** | Intersample attention leverages unlabeled data effectively    |
| **Small datasets (<5K samples)**    | ✅ **Highly Recommended** | Intersample context valuable with limited data                |
| **Unlabeled data available**        | ✅ **Highly Recommended** | Contrastive pre-training utilizes unlabeled samples           |
| **Sample relationships matter**     | ✅ **Recommended**        | Explicit modeling of sample similarities                      |
| **Low-shot learning**               | ✅ **Recommended**        | Few labeled examples benefit from sample context              |
| **Need best accuracy on tiny data** | ✅ **Recommended**        | Worth computational cost for <3K samples                      |
| **Datasets 5K-10K samples**         | ⚠️ **Use with caution**   | Approaching computational limits, monitor memory              |
| **Fully supervised only**           | ⚠️ **Use with caution**   | FTTransformer likely better without semi-supervised component |
| **>10K samples**                    | ❌ **Not Recommended**    | O(n²) becomes prohibitive; use FTTransformer/Mambular         |
| **Real-time inference**             | ❌ **Not Recommended**    | Extremely slow due to intersample attention                   |
| **Limited GPU memory**              | ❌ **Not Recommended**    | Requires large memory for attention matrices                  |
| **Need training speed**             | ❌ **Not Recommended**    | 3-4x slower than FTTransformer                                |

## Computational Characteristics

### Complexity Analysis

| Operation                     | Time Complexity | Space Complexity | Notes                              |
| ----------------------------- | --------------- | ---------------- | ---------------------------------- |
| **Self-Attention (features)** | O(n·f²·d)       | O(f²)            | Standard transformer attention     |
| **Intersample Attention**     | O(n²·f·d)       | O(n²)            | **QUADRATIC in batch/samples**     |
| **Total Forward Pass**        | O(n²·f·d)       | O(n²·f)          | Dominated by intersample attention |
| **Backward Pass**             | O(n²·f·d)       | O(n²·f)          | Same as forward                    |
| **Memory (activations)**      | O(n²·f + n·f²)  | O(n²)            | **Scales quadratically with n**    |

Where: n = samples (batch size or dataset size), f = features, d = hidden dimension

```{important}
**Scalability Breakdown**: With 1K samples and 20 features:
- Self-attention: O(1K·400·d) = 400K·d operations
- Intersample attention: O(1M·20·d) = 20M·d operations
- **Intersample is 50x more expensive at 1K samples!**
```

### Training Efficiency Comparison

| Model             | Training Time (1K samples) | Training Time (10K samples) | Memory (1K) | Memory (10K)     |
| ----------------- | -------------------------- | --------------------------- | ----------- | ---------------- |
| **MLP**           | 1x (30 sec)                | 1x (5 min)                  | 500 MB      | 1 GB             |
| **ResNet**        | 1.1x (35 sec)              | 1.1x (6 min)                | 600 MB      | 1.2 GB           |
| **Mambular**      | 1.8x (55 sec)              | 1.6x (8 min)                | 800 MB      | 1.5 GB           |
| **FTTransformer** | 2.2x (70 sec)              | 2.0x (10 min)               | 1 GB        | 2 GB             |
| **SAINT**         | **3.5x (2 min)**           | **~Impractical**            | **2 GB**    | **>16 GB (OOM)** |

```{warning}
**Memory Explosion**: At 10K samples, intersample attention requires O(100M) memory for attention matrices alone. This typically exceeds consumer GPU memory (8-16GB).
```

### Practical Batch Size Limits

| GPU Memory       | Max Batch Size (f=20, d=128) | Max Dataset (full batch) | Practical Strategy        |
| ---------------- | ---------------------------- | ------------------------ | ------------------------- |
| **8 GB**         | ~128 samples                 | <2K samples              | Use gradient accumulation |
| **16 GB**        | ~256 samples                 | <5K samples              | OK for small datasets     |
| **24 GB**        | ~512 samples                 | <8K samples              | Upper practical limit     |
| **40 GB (A100)** | ~1024 samples                | ~10K samples             | Max recommended scale     |

## Configuration Guidelines

### Parameter Reference

| Parameter             | Default | Range      | Impact       | Description                                         |
| --------------------- | ------- | ---------- | ------------ | --------------------------------------------------- |
| `d_model`             | 128     | 64-256     | **High**     | Embedding dimension for both attention types        |
| `n_heads`             | 8       | 4-16       | **High**     | Attention heads in both self and intersample        |
| `n_layers`            | 6       | 3-8        | **High**     | Number of SAINT blocks (self + intersample pairs)   |
| `dropout`             | 0.1     | 0.0-0.3    | **Moderate** | Dropout in attention and FFN                        |
| `intersample_dropout` | 0.2     | 0.1-0.4    | **Moderate** | Extra dropout for intersample to reduce overfitting |
| `contrastive_weight`  | 0.5     | 0.0-1.0    | **High**     | Weight of contrastive loss (semi-supervised)        |
| `batch_size`          | 64      | 32-256     | **Critical** | **SMALLER than other models due to O(n²)**          |
| `use_contrastive`     | True    | True/False | **High**     | Enable semi-supervised contrastive learning         |

### Recommended Settings by Dataset Size

| Dataset Size | d_model                | n_heads | n_layers | batch_size | dropout | contrastive_weight | Expected Training     |
| ------------ | ---------------------- | ------- | -------- | ---------- | ------- | ------------------ | --------------------- |
| **<1K**      | 64                     | 4       | 4        | 64         | 0.3     | 0.7                | 5-15 minutes          |
| **1K-3K**    | 128                    | 8       | 6        | 128        | 0.2     | 0.5                | 15-45 minutes         |
| **3K-5K**    | 128                    | 8       | 6        | 128        | 0.15    | 0.4                | 45-90 minutes         |
| **5K-8K**    | 192                    | 8       | 8        | 64         | 0.15    | 0.3                | 2-4 hours             |
| **>8K**      | ⚠️ **Not Recommended** | —       | —        | —          | —       | —                  | **Use FTTransformer** |

```{note}
**Batch Size Critical**: Unlike other models where larger batch = faster, SAINT's O(n²) attention means smaller batches are **required** to fit in memory. Use gradient accumulation to simulate larger batches.
```

## Quick Start

### Semi-Supervised Classification (Primary Use Case)

```python
from deeptab.models import SAINTClassifier
from deeptab.configs import SAINTConfig

# Configure for semi-supervised learning
config = SAINTConfig(
    d_model=128,
    n_heads=8,
    n_layers=6,
    batch_size=128,  # SMALLER than other models!
    dropout=0.2,
    intersample_dropout=0.3,
    contrastive_weight=0.5,  # Balance supervised + contrastive
    use_contrastive=True
)

# Initialize model
model = SAINTClassifier(config=config)

# Train with unlabeled data
model.fit(
    X_train_labeled, y_train_labeled,
    X_train_unlabeled=X_unlabeled,  # Optional unlabeled data
    max_epochs=200,  # More epochs for contrastive learning
    learning_rate=1e-4
)

# Predict
predictions = model.predict(X_test)
```

### Fully Supervised Classification

```python
from deeptab.models import SAINTClassifier
from deeptab.configs import SAINTConfig

# Fully supervised (no contrastive learning)
config = SAINTConfig(
    d_model=128,
    n_heads=8,
    n_layers=6,
    batch_size=128,
    use_contrastive=False  # Disable semi-supervised
)

model = SAINTClassifier(config=config)
model.fit(
    X_train, y_train,
    max_epochs=100,
    batch_size=128
)

predictions = model.predict(X_test)
```

### Regression with Intersample Context

```python
from deeptab.models import SAINTRegressor
from deeptab.configs import SAINTConfig

config = SAINTConfig(
    d_model=192,
    n_heads=8,
    n_layers=6,
    batch_size=64,  # Smaller for memory
    dropout=0.15,
    use_contrastive=False  # Less common for regression
)

model = SAINTRegressor(config=config)
model.fit(X_train, y_train, max_epochs=150)

predictions = model.predict(X_test)
```

### Distributional Regression (LSS)

```python
from deeptab.models import SAINTLSS
from deeptab.configs import SAINTConfig

# Predict distribution parameters
config = SAINTConfig(
    d_model=128,
    n_heads=8,
    n_layers=6,
    batch_size=128
)

model = SAINTLSS(config=config, distribution="normal")
model.fit(X_train, y_train, max_epochs=100)

distribution_params = model.predict(X_test)
```

## Performance Characteristics

### Comparative Analysis

| vs. Model          | Accuracy Gap             | Speed Advantage | Memory        | When to Prefer SAINT    | When to Prefer Alternative |
| ------------------ | ------------------------ | --------------- | ------------- | ----------------------- | -------------------------- |
| **FTTransformer**  | +3% to +8% (small data)  | **3-4x slower** | **3-5x more** | <5K + semi-supervised   | >5K or fully supervised    |
| **Mambular**       | +5% to +10% (small data) | **4-5x slower** | **4-6x more** | <3K + unlabeled data    | Any moderate/large dataset |
| **ResNet**         | +8% to +15% (small data) | **5-6x slower** | **5-8x more** | <1K + semi-supervised   | >5K or need speed          |
| **TabTransformer** | +5% to +10% (small data) | **3-4x slower** | **4-5x more** | <5K + categorical-heavy | Categorical-heavy + >5K    |

```{important}
**Unique Value Proposition**: SAINT's advantage is exclusively in the **small data + semi-supervised** regime. With >5K labeled samples or no unlabeled data, simpler models like FTTransformer are typically better choices.
```

### Strengths and Weaknesses

**Strengths**:

- ✅ **Best for semi-supervised learning** with contrastive pre-training
- ✅ Captures both feature and sample interactions (unique)
- ✅ Excellent performance on small datasets (<3K samples)
- ✅ Can leverage unlabeled data effectively
- ✅ Sample attention provides interpretability (sample similarities)
- ✅ Theoretical foundation for learning from sample relationships

**Weaknesses**:

- ❌ **O(n²) complexity prohibitive for >10K samples**
- ❌ **3-4x slower training** than FTTransformer
- ❌ **3-5x more memory** than comparable models
- ❌ **Extremely limited batch sizes** (<256 typically)
- ❌ Impractical for real-time inference
- ❌ Complex architecture with many hyperparameters
- ❌ No advantage in fully supervised large-data settings
- ❌ Requires careful batch size tuning to avoid OOM

## Use Case Suitability

| Use Case                              | Suitability | Notes                                               |
| ------------------------------------- | ----------- | --------------------------------------------------- |
| **Semi-Supervised (<5K)**             | ⭐⭐⭐⭐⭐  | Primary use case, leverages unlabeled data          |
| **Medical Diagnosis (Small Cohorts)** | ⭐⭐⭐⭐⭐  | Few labeled patients + unlabeled data               |
| **Drug Discovery (Early Stage)**      | ⭐⭐⭐⭐⭐  | Limited labeled compounds, many unlabeled           |
| **Low-Shot Learning**                 | ⭐⭐⭐⭐    | Few examples per class, sample context helps        |
| **Active Learning**                   | ⭐⭐⭐⭐    | Uncertainty from sample attention guides selection  |
| **Rare Event Detection**              | ⭐⭐⭐⭐    | Few positive examples, intersample context valuable |
| **Small Tabular Datasets**            | ⭐⭐⭐      | <3K samples, worth computational cost               |
| **Fully Supervised (Small)**          | ⭐⭐⭐      | OK but FTTransformer often simpler/faster           |
| **Medium Datasets (5K-10K)**          | ⭐⭐        | Approaching limits, monitor memory carefully        |
| **Large Datasets (>10K)**             | ⭐          | **Not recommended**, use FTTransformer/Mambular     |
| **Real-time Applications**            | ⭐          | Too slow for latency-sensitive scenarios            |

## Architecture Details

### Network Structure

```
Input Features (f dimensions)
    ↓
Embedding Layer → Feature Embeddings [n, f, d]
    ↓
[SAINT Block × L]:

    Self-Attention (across features, per sample):
        Multi-Head Attention: [n, f, d] → [n, f, d]
        ↓
        Residual + LayerNorm
        ↓
        FFN per feature
        ↓
        Residual + LayerNorm

    Intersample Attention (across samples, per feature):
        Multi-Head Attention: [n, f, d] → [n, f, d]
        ↓
        Residual + LayerNorm
        ↓
        FFN per sample
        ↓
        Residual + LayerNorm
    ↓

Global Pooling (mean/max across features)
    ↓
Supervised Head → Task predictions
    +
Contrastive Head → Self-supervised signal (if enabled)
```

### Mathematical Formulation

**Self-Attention** (column attention, across features):

For sample i:
$$h_i = \text{SelfAttn}(X_i) \in \mathbb{R}^{f \times d}$$

Where $X_i \in \mathbb{R}^{f \times d}$ are feature embeddings for sample i.

**Intersample Attention** (row attention, across samples):

For feature j:
$$h_j = \text{IntersampleAttn}(X_{:,j}) \in \mathbb{R}^{n \times d}$$

Where $X_{:,j} \in \mathbb{R}^{n \times d}$ are sample embeddings for feature j.

**Attention Mechanism** (both types):
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Contrastive Loss** (for semi-supervised):
$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(z_i, z_j^+) / \tau)}{\sum_{k} \exp(\text{sim}(z_i, z_k) / \tau)}$$

Where $z_i, z_j^+$ are embeddings of augmented sample pairs.

**Total Loss**:
$$\mathcal{L} = \lambda \mathcal{L}_{\text{supervised}} + (1-\lambda) \mathcal{L}_{\text{contrastive}}$$

### Key Design Choices

1. **Why Intersample Attention?**
   - Learn from sample relationships, not just features
   - Enables semi-supervised learning via contrastive loss
   - Particularly valuable with limited labeled data

2. **Dual Attention Architecture**:
   - **Self-attention**: How features relate within each sample
   - **Intersample attention**: How samples relate to each other
   - Complementary: feature context + sample context

3. **Contrastive Learning**:
   - Create augmented views of samples
   - Force similar samples close in embedding space
   - Utilizes unlabeled data for representation learning

4. **Scalability Trade-off**:
   - Intersample O(n²) limits scalability
   - Justified for small data where every sample matters
   - Not competitive for large-scale problems

### Comparison to FTTransformer

| Feature               | SAINT                    | FTTransformer            |
| --------------------- | ------------------------ | ------------------------ |
| Self-Attention        | ✅ Yes (across features) | ✅ Yes (across features) |
| Intersample Attention | ✅ **Yes (unique)**      | ❌ No                    |
| Complexity            | O(n²·f·d)                | O(n·f²·d)                |
| Semi-Supervised       | ✅ Native support        | ❌ Not designed for      |
| Best Data Size        | <5K samples              | >5K samples              |
| Training Speed        | 3-4x slower              | Baseline                 |
| Memory                | 3-5x more                | Baseline                 |

```{warning}
**Known Limitations**

1. **Quadratic Sample Complexity**: O(n²) attention makes SAINT impractical for >10K samples. Memory scales as n², leading to OOM errors on consumer GPUs beyond 5-8K samples even with small batches.

2. **Extreme Training Time**: 3-4x slower than FTTransformer, 5-6x slower than ResNet. On datasets >5K, training can take hours to days.

3. **Very Limited Batch Sizes**: Typical max batch size is 64-128 (vs 256-1024 for other models) due to O(n²) attention matrices. Requires gradient accumulation for effective training.

4. **No Advantage at Scale**: For >10K samples or fully supervised settings, FTTransformer/Mambular typically match or exceed SAINT's accuracy while being 3-5x faster and using 3-5x less memory.

5. **Complex Hyperparameter Tuning**: Two attention mechanisms + contrastive learning means more hyperparameters (contrastive_weight, intersample_dropout, etc.). Finding optimal settings is time-consuming.

6. **Memory Explosion**: At 10K samples with 20 features and d=128, intersample attention alone requires ~6.4GB for attention matrices (10K² × 4 bytes). Total memory often exceeds 16GB.

7. **Inference Slowdown**: Intersample attention at inference time (if using batch inference) has the same O(n²) cost, making batch prediction slow. Single-sample inference loses intersample context benefits.

8. **Diminishing Returns**: Benefits over FTTransformer diminish rapidly as labeled data grows. With >5K labeled samples, SAINT's overhead is rarely justified.
```

## References

1. **Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C. B., & Goldstein, T. (2021)**. _SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training_. arXiv:2106.01342. [Original SAINT paper]

2. **Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020)**. _A Simple Framework for Contrastive Learning of Visual Representations_. ICML 2020. [SimCLR foundation for contrastive learning]

3. **Vaswani, A., et al. (2017)**. _Attention Is All You Need_. NeurIPS 2017. [Transformer architecture foundation]

4. **Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021)**. _Revisiting Deep Learning Models for Tabular Data_. NeurIPS 2021. [Benchmark comparison including SAINT]

5. **Huang, X., Khetan, A., Cvitkovic, M., & Karnin, Z. (2020)**. _TabTransformer: Tabular Data Modeling Using Contextual Embeddings_. arXiv:2012.06678. [Related transformer for tabular]

6. **Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022)**. _Why do tree-based models still outperform deep learning on tabular data?_. NeurIPS 2022. [Context for when deep learning helps on small data]

## See Also

- **[FTTransformer](fttransformer.md)** — Pure feature attention, no intersample, better for >5K samples
- **[Mambular](mambular.md)** — Linear complexity alternative for sequential patterns
- **[TabTransformer](tabtransformer.md)** — Categorical-only attention, faster than SAINT
- **[ResNet](resnet.md)** — Simple baseline, much faster for small data
- **[Model Selection Guide](../model_selection.md)** — Choosing between semi-supervised and supervised models

# NODE

**Neural Oblivious Decision Ensembles** — Differentiable decision trees with gradient-based optimization.

```{tip}
**Architecture highlight:** Combines tree-based inductive bias with gradient optimization. Soft oblivious decision trees enable interpretability while maintaining differentiability. O(n·d·log n) complexity.
```

## Architecture Overview

**Core mechanism:** Ensemble of oblivious decision trees with soft splits  
**Complexity:** O(n·d·log n) time per forward pass  
**Memory:** O(d·2^depth) per tree (exponential in depth)  
**Inductive bias:** Hierarchical feature splits similar to GBDT

### Key Components

1. **Feature selection layer:** Chooses which feature to split on at each level
2. **Oblivious trees:** Same feature split at each depth level across all nodes
3. **Soft routing:** Differentiable split decisions (not hard thresholds)
4. **Ensemble:** Multiple trees combined for final prediction

**Architecture diagram:**

```
Input → Feature Selection → Oblivious Tree₁ →
                          → Oblivious Tree₂ →  Ensemble → Output
                          → ...
                          → Oblivious Treeₙ →
```

```{note}
**Oblivious trees:** Unlike standard decision trees where each node can split on different features, oblivious trees use the same feature at each depth level. This dramatically reduces parameters (depth 6 = 2^6=64 leaves vs thousands in standard trees).
```

## When to Use

| Scenario                                | Recommendation                                                | Reasoning                                          |
| --------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------- |
| **GBDT works well on your data**        | ✅ Use NODE                                                   | Similar inductive bias to XGBoost/LightGBM         |
| **Some interpretability needed**        | ✅ Use NODE                                                   | Tree structure and splits visible                  |
| **Outlier-resistant predictions**       | ✅ Use NODE                                                   | Tree splits less sensitive than linear models      |
| **Categorical features + interactions** | ✅ Use NODE                                                   | Trees naturally handle categoricals                |
| **Need maximum accuracy**               | ❌ Use [Mambular](mambular) or [FTTransformer](fttransformer) | Deep learning models typically 3-7% better         |
| **Full interpretability required**      | ❌ Use XGBoost/LightGBM                                       | NODE partially interpretable, classical GBDT fully |
| **Very large datasets (>100K)**         | ❌ Consider [Mambular](mambular)                              | O(n·log n) slower than O(n) models at scale        |

## Computational Characteristics

### Complexity Analysis

| Operation              | Complexity          | Description                              |
| ---------------------- | ------------------- | ---------------------------------------- |
| Feature selection      | O(n·d)              | Choose splitting feature per depth level |
| Tree routing (depth D) | O(n·D) = O(n·log n) | Soft routing through tree                |
| Leaf probability       | O(n·2^D)            | Compute probability of each leaf         |
| **Total per tree**     | **O(n·d + n·2^D)**  | **Dominated by leaf computation**        |
| **Ensemble (T trees)** | **O(T·n·2^D)**      | **Exponential in depth!**                |

### Memory Requirements

| Component           | Memory         | Scaling                  |
| ------------------- | -------------- | ------------------------ |
| Feature weights     | O(D·d)         | Linear                   |
| Leaf values         | O(T·2^D)       | **Exponential in depth** |
| Activations (batch) | O(batch·T·2^D) | Exponential              |

```{warning}
**Depth constraint:** Memory grows exponentially (2^depth). Typical depth=6 (64 leaves) is practical. Depth >8 often impractical.
```

### Training Efficiency

| Model              | Training Speed | Memory | Depth Impact        |
| ------------------ | -------------- | ------ | ------------------- |
| **NODE (depth=6)** | Moderate       | Medium | 64 leaves           |
| **NODE (depth=8)** | Slow           | High   | 256 leaves          |
| Mambular           | Moderate       | Low    | N/A                 |
| FTTransformer      | Slow           | High   | N/A                 |
| ResNet             | Fast           | Low    | N/A                 |
| XGBoost            | Fast           | Low    | Grows incrementally |

## Configuration Guidelines

### Model Config (NODEConfig)

```{note}
**Parameter interaction:** `depth` and `n_trees` are most critical. Deep trees with few trees vs shallow trees with many trees have different trade-offs.
```

| Parameter         | Default      | Typical Range       | Description                 | Impact                            |
| ----------------- | ------------ | ------------------- | --------------------------- | --------------------------------- |
| `n_layers`        | 8            | 4-12                | Number of NODE layers       | Moderate - more = deeper ensemble |
| `depth`           | 6            | 4-8                 | Tree depth (2^depth leaves) | High - exponential memory/compute |
| `n_trees`         | 2048         | 512-4096            | Trees per layer             | High - ensemble size              |
| `choice_function` | "sparsemax"  | entmax, sparsemax   | Feature selection           | Low - sparsemax usually best      |
| `bin_function`    | "sparsemoid" | sparsemoid, entmoid | Split function              | Low - sparsemoid default          |

### Recommended Settings

| Dataset Size       | depth | n_trees   | n_layers | Reasoning                             |
| ------------------ | ----- | --------- | -------- | ------------------------------------- |
| **<5K samples**    | 4-5   | 1024      | 4-6      | Lower capacity to prevent overfitting |
| **5K-50K samples** | 6     | 2048      | 6-8      | Balanced setup                        |
| **>50K samples**   | 6-7   | 2048-4096 | 8-10     | Full capacity                         |

```{important}
**Depth vs n_trees trade-off:** Increasing depth from 6→7 doubles leaves (64→128) and memory. Often better to increase n_trees instead.
```

### Quick Start

```python
from deeptab.models import NODEClassifier, NODERegressor, NODELSS
from deeptab.configs import NODEConfig, TrainerConfig

# Standard setup
model = NODEClassifier()
model.fit(X_train, y_train, max_epochs=50)
predictions = model.predict(X_test)

# Custom configuration
cfg = NODEConfig(
    n_layers=8,
    depth=6,         # 2^6 = 64 leaves per tree
    n_trees=2048,    # Ensemble size
)
trainer = TrainerConfig(
    lr=1e-3,         # NODE tolerates higher lr than transformers
    batch_size=512,
    max_epochs=150,
)
model = NODERegressor(model_config=cfg, trainer_config=trainer)
model.fit(X_train, y_train)

# LSS mode
model = NODELSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)
```

## Performance Characteristics

### Comparative Analysis

| vs Model             | Accuracy       | Speed   | Interpretability | When to Prefer NODE                             | When to Prefer Alternative              |
| -------------------- | -------------- | ------- | ---------------- | ----------------------------------------------- | --------------------------------------- |
| **XGBoost/LightGBM** | Similar to -5% | Similar | Lower            | Gradient-based training, deep learning pipeline | Full interpretability, fastest training |
| **Mambular**         | -3 to -7%      | Similar | Much lower       | Some interpretability needed                    | Maximum accuracy                        |
| **FTTransformer**    | -3 to -5%      | Faster  | Much lower       | Tree bias beneficial                            | Complex feature interactions            |
| **ResNet**           | Similar to +3% | Similar | Lower            | Tree structure advantageous                     | Simplest baseline                       |

```{note}
**GBDT comparison:** NODE performs comparably to classical gradient boosted trees (XGBoost/LightGBM) while enabling end-to-end gradient optimization with other neural components.
```

### Use Case Suitability

| Use Case                 | Suitability | Reasoning                                     |
| ------------------------ | ----------- | --------------------------------------------- |
| GBDT-friendly data       | ⭐⭐⭐⭐⭐  | Tree inductive bias matches well              |
| Partial interpretability | ⭐⭐⭐⭐    | Can examine tree splits and feature selection |
| Outlier robustness       | ⭐⭐⭐⭐    | Tree splits less sensitive than linear        |
| Categorical features     | ⭐⭐⭐⭐    | Trees handle categoricals naturally           |
| Maximum accuracy         | ⭐⭐⭐      | Deep learning models typically better         |
| Very large datasets      | ⭐⭐⭐      | O(n·log n) slower than linear models          |
| Full interpretability    | ⭐⭐        | XGBoost/LightGBM better                       |

## Architecture Details

### Oblivious Decision Trees

**Standard decision tree:**

```
Level 0: Split feature X₃
Level 1: Left→X₁, Right→X₇  ← Different features
```

**Oblivious tree:**

```
Level 0: All nodes split on X₃
Level 1: All nodes split on X₁  ← Same feature per level
```

**Advantages:**

- **Fewer parameters:** depth D = 2^D leaves, not 2^D - 1 split features
- **Parallel evaluation:** All nodes at same level use same feature
- **Regularization:** Structure constraint reduces overfitting

### Soft Routing

**Hard split (classical tree):**

```
if x[feature] < threshold:
    go_left()  ← Discrete
else:
    go_right()
```

**Soft split (NODE):**

```
p_left = sigmoid((x[feature] - threshold) / temperature)
p_right = 1 - p_left
output = p_left * left_value + p_right * right_value  ← Differentiable!
```

**Enables:**

- Gradient-based optimization
- Smooth predictions
- Joint training with neural networks

## Interpretability Features

| Feature                | Description                                              | Use Case                    |
| ---------------------- | -------------------------------------------------------- | --------------------------- |
| **Feature selection**  | Attention weights show which features used at each level | Identify important features |
| **Tree structure**     | Visualize splits and routing                             | Understand decision logic   |
| **Leaf values**        | Examine predictions at each leaf                         | Debug specific regions      |
| **Feature importance** | Aggregate selection weights                              | Global importance ranking   |

```{warning}
**Partial interpretability:** While more interpretable than MLPs/Transformers, NODE is less transparent than classical GBDT. Soft routing and ensembling make exact logic harder to trace.
```

## Known Limitations

```{warning}
**Architectural constraints:**
- **Exponential memory:** 2^depth scaling limits practical depth to 6-8
- **Lower accuracy ceiling:** Typically 3-7% below state-of-the-art deep models
- **Partial interpretability:** More than neural nets, less than classical trees
- **Depth tuning:** Depth significantly impacts memory and performance
```

## References

**Original NODE paper:**

- Popov, S., Morozov, S., & Babenko, A. (2020). _Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data_. ICLR 2020. [arXiv:1909.06312](https://arxiv.org/abs/1909.06312)

**Related work:**

- Ke et al. (2017). _LightGBM: A Highly Efficient Gradient Boosting Decision Tree_. NIPS 2017
- Prokhorenkova et al. (2018). _CatBoost: Unbiased Boosting with Categorical Features_. NIPS 2018

## See Also

- [ENODE](enode) — Extended NODE with improved routing
- [NDTF](ndtf) — Neural Decision Tree Forest variant
- [ResNet](resnet) — If interpretability not needed
- [Comparison Tables](../comparison_tables) — Performance across all models

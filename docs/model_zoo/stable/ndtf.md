# NDTF

**Neural Decision Tree Forest** — Differentiable ensemble of decision trees trained end-to-end.

```{tip}
**Architecture highlight:** Combines forest ensemble diversity with gradient-based optimization. O(n·T·d·log d) complexity where T = number of trees. Provides random forest-like benefits (bagging, variance reduction) in fully differentiable form. Best when tree inductive bias helps and ensemble diversity matters.
```

## Architecture Overview

**Core mechanism:** Ensemble of differentiable decision trees  
**Complexity:** O(n·T·d·log d) where T = number of trees  
**Memory:** O(T·d·2^depth) for forest parameters  
**Inductive bias:** Hierarchical splits with ensemble averaging

### Key Components

1. **Multiple decision trees:** Independent trees for diversity
2. **Soft routing:** Probabilistic paths through trees
3. **Ensemble aggregation:** Average or weighted combination
4. **End-to-end training:** All trees trained jointly via backpropagation

**Architecture comparison:**

| Model         | Structure                | Complexity     | Training Method  | Diversity Mechanism |
| ------------- | ------------------------ | -------------- | ---------------- | ------------------- |
| **NDTF**      | Forest ensemble          | O(n·T·d·log d) | Gradient descent | Multiple trees      |
| NODE          | Single or ensemble trees | O(n·d·log d)   | Gradient descent | Single architecture |
| ENODE         | Enhanced trees           | O(n·d·log d)   | Gradient descent | Embeddings          |
| XGBoost       | Boosted trees            | O(n·T·d·log d) | Boosting         | Sequential fitting  |
| Random Forest | Bagged trees             | O(n·T·d·log d) | Greedy splits    | Bootstrap samples   |

```{note}
**Design philosophy:** NDTF brings random forest's ensemble diversity to neural networks. Unlike boosting (sequential), all trees trained in parallel. Unlike bagging, shares gradients across forest. Best of both worlds: ensemble diversity + unified optimization.
```

## When to Use

| Scenario                       | Recommendation                        | Reasoning                                |
| ------------------------------ | ------------------------------------- | ---------------------------------------- |
| **Random forests work well**   | ✅ Use NDTF                           | Neural version maintains forest benefits |
| **Need ensemble diversity**    | ✅ Use NDTF                           | Multiple trees reduce variance           |
| **Tree inductive bias helps**  | ✅ Use NDTF                           | Hierarchical decision boundaries         |
| **Want interpretability**      | ✅ Use NDTF                           | Tree structure interpretable             |
| **Medium datasets (5-20K)**    | ✅ Use NDTF                           | Sweet spot for forest methods            |
| **Tabular with mixed types**   | ✅ Use NDTF                           | Trees handle naturally                   |
| **Trees don't help**           | ❌ Use [Mambular](mambular)           | Different inductive bias                 |
| **Need single-model accuracy** | ❌ Use [Mambular](mambular)           | Better single-model capacity             |
| **Speed critical**             | ❌ Use [ResNet](resnet) or [MLP](mlp) | Simpler, faster                          |
| **Very small datasets (<1K)**  | ❌ Use simpler models                 | Forest complexity risks overfitting      |

## Computational Characteristics

### Complexity Analysis

| Model         | Time Complexity | Number of Trees | Parameters | Memory |
| ------------- | --------------- | --------------- | ---------- | ------ |
| **NDTF**      | O(n·T·d·log d)  | T (parallel)    | ~150K-600K | Medium |
| NODE          | O(n·d·log d)    | 1 or ensemble   | ~100K-400K | Medium |
| ENODE         | O(n·d·log d)    | Ensemble        | ~200K-800K | Medium |
| XGBoost       | O(n·T·d·log d)  | T (sequential)  | N/A        | Low    |
| Random Forest | O(n·T·d·log d)  | T (parallel)    | N/A        | Low    |

### Training Efficiency

| Model         | Training Speed | GPU Utilization | Parallelization       | Best Use Case          |
| ------------- | -------------- | --------------- | --------------------- | ---------------------- |
| **NDTF**      | Moderate       | High            | Full (gradient-based) | Neural forest          |
| NODE          | Moderate-Fast  | High            | Full                  | Single/simple ensemble |
| ENODE         | Moderate       | High            | Full                  | Enhanced features      |
| XGBoost       | Fast (CPU)     | Low             | Limited (boosting)    | Traditional baseline   |
| Random Forest | Fast (CPU)     | Low             | Good (bagging)        | Traditional baseline   |

```{tip}
**Parallelization advantage:** Unlike XGBoost (sequential boosting), NDTF trains all trees in parallel via unified loss. Unlike Random Forest (CPU-bound), NDTF leverages GPU for gradient computation.
```

### Scaling with Number of Trees

| Number of Trees | Training Time | Accuracy Improvement | Diminishing Returns? |
| --------------- | ------------- | -------------------- | -------------------- |
| 2-4             | Fast          | Baseline             | No                   |
| 4-8             | Moderate      | +2-3%                | No                   |
| 8-16            | Moderate-Slow | +1-2%                | Starting             |
| 16-32           | Slow          | +0.5-1%              | Yes                  |

## Configuration Guidelines

### Model Config (NDTFConfig)

```{note}
**Key parameters:** `n_ensembles` controls number of trees (more = diversity but slower), `max_depth` controls tree depth (deeper = more complex boundaries), `d_model` affects embedding dimension if used. Trees grow exponentially with depth (2^depth leaves).
```

| Parameter         | Default    | Typical Range | Description                | Impact                    |
| ----------------- | ---------- | ------------- | -------------------------- | ------------------------- |
| `n_ensembles`     | 8          | 4-16          | Number of trees            | High - diversity vs speed |
| `max_depth`       | 6          | 4-8           | Tree depth                 | High - complexity         |
| `d_model`         | 64         | 32-128        | Embedding/hidden dimension | Moderate - capacity       |
| `dropout`         | 0.0        | 0.0-0.2       | Dropout rate               | Dataset-dependent         |
| `choice_function` | "entmax15" | Various       | Routing sparsity           | Moderate                  |

### Parameter Impact Analysis

| Parameter Change     | Effect on Model     | Effect on Performance     | When to Adjust           |
| -------------------- | ------------------- | ------------------------- | ------------------------ |
| Increase n_ensembles | More trees, slower  | Better variance reduction | Noisy data, have compute |
| Increase max_depth   | Deeper trees        | More complex boundaries   | Complex decision regions |
| Increase d_model     | Larger embeddings   | Higher capacity           | Rich features            |
| Increase dropout     | More regularization | Reduces overfitting       | Small datasets           |

### Recommended Settings by Dataset Size

| Dataset Size        | n_ensembles | max_depth | d_model | dropout | batch_size | Reasoning                           |
| ------------------- | ----------- | --------- | ------- | ------- | ---------- | ----------------------------------- |
| **<1K samples**     | 4           | 4-5       | 32-64   | 0.1-0.2 | 64         | Minimal forest prevents overfitting |
| **1K-5K samples**   | 8           | 5-6       | 64      | 0.1     | 128        | Balanced ensemble                   |
| **5K-10K samples**  | 8-12        | 6         | 64-128  | 0.0-0.1 | 256        | Full forest justified               |
| **10K-20K samples** | 12-16       | 6-7       | 128     | 0.0     | 512        | Large ensemble beneficial           |
| **>20K samples**    | 16          | 6-8       | 128     | 0.0     | 512        | Maximum ensemble                    |

### Quick Start

```python
from deeptab.models import NDTFClassifier, NDTFRegressor, NDTFLSS
from deeptab.configs import NDTFConfig, TrainerConfig

# Fast baseline with defaults
model = NDTFClassifier()
model.fit(X_train, y_train, max_epochs=50)
predictions = model.predict(X_test)

# Custom configuration for forest ensemble
cfg = NDTFConfig(
    n_ensembles=8,  # Number of trees
    max_depth=6,    # Tree depth
    d_model=64,
)
trainer = TrainerConfig(
    lr=5e-4,
    batch_size=256,
    max_epochs=100,
)
model = NDTFRegressor(model_config=cfg, trainer_config=trainer)
model.fit(X_train, y_train)

# Compare with Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=8, max_depth=6)
rf.fit(X_train, y_train)
# NDTF typically competitive, sometimes better via gradient optimization

# LSS mode for distributional regression
model = NDTFLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)
```

## Performance Characteristics

### Comparative Analysis

| vs Model          | Accuracy Gap       | Speed Comparison    | Memory  | When to Prefer NDTF            | When to Prefer Alternative  |
| ----------------- | ------------------ | ------------------- | ------- | ------------------------------ | --------------------------- |
| **XGBoost**       | -3 to +3% (varies) | Slower (GPU vs CPU) | Higher  | Neural approach, GPU available | CPU-only, fastest training  |
| **Random Forest** | Similar to +3%     | Slower              | Higher  | Gradient optimization benefit  | CPU-only, fast training     |
| **NODE**          | +1 to +3%          | Slower (more trees) | Higher  | Forest diversity matters       | Single model sufficient     |
| **ENODE**         | -2 to +2%          | Similar             | Similar | Forest structure preference    | Feature embeddings priority |
| **Mambular**      | -3 to -7%          | Similar             | Lower   | Tree inductive bias            | General purpose             |

```{note}
**Performance profile:** NDTF excels when random forests competitive but want gradient-based optimization. Ensemble diversity reduces variance on noisy datasets. Typical performance: competitive with traditional forests, occasionally better via unified optimization.
```

### When Each Model Wins

| Scenario                  | Best Model             | Why                            |
| ------------------------- | ---------------------- | ------------------------------ |
| Trees + diversity matter  | **NDTF**               | Forest ensemble in neural form |
| CPU-only environment      | XGBoost, Random Forest | Optimized for CPU              |
| GPU available, trees help | **NDTF**               | GPU-accelerated trees          |
| Need interpretability     | XGBoost                | Clearer tree visualization     |
| General purpose           | Mambular               | Typically best overall         |

### Use Case Suitability

| Use Case                   | Suitability | Reasoning                         |
| -------------------------- | ----------- | --------------------------------- |
| Random forests competitive | ⭐⭐⭐⭐⭐  | Neural version of proven approach |
| Tree inductive bias helps  | ⭐⭐⭐⭐⭐  | Hierarchical decision boundaries  |
| Need ensemble diversity    | ⭐⭐⭐⭐⭐  | Multiple trees reduce variance    |
| GPU available              | ⭐⭐⭐⭐    | Leverages parallel training       |
| Interpretability matters   | ⭐⭐⭐⭐    | Tree structure interpretable      |
| Medium datasets (5-20K)    | ⭐⭐⭐⭐    | Sweet spot                        |
| Large datasets (>20K)      | ⭐⭐⭐      | Consider Mambular                 |
| Trees don't help           | ⭐⭐        | Try different architecture        |

## Architecture Details

### Forest Ensemble Structure

**Traditional Random Forest:**

```
Bootstrap sample 1 → Tree 1 ┐
Bootstrap sample 2 → Tree 2 ├→ Vote/Average → Prediction
...                         │
Bootstrap sample T → Tree T ┘
```

**NDTF (Neural Decision Tree Forest):**

```
Input → Tree 1 (soft routing) ┐
     → Tree 2 (soft routing) ├→ Average → Prediction
     ...                     │
     → Tree T (soft routing) ┘
     ↓ all share gradients
Unified loss → backpropagation
```

**Key differences:**

| Aspect            | Random Forest        | NDTF                   |
| ----------------- | -------------------- | ---------------------- |
| **Tree training** | Independent (greedy) | Joint (gradient-based) |
| **Data per tree** | Bootstrap sample     | Full dataset           |
| **Routing**       | Hard (discrete)      | Soft (probabilistic)   |
| **Optimization**  | Greedy splits        | Backpropagation        |
| **Hardware**      | CPU                  | GPU                    |

### Differentiable Trees

**Hard routing (traditional):**

```
Sample x → Decision node
         → Go left OR right (binary)
         → Leaf with prediction
```

**Soft routing (NDTF):**

```
Sample x → Decision node
         → Probability of left: p
         → Probability of right: 1-p
         → Weighted combination of both paths
```

**Mathematical formulation:**

For tree with depth $D$:

$$
P(\text{leaf}_l | \mathbf{x}) = \prod_{d \in \text{path}_l} p_d(\mathbf{x})
$$

Where $p_d(\mathbf{x})$ is probability of taking decision at depth $d$.

**Tree prediction:**

$$
\hat{y}_t = \sum_{l=1}^{2^D} P(\text{leaf}_l | \mathbf{x}) \cdot w_{l,t}
$$

**Forest prediction:**

$$
\hat{y} = \frac{1}{T} \sum_{t=1}^{T} \hat{y}_t
$$

### Full Architecture

```
Input features x ∈ ℝᵈ
        ↓
Optional embedding
   x → e ∈ ℝ^(d_model)
        ↓
┌─────────────────────┐
│ Tree 1              │
│ Soft routing        │
│ Probabilistic paths │
│ → prediction₁       │
└─────────────────────┘
┌─────────────────────┐
│ Tree 2              │
│ Soft routing        │
│ → prediction₂       │
└─────────────────────┘
        ...
┌─────────────────────┐
│ Tree T              │
│ → predictionₜ       │
└─────────────────────┘
        ↓
Ensemble average
  (prediction₁ + ... + predictionₜ) / T
        ↓
Final prediction
```

### Diversity Mechanisms

**How NDTF creates diverse trees:**

1. **Random initialization:** Each tree starts with different weights
2. **Gradient noise:** Stochastic optimization creates variation
3. **Different update paths:** Each tree sees different gradients
4. **Regularization:** Dropout, weight decay differ across trees

**Unlike Random Forest:**

- No bootstrap sampling (all trees see all data)
- Diversity from optimization dynamics, not data subsampling

## Known Limitations

```{warning}
**Constraints and trade-offs:**
- **Training time:** Scales linearly with number of trees
- **Memory:** Multiple trees increase parameter count
- **Not always better:** If trees don't help, forest won't either
- **Hyperparameter sensitivity:** Must tune n_ensembles, max_depth
- **Less interpretable than XGBoost:** Soft routing harder to visualize
- **GPU dependency:** Best performance requires GPU
```

**When limitations matter:**

- Speed critical → Use NODE (fewer trees) or traditional ML
- Trees don't help → Use Mambular or ResNet
- CPU-only environment → Use XGBoost or Random Forest
- Need clear interpretability → Use XGBoost with SHAP
- Very small datasets (<1K) → Simpler models better

## Ensemble Analysis

```{tip}
**Examining tree diversity:** Check prediction variance across trees to assess ensemble quality. High variance = good diversity. Can also compare individual tree accuracies.
```

**Analyzing ensemble:**

```python
# After training
model = NDTFClassifier()
model.fit(X_train, y_train, max_epochs=50)

# Get predictions from individual trees (requires model internals)
# tree_predictions = [tree_i.predict(X_test) for each tree]
# ensemble_prediction = mean(tree_predictions)

# Measure diversity: variance across tree predictions
# High variance = diverse ensemble (good)
```

## Comparison with Traditional Forests

| Aspect                           | Random Forest      | XGBoost               | NDTF                |
| -------------------------------- | ------------------ | --------------------- | ------------------- |
| **Training**                     | Parallel (bagging) | Sequential (boosting) | Parallel (gradient) |
| **Optimization**                 | Greedy             | Greedy + boosting     | Gradient descent    |
| **Hardware**                     | CPU                | CPU                   | GPU                 |
| **Routing**                      | Hard               | Hard                  | Soft                |
| **Differentable**                | No                 | No                    | Yes                 |
| **Integration with neural nets** | Hard               | Hard                  | Easy                |

## References

**Neural decision trees:**

- Kontschieder, P., Fiterau, M., Criminisi, A., & Rota Bulò, S. (2015). _Deep Neural Decision Forests_. ICCV 2015

**Related tree ensembles:**

- Breiman, L. (2001). _Random Forests_. Machine Learning, 45(1). (Foundation for random forests)
- Chen, T., & Guestrin, C. (2016). _XGBoost: A Scalable Tree Boosting System_. KDD 2016

**Differentiable tree approaches:**

- Various implementations of soft decision trees and neural decision forests

## See Also

- [NODE](node) — Single tree architecture
- [ENODE](enode) — Enhanced NODE with embeddings
- [XGBoost Guide](../../tutorials/comparing_with_gbdt) — Traditional GBDT baseline
- [Random Forest Tutorial](../../tutorials/tree_based_methods) — Classical forests
- [Comparison Tables](../comparison_tables) — Performance across all models

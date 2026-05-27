# ENODE

**Extended Neural Oblivious Decision Ensembles** — Enhanced NODE with feature embeddings and improved routing.

```{tip}
**Architecture highlight:** Extends NODE with learned feature embeddings for richer representations. O(n·d·log d) tree-based routing with embedding enhancement. Trades ~20% slower training for 2-5% accuracy gain over NODE. Best when tree inductive bias helps but feature representation matters.
```

## Architecture Overview

**Core mechanism:** Oblivious decision trees with feature embedding layer  
**Complexity:** O(n·d·log d) per forward pass (tree depth logarithmic)  
**Memory:** O(d·2^depth) for tree parameters  
**Inductive bias:** Hierarchical decision boundaries with rich feature space

### Key Components

1. **Feature embedding:** Maps inputs to learned representation space
2. **Oblivious decision trees:** All nodes at same depth use same feature split
3. **Ensemble of trees:** Multiple trees for robustness
4. **Routing probabilities:** Soft routing through tree paths

**Architecture comparison:**

| Model     | Feature Processing | Complexity   | Interpretability | Training Speed |
| --------- | ------------------ | ------------ | ---------------- | -------------- |
| **ENODE** | Learned embeddings | O(n·d·log d) | Good             | Moderate       |
| NODE      | Direct features    | O(n·d·log d) | Better           | Faster (~1.2x) |
| NDTF      | Forest ensemble    | O(n·d·log d) | Good             | Similar        |
| Mambular  | Sequential SSM     | O(n·d)       | Lower            | Similar        |

```{note}
**Design trade-off:** ENODE adds embedding layer to NODE, enabling richer feature representations at cost of additional parameters and slower training. Worth it when feature quality limits NODE performance.
```

## When to Use

| Scenario                            | Recommendation                                  | Reasoning                                         |
| ----------------------------------- | ----------------------------------------------- | ------------------------------------------------- |
| **NODE works but plateaus**         | ✅ Use ENODE                                    | Embedding layer can unlock additional capacity    |
| **Tree-based inductive bias helps** | ✅ Use ENODE                                    | Retains tree structure with better features       |
| **Mixed feature types**             | ✅ Use ENODE                                    | Embeddings unify categorical + numerical          |
| **Interpretability matters**        | ✅ Use ENODE                                    | Tree routing interpretable, better than black-box |
| **Medium datasets (5-20K)**         | ✅ Use ENODE                                    | Sweet spot for embedding benefit                  |
| **Random forests competitive**      | ✅ Try ENODE                                    | Neural version may improve further                |
| **NODE doesn't help**               | ❌ Use [Mambular](mambular) or [ResNet](resnet) | Tree bias not helping your data                   |
| **Speed critical**                  | ❌ Use [NODE](node)                             | Faster with ~2-3% less accuracy                   |
| **Very small datasets (<1K)**       | ❌ Use [NODE](node)                             | Embeddings add parameters, overfitting risk       |
| **Maximum accuracy**                | ❌ Use [Mambular](mambular)                     | Typically 3-7% better                             |

## Computational Characteristics

### Complexity Analysis

| Model     | Time Complexity | Tree Operations | Parameters | Memory     |
| --------- | --------------- | --------------- | ---------- | ---------- |
| **ENODE** | O(n·d·log d)    | Soft routing    | ~200K-800K | Medium     |
| NODE      | O(n·d·log d)    | Soft routing    | ~100K-400K | Medium     |
| NDTF      | O(n·d·log d)    | Forest routing  | ~150K-600K | Medium     |
| XGBoost   | O(n·d·log d)    | Hard routing    | N/A        | Low        |
| Mambular  | O(n·d)          | No trees        | ~100K-500K | Low-Medium |

### Training Efficiency

| Model     | Relative Training Speed | GPU Memory | Interpretability | Best Use Case        |
| --------- | ----------------------- | ---------- | ---------------- | -------------------- |
| **ENODE** | Baseline (moderate)     | Medium     | Good             | NODE + feature boost |
| NODE      | ~1.2x faster            | Medium     | Better           | Faster tree baseline |
| NDTF      | Similar                 | Medium     | Good             | Forest ensemble      |
| XGBoost   | Much faster (CPU)       | Low        | Best             | Traditional baseline |
| Mambular  | Similar                 | Low-Medium | Lower            | General purpose      |

```{tip}
**Speed-accuracy trade-off:** ENODE trains ~20% slower than NODE but typically gains 2-5% accuracy. Worth it for production where accuracy matters more than training time.
```

### Capacity vs Speed Trade-off

| Model     | Parameters       | Typical Accuracy (relative) | Training Time (relative) | When to Prefer        |
| --------- | ---------------- | --------------------------- | ------------------------ | --------------------- |
| NODE      | 100% (reference) | 100% (reference)            | 1.0x                     | Speed > accuracy      |
| **ENODE** | ~150%            | 102-105%                    | 1.2x                     | Accuracy > speed      |
| NDTF      | ~120%            | 100-103%                    | 1.1x                     | Forest inductive bias |

## Configuration Guidelines

### Model Config (ENODEConfig)

```{note}
**Key parameters:** `d_model` controls embedding richness (larger = more capacity), `n_layers` is number of trees in ensemble, `depth` controls tree depth (deeper = more complex boundaries). Tree parameters grow exponentially with depth (2^depth leaves).
```

| Parameter         | Default    | Typical Range | Description         | Impact                                |
| ----------------- | ---------- | ------------- | ------------------- | ------------------------------------- |
| `d_model`         | 128        | 64-256        | Embedding dimension | High - feature representation quality |
| `n_layers`        | 8          | 4-16          | Number of trees     | High - ensemble diversity             |
| `depth`           | 6          | 4-8           | Tree depth          | High - decision boundary complexity   |
| `dropout`         | 0.0        | 0.0-0.2       | Dropout rate        | Dataset-dependent                     |
| `choice_function` | "entmax15" | Various       | Routing function    | Moderate - sparsity control           |

### Parameter Impact Analysis

| Parameter Change  | Effect on Model                    | Effect on Performance   | When to Adjust                 |
| ----------------- | ---------------------------------- | ----------------------- | ------------------------------ |
| Increase d_model  | Richer embeddings, more parameters | Higher capacity, slower | Features complex, have compute |
| Increase n_layers | More trees, more parameters        | Better ensemble, slower | Variance reduction needed      |
| Increase depth    | Deeper trees, exponential growth   | More complex boundaries | Decision boundaries complex    |
| Increase dropout  | More regularization                | Reduces overfitting     | Small datasets                 |

### Recommended Settings by Dataset Size

| Dataset Size        | d_model  | n_layers | depth | dropout | batch_size | Reasoning                 |
| ------------------- | -------- | -------- | ----- | ------- | ---------- | ------------------------- |
| **<1K samples**     | Use NODE | -        | -     | -       | -          | Embeddings add complexity |
| **1K-5K samples**   | 64-128   | 4-8      | 5-6   | 0.1-0.2 | 128        | Conservative capacity     |
| **5K-10K samples**  | 128      | 8-12     | 6     | 0.0-0.1 | 256        | Balanced settings         |
| **10K-20K samples** | 128-192  | 8-16     | 6-7   | 0.0     | 512        | Full capacity justified   |
| **>20K samples**    | 192-256  | 12-16    | 6-8   | 0.0     | 512        | Maximum capacity          |

### Quick Start

```python
from deeptab.models import ENODEClassifier, ENODERegressor, ENODELSS
from deeptab.configs import ENODEConfig, TrainerConfig

# Fast baseline with defaults
model = ENODEClassifier()
model.fit(X_train, y_train, max_epochs=50)
predictions = model.predict(X_test)

# Custom configuration for better accuracy
cfg = ENODEConfig(
    d_model=128,
    n_layers=8,
    depth=6,
)
trainer = TrainerConfig(
    lr=5e-4,
    batch_size=256,
    max_epochs=100,
)
model = ENODERegressor(model_config=cfg, trainer_config=trainer)
model.fit(X_train, y_train)

# Compare with NODE baseline
from deeptab.models import NODEClassifier
node_model = NODEClassifier()
node_model.fit(X_train, y_train, max_epochs=50)
# ENODE typically 2-5% better, 20% slower training

# LSS mode for distributional regression
model = ENODELSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)
```

## Performance Characteristics

### Comparative Analysis

| vs Model     | Accuracy Gap       | Speed Advantage   | Memory  | When to Prefer ENODE    | When to Prefer Alternative |
| ------------ | ------------------ | ----------------- | ------- | ----------------------- | -------------------------- |
| **NODE**     | +2 to +5%          | 0.8x (20% slower) | Higher  | Accuracy matters        | Speed critical             |
| **NDTF**     | Similar to +2%     | Similar           | Similar | Feature embeddings help | Forest diversity priority  |
| **XGBoost**  | -5 to +5% (varies) | Much slower       | Higher  | Neural approach needed  | Traditional ML sufficient  |
| **Mambular** | -3 to -7%          | Similar           | Lower   | Tree inductive bias     | General purpose            |
| **ResNet**   | Similar to +3%     | Slightly slower   | Similar | Tree interpretability   | Fast baseline              |

```{note}
**Performance profile:** ENODE performs best when NODE shows promise but accuracy plateaus. Embedding layer helps with complex feature interactions and mixed data types. Typical gain: 2-5% over NODE, but requires 20% longer training.
```

### When Each Model Wins

| Scenario                  | Best Model | Why                       |
| ------------------------- | ---------- | ------------------------- |
| Trees help, need accuracy | **ENODE**  | Best of tree-based neural |
| Trees help, need speed    | NODE       | Faster tree baseline      |
| Need forest diversity     | NDTF       | Explicit forest structure |
| General purpose           | Mambular   | Typically best overall    |
| Traditional ML sufficient | XGBoost    | Fast, interpretable       |

### Use Case Suitability

| Use Case                   | Suitability | Reasoning                        |
| -------------------------- | ----------- | -------------------------------- |
| NODE promising but limited | ⭐⭐⭐⭐⭐  | Designed for this                |
| Tree inductive bias helps  | ⭐⭐⭐⭐⭐  | Enhanced tree structure          |
| Interpretability important | ⭐⭐⭐⭐    | Tree routing interpretable       |
| Mixed feature types        | ⭐⭐⭐⭐    | Embeddings unify representations |
| Medium datasets (5-20K)    | ⭐⭐⭐⭐    | Sweet spot                       |
| Large datasets (>20K)      | ⭐⭐⭐      | Consider Mambular                |
| Speed critical             | ⭐⭐        | Use NODE instead                 |
| Trees don't help           | ⭐⭐        | Try different architecture       |

## Architecture Details

### Oblivious Decision Trees

**Oblivious property:** All nodes at same depth use the same feature for splitting

**Standard tree:**

```
        [Feature 3]
       /           \
  [Feature 1]   [Feature 7]
   /      \      /      \
Leaf1  Leaf2  Leaf3  Leaf4
```

**Oblivious tree:**

```
        [Feature 3]
       /           \
  [Feature 1]   [Feature 1]  ← Same feature!
   /      \      /      \
Leaf1  Leaf2  Leaf3  Leaf4
```

**Advantages:**

- Fewer parameters (one split per depth level)
- Better generalization
- Faster evaluation
- Still expressively powerful

### ENODE Enhancement

**NODE flow:**

```
Input → Trees → Ensemble → Output
```

**ENODE flow:**

```
Input → Embeddings → Trees → Ensemble → Output
        ↓ learned    ↓ oblivious
        ↓ features   ↓ structure
```

**Embedding benefit:**

| Aspect                   | NODE (Direct) | ENODE (Embedded)      | Advantage             |
| ------------------------ | ------------- | --------------------- | --------------------- |
| **Categorical features** | One-hot       | Dense embedding       | More efficient        |
| **Numerical features**   | As-is         | Learned transform     | Better representation |
| **Feature interactions** | None          | Implicit in embedding | Captures dependencies |
| **Mixed data**           | Inconsistent  | Unified space         | Better integration    |

### Mathematical Formulation

**Input:** $\mathbf{x} \in \mathbb{R}^d$ (features)

**Step 1: Embedding**

$$
\mathbf{e} = \text{Embed}(\mathbf{x}) \in \mathbb{R}^{d_{\text{model}}}
$$

**Step 2: Tree routing (per tree)**

For depth $D$ oblivious tree:

$$
P(\text{leaf}_l | \mathbf{e}) = \prod_{d=1}^{D} P(\text{decision}_d | \mathbf{e})
$$

Where decisions are soft (probabilistic):

$$
P(\text{left}_d | \mathbf{e}) = \sigma(f_d(\mathbf{e}))
$$

**Step 3: Ensemble prediction**

$$
\hat{y} = \frac{1}{L} \sum_{t=1}^{L} \sum_{l=1}^{2^D} P(\text{leaf}_l^{(t)} | \mathbf{e}) \cdot w_l^{(t)}
$$

Where $L$ = n_layers (number of trees), $w_l^{(t)}$ = learned leaf weights.

### Full Architecture

```
Input features x ∈ ℝᵈ
        ↓
Embedding network
   x → e ∈ ℝ^(d_model)
        ↓
╔═══════════════════════════════╗
║ Tree 1                        ║
║ Depth 1: Feature selection    ║
║ Depth 2: Feature selection    ║
║ ...                           ║
║ Depth D: Leaf probabilities   ║
║ → prediction₁                 ║
╚═══════════════════════════════╝
        ↓
╔═══════════════════════════════╗
║ Tree 2 (similar structure)    ║
║ → prediction₂                 ║
╚═══════════════════════════════╝
        ↓
    ... (L trees total)
        ↓
Ensemble average
        ↓
Final prediction
```

## Known Limitations

```{warning}
**Constraints and trade-offs:**
- **Training speed:** 20% slower than NODE due to embedding layer
- **Parameter count:** ~50% more parameters than NODE
- **Small datasets:** Embedding overhead risks overfitting with <1K samples
- **Not always better:** If NODE doesn't help, ENODE won't either
- **Interpretability trade-off:** Embeddings reduce interpretability vs NODE
- **Hyperparameter sensitivity:** More parameters to tune than NODE
```

**When limitations matter:**

- Speed critical → Use NODE (similar accuracy, faster)
- Very small data (<1K) → Use NODE or simpler models
- Trees don't help your data → Try Mambular or ResNet
- Need maximum interpretability → Use NODE or XGBoost
- Limited compute → NODE more efficient

## Progression Path

```{tip}
**Recommended workflow:** Start with NODE for fast baseline, upgrade to ENODE if accuracy matters and compute allows, consider Mambular if trees don't help.
```

**Decision tree:**

```
Random forests competitive?
    ↓ No → Try Mambular, ResNet
    ↓ Yes
Try NODE first (fast baseline)
    ↓
NODE promising?
    ↓ No → Try other architectures
    ↓ Yes
Need more accuracy and have compute?
    ↓ No → Stay with NODE
    ↓ Yes
→ Use ENODE (2-5% better)
```

## Interpretability Analysis

**Tree routing can be visualized:**

```python
# After training
model = ENODEClassifier()
model.fit(X_train, y_train, max_epochs=50)

# Examine tree structure (requires model internals)
# Each tree shows which features split at each depth
# Routing probabilities show sample paths through tree
```

**Interpretability comparison:**

| Model     | Interpretability | Method                    |
| --------- | ---------------- | ------------------------- |
| XGBoost   | ⭐⭐⭐⭐⭐       | Direct tree visualization |
| NODE      | ⭐⭐⭐⭐         | Soft tree routing         |
| **ENODE** | ⭐⭐⭐           | Soft routing + embeddings |
| NDTF      | ⭐⭐⭐           | Forest routing            |
| Mambular  | ⭐⭐             | Feature importance only   |

## References

**NODE foundation:**

- Popov, S., Morozov, S., & Babenko, A. (2019). _Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data_. ICLR 2020. arXiv:1909.06312

**ENODE enhancement:**

- Extended with feature embeddings for improved representation learning
- Combines NODE's tree structure with embedding networks

**Related tree-based neural architectures:**

- Kontschieder, P., et al. (2015). _Deep Neural Decision Forests_. ICCV 2015

## See Also

- [NODE](node) — Original architecture without embeddings
- [NDTF](ndtf) — Forest variant
- [Mambular](mambular) — General-purpose alternative
- [XGBoost Guide](../../tutorials/comparing_with_gbdt) — Traditional baseline
- [Comparison Tables](../comparison_tables) — Performance across all models

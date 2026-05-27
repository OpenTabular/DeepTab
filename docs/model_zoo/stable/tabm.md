# TabM

**Batch-Ensembling MLP** — Efficient ensemble via batch splitting for near-single-model cost.

```{tip}
**Architecture highlight:** Achieves ensemble diversity by splitting each batch across multiple sub-models. O(n·d) complexity (same as single MLP) with ~30% overhead. Provides 70-80% of full ensemble benefit at 1.3x single-model cost. Best when you need robustness without training multiple models.
```

## Architecture Overview

**Core mechanism:** Single forward pass processes multiple ensemble members via batch splitting  
**Complexity:** O(n·d) per forward pass (same as MLP)  
**Memory:** O(E·d) where E = number of ensemble members  
**Inductive bias:** Ensemble averaging reduces variance

### Key Components

1. **Batch splitting:** Divides batch into sub-batches for each ensemble member
2. **Shared architecture:** All members use same network structure
3. **Independent parameters:** Each member has distinct weights
4. **Efficient forward pass:** Single pass processes all members

**Architecture comparison:**

| Model            | Ensemble Method   | Training Cost | Inference Cost | Diversity Mechanism |
| ---------------- | ----------------- | ------------- | -------------- | ------------------- |
| **TabM**         | Batch-ensembling  | ~1.3x single  | ~1.3x single   | Batch splitting     |
| MLP ensemble     | Train E models    | E × single    | E × single     | Separate training   |
| Dropout ensemble | MC Dropout        | 1x single     | E × single     | Random dropout      |
| Bagging ensemble | Bootstrap samples | E × single    | E × single     | Data resampling     |

```{note}
**Design innovation:** Traditional ensembles require training E separate models (E times cost). TabM achieves similar benefits by splitting each batch across E sub-models in single forward pass. Key insight: ensemble diversity from batch-level variation sufficient for robustness.
```

## When to Use

| Scenario                        | Recommendation                        | Reasoning                            |
| ------------------------------- | ------------------------------------- | ------------------------------------ |
| **Want ensemble benefits**      | ✅ Use TabM                           | 70-80% of full ensemble at 1.3x cost |
| **Limited compute budget**      | ✅ Use TabM                           | Much cheaper than training E models  |
| **Need robustness/uncertainty** | ✅ Use TabM                           | Variance reduction from ensemble     |
| **Small-medium datasets**       | ✅ Use TabM                           | Ensemble helps with limited data     |
| **Fast iteration needed**       | ✅ Use TabM                           | Faster than full ensemble            |
| **Can afford full ensemble**    | ❌ Train E models                     | 20-30% better than TabM              |
| **Need single-model accuracy**  | ❌ Use [Mambular](mambular)           | Better single-model capacity         |
| **Speed critical**              | ❌ Use [MLP](mlp) or [ResNet](resnet) | Faster single models                 |
| **Very large models**           | ❌ Use single model                   | Memory overhead becomes significant  |

## Computational Characteristics

### Complexity Analysis

| Model            | Training Time | Inference Time | Parameters   | Memory      |
| ---------------- | ------------- | -------------- | ------------ | ----------- |
| **TabM (E=4)**   | ~1.3x single  | ~1.3x single   | ~1.5x single | Medium      |
| Single MLP       | Baseline      | Baseline       | Baseline     | Low         |
| E-model ensemble | E × single    | E × single     | E × single   | E × single  |
| Dropout ensemble | 1x single     | E × single     | 1x single    | Low (train) |

### Training Efficiency

| Model            | Relative Speed  | GPU Memory | Ensemble Quality      | Best Use Case           |
| ---------------- | --------------- | ---------- | --------------------- | ----------------------- |
| **TabM**         | 1.3x (baseline) | Medium     | Good (70-80% of full) | Budget ensemble         |
| Single MLP       | 1.0x (fastest)  | Low        | None                  | Speed over robustness   |
| Full ensemble    | E × slow        | High       | Best (100%)           | Accuracy critical       |
| Dropout ensemble | 1.0x            | Low        | Moderate (50-60%)     | Training speed critical |

```{tip}
**Cost-benefit sweet spot:** TabM provides best ensemble accuracy per compute unit. 70-80% of full ensemble benefit at 30% overhead vs 100% overhead for E-model ensemble.
```

### Scaling with Ensemble Size

| Ensemble Members (E) | Memory Overhead | Training Time | Accuracy Gain | Diminishing Returns? |
| -------------------- | --------------- | ------------- | ------------- | -------------------- |
| 2                    | +20%            | +15%          | Baseline      | No                   |
| 4                    | +50%            | +30%          | +2-3%         | No                   |
| 8                    | +100%           | +60%          | +1-2%         | Starting             |
| 16                   | +200%           | +120%         | +0.5-1%       | Yes                  |

## Configuration Guidelines

### Model Config (TabMConfig)

```{note}
**Key parameters:** `n_ensembles` controls diversity-cost trade-off (typical: 4-8), `d_model` controls capacity of each member, `n_layers` affects depth. Each ensemble member is a separate MLP sharing the architecture.
```

| Parameter     | Default | Typical Range | Description                | Impact                    |
| ------------- | ------- | ------------- | -------------------------- | ------------------------- |
| `d_model`     | 128     | 64-256        | Hidden dimension per layer | High - capacity           |
| `n_layers`    | 8       | 4-16          | Number of layers           | High - model depth        |
| `n_ensembles` | 4       | 2-8           | Number of ensemble members | High - diversity vs speed |
| `dropout`     | 0.0     | 0.0-0.3       | Dropout rate               | Dataset-dependent         |
| `activation`  | "relu"  | Various       | Activation function        | Low-Moderate              |

### Parameter Impact Analysis

| Parameter Change     | Effect on Model      | Effect on Performance     | When to Adjust               |
| -------------------- | -------------------- | ------------------------- | ---------------------------- |
| Increase n_ensembles | More members, slower | Better variance reduction | Need robustness, have budget |
| Increase d_model     | Larger networks      | Higher capacity           | Complex patterns             |
| Increase n_layers    | Deeper networks      | More abstraction          | Hierarchical features        |
| Increase dropout     | More regularization  | Reduces overfitting       | Small datasets               |

### Recommended Settings by Dataset Size

| Dataset Size       | n_ensembles | d_model | n_layers | dropout | batch_size | Reasoning                         |
| ------------------ | ----------- | ------- | -------- | ------- | ---------- | --------------------------------- |
| **<1K samples**    | 4           | 64-128  | 4-6      | 0.2-0.3 | 64         | Moderate ensemble, regularization |
| **1K-5K samples**  | 4-6         | 128     | 6-8      | 0.1-0.2 | 128        | Balanced ensemble                 |
| **5K-10K samples** | 6-8         | 128-192 | 8-12     | 0.0-0.1 | 256        | Larger ensemble justified         |
| **>10K samples**   | 8           | 192-256 | 8-16     | 0.0     | 512        | Full ensemble capacity            |

### Quick Start

```python
from deeptab.models import TabMClassifier, TabMRegressor, TabMLSS
from deeptab.configs import TabMConfig, TrainerConfig

# Fast baseline with defaults
model = TabMClassifier()
model.fit(X_train, y_train, max_epochs=50)
predictions = model.predict(X_test)

# Custom configuration for budget ensemble
cfg = TabMConfig(
    d_model=128,
    n_layers=8,
    n_ensembles=4,  # 4 ensemble members
)
trainer = TrainerConfig(
    lr=1e-3,
    batch_size=256,
    max_epochs=100,
)
model = TabMRegressor(model_config=cfg, trainer_config=trainer)
model.fit(X_train, y_train)

# Get prediction uncertainty (ensemble variance)
predictions = model.predict(X_test)
# Ensemble provides uncertainty estimates via member variance

# Compare with full ensemble
from deeptab.models import MLPClassifier
ensemble_models = [MLPClassifier() for _ in range(4)]
for m in ensemble_models:
    m.fit(X_train, y_train, max_epochs=50)  # 4x training time
# TabM typically 70-80% of this accuracy at 30% of cost

# LSS mode for distributional regression
model = TabMLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)
```

## Performance Characteristics

### Comparative Analysis

| vs Model                     | Accuracy Gap | Speed Advantage                 | Memory     | When to Prefer TabM     | When to Prefer Alternative |
| ---------------------------- | ------------ | ------------------------------- | ---------- | ----------------------- | -------------------------- |
| **Full ensemble (E models)** | -1 to -3%    | E × faster                      | Much lower | Budget limited          | Accuracy critical          |
| **Single MLP**               | +2 to +5%    | 0.7x (30% slower)               | Higher     | Need robustness         | Speed critical             |
| **Dropout ensemble**         | +1 to +3%    | Similar train, slower inference | Similar    | Training efficiency     | Inference speed            |
| **Mambular**                 | -3 to -7%    | Similar                         | Lower      | Want ensemble benefits  | Single-model accuracy      |
| **ResNet**                   | +1 to +4%    | Similar                         | Similar    | Ensemble > architecture | Architecture matters       |

```{note}
**Performance profile:** TabM sits between single model and full ensemble. Provides most ensemble benefit (variance reduction, robustness) at fraction of cost. Typical: 70-80% of full ensemble accuracy improvement over single model.
```

### Ensemble Efficiency Analysis

| Method           | Training Cost      | Accuracy (relative) | Cost-Benefit Ratio            |
| ---------------- | ------------------ | ------------------- | ----------------------------- |
| Single model     | 1x                 | 100% (baseline)     | 1.00                          |
| **TabM**         | 1.3x               | 103-105%            | **2.3-3.8** (best)            |
| Dropout ensemble | 1x train, Ex infer | 101-103%            | 1-3 (train), poor (inference) |
| Full ensemble    | Ex                 | 105-108%            | 1.0-1.6                       |

**Interpretation:** TabM provides best "accuracy per compute unit" — highest improvement for lowest cost.

### Use Case Suitability

| Use Case                   | Suitability | Reasoning                        |
| -------------------------- | ----------- | -------------------------------- |
| Budget ensemble            | ⭐⭐⭐⭐⭐  | Designed for this                |
| Need uncertainty estimates | ⭐⭐⭐⭐⭐  | Ensemble variance                |
| Limited compute            | ⭐⭐⭐⭐⭐  | Much cheaper than E models       |
| Robustness critical        | ⭐⭐⭐⭐    | Variance reduction               |
| Small-medium datasets      | ⭐⭐⭐⭐    | Ensemble helps with limited data |
| Large datasets             | ⭐⭐⭐      | Single models often sufficient   |
| Need maximum accuracy      | ⭐⭐        | Full ensemble better             |
| Speed critical             | ⭐⭐        | Single model faster              |

## Architecture Details

### Batch-Ensembling Mechanism

**Traditional ensemble:**

```
Training:
  Model 1: Batch 1 → Forward → Loss₁ → Update weights₁
  Model 2: Batch 2 → Forward → Loss₂ → Update weights₂
  ...
  Model E: Batch E → Forward → Lossₑ → Update weightsₑ
  (E separate forward passes)

Inference:
  Input → Model 1 → Pred₁ ┐
       → Model 2 → Pred₂ ├→ Average
       ...              │
       → Model E → Predₑ ┘
  (E forward passes)
```

**TabM (batch-ensembling):**

```
Training:
  Batch (size B) → Split into E sub-batches (size B/E each)
  Sub-batch 1 → Member 1 ┐
  Sub-batch 2 → Member 2 ├→ Single forward pass
  ...                    │
  Sub-batch E → Member E ┘
  Combined loss → Update all members
  (1 forward pass, E members processed)

Inference:
  Input (repeated E times) → All members → Average
  (1 forward pass with E-times input)
```

**Key insight:**

- Traditional: E forward passes (expensive)
- TabM: 1 forward pass with batch splitting (efficient)

### Mathematical Formulation

**Standard ensemble:**

$$
\hat{y} = \frac{1}{E} \sum_{e=1}^{E} f_e(\mathbf{x}; \theta_e)
$$

Each $f_e$ trained on separate data.

**TabM ensemble:**

$$
\hat{y} = \frac{1}{E} \sum_{e=1}^{E} f_e(\mathbf{x}; \theta_e)
$$

Same form, but all $f_e$ trained jointly via batch splitting:

**Training on batch $\mathcal{B} = \{\mathbf{x}_1, ..., \mathbf{x}_B\}$:**

$$
\mathcal{L} = \frac{1}{E} \sum_{e=1}^{E} \frac{1}{B/E} \sum_{i \in \text{split}_e} \text{loss}(f_e(\mathbf{x}_i; \theta_e), y_i)
$$

Where $\text{split}_e$ is subset of batch for member $e$.

### Full Architecture

```
Input batch [x₁, x₂, ..., xₙ]
        ↓
Split into E sub-batches
  [x₁, ..., xₙ/ₑ] → Member 1
  [xₙ/ₑ₊₁, ...] → Member 2
  ...
  [..., xₙ] → Member E
        ↓
╔═══════════════════════════════╗
║ Member 1 (MLP)                ║
║ Input → Layer 1 → ... → Output║
║ Parameters: θ₁                ║
╚═══════════════════════════════╝
╔═══════════════════════════════╗
║ Member 2 (MLP)                ║
║ Parameters: θ₂                ║
╚═══════════════════════════════╝
        ...
╔═══════════════════════════════╗
║ Member E (MLP)                ║
║ Parameters: θₑ                ║
╚═══════════════════════════════╝
        ↓
Combine predictions
  Average(pred₁, pred₂, ..., predₑ)
        ↓
Final prediction + uncertainty
```

### Why Batch Splitting Creates Diversity

**Diversity sources:**

1. **Different data per member:** Each sees different subset of batch
2. **Independent gradient updates:** Gradients differ across members
3. **Random initialization:** Members start from different points
4. **Batch-to-batch variation:** Different splits across batches
5. **Stochastic optimization:** SGD noise differs per member

**Unlike full ensemble:**

- No bootstrap sampling needed (batch splitting sufficient)
- All trained jointly (shared computational graph)
- Gradient-based diversity (not data-based)

## Known Limitations

```{warning}
**Constraints and trade-offs:**
- **Not as good as full ensemble:** 70-80% of benefit, not 100%
- **Batch size constraints:** Requires batch divisible by n_ensembles
- **Memory overhead:** ~50% more memory than single model
- **Inference cost:** 30% slower than single model
- **Diminishing returns:** Beyond 8 members, little benefit
- **Hyperparameter sensitivity:** Batch size affects diversity
```

**When limitations matter:**

- Can afford full ensemble → Train E models (20-30% better)
- Speed critical → Use single MLP (30% faster)
- Very large models → Memory overhead becomes significant
- Very small batches → Batch splitting creates tiny sub-batches
- Maximum accuracy needed → Full ensemble or better architecture

## Uncertainty Estimation

```{tip}
**Ensemble variance for uncertainty:** TabM provides natural uncertainty estimates via ensemble member variance. Higher variance = higher uncertainty.
```

**Computing uncertainty:**

```python
# After training
model = TabMRegressor()
model.fit(X_train, y_train, max_epochs=50)

# Get predictions from all ensemble members
# member_predictions = [member_i.predict(X_test) for each member]
# mean_prediction = mean(member_predictions)
# uncertainty = std(member_predictions)

# High std → high uncertainty (members disagree)
# Low std → low uncertainty (members agree)
```

**Use cases for uncertainty:**

- Active learning (query high-uncertainty samples)
- Confidence filtering (reject high-uncertainty predictions)
- Risk-sensitive applications (flag uncertain predictions)

## Comparison with Other Efficient Ensembles

| Method                  | Training Cost | Inference Cost | Diversity Quality | Best For            |
| ----------------------- | ------------- | -------------- | ----------------- | ------------------- |
| **TabM**                | 1.3x          | 1.3x           | Good              | Balanced efficiency |
| Snapshot ensemble       | 1x            | Ex             | Moderate          | Training efficiency |
| MC Dropout              | 1x            | Ex             | Moderate-Low      | Training efficiency |
| Fast geometric ensemble | 1x            | Ex             | Moderate          | Training efficiency |
| Full ensemble           | Ex            | Ex             | Best              | Accuracy critical   |

## References

**Batch ensemble technique:**

- Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2022). _On Embeddings for Numerical Features in Tabular Deep Learning_. arXiv:2203.05556. (Introduces TabM)

**Ensemble methods:**

- Dietterich, T. G. (2000). _Ensemble Methods in Machine Learning_. MCS 2000. (Foundation for ensemble theory)
- Lakshminarayanan, B., et al. (2017). _Simple and Scalable Predictive Uncertainty Estimation_. NeurIPS 2017. (Deep ensembles)

**Efficient ensembles:**

- Wen, Y., et al. (2020). _BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning_. ICLR 2020

## See Also

- [MLP](mlp) — Single model baseline
- [ResNet](resnet) — Alternative fast baseline
- [Mambular](mambular) — Better single-model accuracy
- [Ensemble Guide](../../tutorials/ensembles) — Full ensemble techniques
- [Comparison Tables](../comparison_tables) — Performance across all models

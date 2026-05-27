# TabularNN

**Recurrent Neural Networks for Tabular Data** — Processes features sequentially using LSTM/GRU/RNN cells.

```{tip}
**Architecture highlight:** Treats features as sequence, processes with recurrent cells. O(n·f·d) complexity where f = feature count. Captures sequential dependencies between features when ordering matters. Best for temporal/ordered tabular features or when feature relationships sequential in nature.
```

## Architecture Overview

**Core mechanism:** Sequential processing of features via RNN/LSTM/GRU  
**Complexity:** O(n·f·d) per forward pass where f = feature sequence length  
**Memory:** O(f·d) for hidden states  
**Inductive bias:** Sequential dependencies between features

### Key Components

1. **Feature ordering:** Determines sequence in which features processed
2. **Recurrent cell:** LSTM, GRU, or vanilla RNN for sequential modeling
3. **Hidden states:** Carry information across feature sequence
4. **Output aggregation:** Final hidden state or pooling for prediction

**Architecture comparison:**

| Model         | Feature Processing     | Complexity | Sequential Assumption       | Best For                  |
| ------------- | ---------------------- | ---------- | --------------------------- | ------------------------- |
| **TabularNN** | Sequential (RNN)       | O(n·f·d)   | Yes - feature order matters | Ordered/temporal features |
| Mambular      | Sequential (SSM)       | O(n·f·d)   | Weak - learned ordering     | General purpose           |
| FTTransformer | Parallel (attention)   | O(n·f·d)   | No - permutation invariant  | Unordered features        |
| MLP           | Parallel (feedforward) | O(n·f·d²)  | No - all at once            | Unordered features        |

```{note}
**Design assumption:** TabularNN assumes feature ordering is meaningful. Unlike transformers (permutation invariant), RNNs sensitive to order. Use when: (1) features naturally ordered (temporal), (2) domain knowledge suggests processing order, or (3) you want to learn sequential patterns between features.
```

## When to Use

| Scenario                        | Recommendation                                                | Reasoning                                    |
| ------------------------------- | ------------------------------------------------------------- | -------------------------------------------- |
| **Features naturally ordered**  | ✅ Use TabularNN                                              | Sequential processing matches data structure |
| **Temporal dependencies**       | ✅ Use TabularNN                                              | RNNs designed for temporal patterns          |
| **Time series features**        | ✅ Use TabularNN                                              | Each feature is time step                    |
| **Domain suggests ordering**    | ✅ Use TabularNN                                              | E.g., medical tests in chronological order   |
| **Want to learn feature order** | ✅ Try TabularNN                                              | Can discover dependencies                    |
| **Features unordered**          | ❌ Use [FTTransformer](fttransformer) or [Mambular](mambular) | Permutation invariance better                |
| **Need speed**                  | ❌ Use [ResNet](resnet) or [MLP](mlp)                         | RNNs inherently sequential (slow)            |
| **Very long sequences**         | ❌ Use [Mambular](mambular)                                   | SSM better for long sequences                |
| **Simple patterns**             | ❌ Use [MLP](mlp)                                             | Simpler models sufficient                    |

## Computational Characteristics

### Complexity Analysis

| Model                | Time Complexity | Parallelization      | Sequential Steps | Parameters |
| -------------------- | --------------- | -------------------- | ---------------- | ---------- |
| **TabularNN (LSTM)** | O(n·f·d)        | Limited (sequential) | f                | ~200K-800K |
| Mambular (SSM)       | O(n·f·d)        | Better               | f                | ~100K-500K |
| FTTransformer        | O(n·f·d)        | Full (parallel)      | 1                | ~200K-1M   |
| MLP                  | O(n·f·d²)       | Full                 | 1                | ~100K-300K |

### Training Efficiency

| Model         | Training Speed | GPU Utilization | Parallelization      | Best Use Case       |
| ------------- | -------------- | --------------- | -------------------- | ------------------- |
| **TabularNN** | Slow-Moderate  | Medium          | Limited (sequential) | Sequential features |
| Mambular      | Moderate       | High            | Better (SSM)         | General purpose     |
| FTTransformer | Moderate-Slow  | High            | Full (attention)     | Many features       |
| MLP           | Fast           | High            | Full                 | Simple patterns     |
| ResNet        | Fast-Moderate  | High            | Full                 | Fast baseline       |

```{tip}
**Sequential bottleneck:** RNNs process features one-by-one, limiting parallelization. GPUs optimize parallel operations, so RNNs underutilize hardware. Use when sequential dependencies worth the speed cost.
```

### RNN Variant Comparison

| Cell Type | Parameters    | Training Speed | Memory  | Gradient Flow    | Best For                          |
| --------- | ------------- | -------------- | ------- | ---------------- | --------------------------------- |
| **LSTM**  | Highest (~4x) | Slowest        | Highest | Best (gates)     | Default choice, long dependencies |
| **GRU**   | Medium (~3x)  | Moderate       | Medium  | Good             | Speed-accuracy balance            |
| **RNN**   | Lowest (1x)   | Fastest        | Lowest  | Poor (vanishing) | Short sequences, speed critical   |

## Configuration Guidelines

### Model Config (TabularNNConfig)

```{note}
**Key parameters:** `model_type` chooses RNN variant (LSTM recommended), `d_model` controls hidden state size, `n_layers` stacks recurrent layers for hierarchical patterns. Deeper stacks capture more complex sequential dependencies.
```

| Parameter       | Default | Typical Range | Description                      | Impact                             |
| --------------- | ------- | ------------- | -------------------------------- | ---------------------------------- |
| `model_type`    | "lstm"  | lstm/gru/rnn  | RNN cell type                    | High - gradient flow & capacity    |
| `d_model`       | 128     | 64-256        | Hidden state dimension           | High - capacity                    |
| `n_layers`      | 4       | 2-8           | Number of recurrent layers       | High - hierarchical patterns       |
| `dropout`       | 0.1     | 0.0-0.3       | Dropout rate                     | Dataset-dependent                  |
| `bidirectional` | False   | True/False    | Process sequence both directions | Moderate - captures future context |

### Parameter Impact Analysis

| Parameter Change     | Effect on Model          | Effect on Performance             | When to Adjust                |
| -------------------- | ------------------------ | --------------------------------- | ----------------------------- |
| LSTM → GRU           | Fewer parameters, faster | Similar accuracy, faster training | Speed matters                 |
| LSTM → RNN           | Much fewer parameters    | Worse on long sequences           | Very short sequences          |
| Increase d_model     | Larger states            | Higher capacity                   | Complex dependencies          |
| Increase n_layers    | Deeper hierarchy         | More abstraction                  | Hierarchical patterns         |
| Enable bidirectional | 2x parameters            | Better context (sees future)      | Batch processing (not online) |

### Recommended Settings by Dataset Size

| Dataset Size       | model_type | d_model | n_layers | dropout | bidirectional | batch_size | Reasoning                        |
| ------------------ | ---------- | ------- | -------- | ------- | ------------- | ---------- | -------------------------------- |
| **<1K samples**    | "gru"      | 64-128  | 2-3      | 0.2-0.3 | False         | 32         | Minimal capacity, regularization |
| **1K-5K samples**  | "lstm"     | 128     | 3-4      | 0.1-0.2 | False         | 64         | Balanced LSTM                    |
| **5K-10K samples** | "lstm"     | 128-192 | 4-6      | 0.1     | True          | 128        | Bidirectional justified          |
| **>10K samples**   | "lstm"     | 192-256 | 4-8      | 0.0-0.1 | True          | 256        | Full capacity                    |

### Quick Start

```python
from deeptab.models import TabularNNClassifier, TabularNNRegressor, TabularNNLSS
from deeptab.configs import TabularNNConfig, TrainerConfig

# Fast baseline with defaults (LSTM)
model = TabularNNClassifier()
model.fit(X_train, y_train, max_epochs=50)
predictions = model.predict(X_test)

# Custom configuration for temporal features
cfg = TabularNNConfig(
    model_type="lstm",  # or "gru", "rnn"
    d_model=128,
    n_layers=4,
    dropout=0.1,
    bidirectional=True,  # if batch processing (not online)
)
trainer = TrainerConfig(
    lr=1e-3,
    batch_size=128,
    max_epochs=100,
)
model = TabularNNRegressor(model_config=cfg, trainer_config=trainer)
model.fit(X_train, y_train)

# Try different RNN types
for rnn_type in ["lstm", "gru", "rnn"]:
    cfg = TabularNNConfig(model_type=rnn_type)
    model = TabularNNClassifier(model_config=cfg)
    model.fit(X_train, y_train, max_epochs=50)
    # LSTM typically best but slowest

# LSS mode for distributional regression
model = TabularNNLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)
```

## Performance Characteristics

### Comparative Analysis

| vs Model          | Accuracy Gap       | Speed Comparison  | Memory  | When to Prefer TabularNN      | When to Prefer Alternative |
| ----------------- | ------------------ | ----------------- | ------- | ----------------------------- | -------------------------- |
| **Mambular**      | -3 to +2%          | 0.6-0.7x (slower) | Higher  | Sequential dependencies clear | General purpose            |
| **FTTransformer** | -5 to +5% (varies) | 0.5-0.6x (slower) | Similar | Features ordered              | Features unordered         |
| **MLP**           | Varies widely      | 0.3-0.4x (slower) | Similar | Sequential patterns           | Simple patterns            |
| **ResNet**        | Varies             | 0.4-0.5x (slower) | Similar | Feature order matters         | Speed critical             |

```{note}
**Performance profile:** TabularNN excels when feature ordering is meaningful. On unordered features, sequential processing is unnecessary overhead. On temporal features or when domain suggests ordering, can outperform order-agnostic models by 2-10%.
```

### When Feature Order Matters

| Feature Type                  | Order Matters?     | TabularNN Advantage | Best Alternative        |
| ----------------------------- | ------------------ | ------------------- | ----------------------- |
| Time series features          | Yes (temporal)     | High                | Mambular                |
| Medical tests (chronological) | Yes (time-ordered) | High                | Mambular                |
| Sensor readings (sequential)  | Yes (temporal)     | High                | Mambular                |
| Patient history (age-ordered) | Maybe              | Moderate            | Mambular, FTTransformer |
| Mixed categorical/numerical   | No                 | None (overhead)     | FTTransformer, MLP      |
| Random feature order          | No                 | None (harmful)      | Any non-sequential      |

### Use Case Suitability

| Use Case                     | Suitability | Reasoning                           |
| ---------------------------- | ----------- | ----------------------------------- |
| Temporal tabular features    | ⭐⭐⭐⭐⭐  | Designed for temporal sequences     |
| Time series as features      | ⭐⭐⭐⭐⭐  | Natural fit for RNN                 |
| Ordered domain features      | ⭐⭐⭐⭐    | Sequential dependencies             |
| Learn feature dependencies   | ⭐⭐⭐⭐    | Can discover ordering               |
| Small-medium sequences (<50) | ⭐⭐⭐      | RNN works well                      |
| Long sequences (>50)         | ⭐⭐        | Consider Mambular (better for long) |
| Unordered features           | ⭐⭐        | Unnecessary overhead                |
| Speed critical               | ⭐⭐        | Inherently sequential (slow)        |

## Architecture Details

### Sequential Feature Processing

**Standard MLP (parallel):**

```
All features → Hidden layer → ... → Output
[f₁, f₂, ..., fₙ] processed simultaneously
```

**TabularNN (sequential):**

```
f₁ → RNN → h₁
     h₁ + f₂ → RNN → h₂
               h₂ + f₃ → RNN → h₃
                         ...
                              hₙ → Output
```

**Hidden state carries information:**

- h₁ contains information about f₁
- h₂ contains information about f₁ and f₂
- h₃ contains information about f₁, f₂, and f₃
- etc.

### LSTM Cell Details

**LSTM gates control information flow:**

$$
\begin{align}
\mathbf{f}_t &= \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) && \text{(Forget gate)} \\
\mathbf{i}_t &= \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) && \text{(Input gate)} \\
\tilde{\mathbf{C}}_t &= \tanh(\mathbf{W}_C \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C) && \text{(Candidate)} \\
\mathbf{C}_t &= \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t && \text{(Cell state)} \\
\mathbf{o}_t &= \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) && \text{(Output gate)} \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{C}_t) && \text{(Hidden state)}
\end{align}
$$

**For tabular data:**

- $t$ indexes features (not time in traditional sense)
- $\mathbf{x}_t$ is feature $t$ value
- $\mathbf{h}_t$ accumulates information up to feature $t$

### Bidirectional Processing

**Unidirectional (default):**

```
f₁ → f₂ → f₃ → ... → fₙ
→  →  →  →  →  →  (forward only)
```

**Bidirectional:**

```
f₁ → f₂ → f₃ → ... → fₙ  (forward)
←  ←  ←  ←  ←  ←  (backward)
fₙ ← fₙ₋₁ ← fₙ₋₂ ← ... ← f₁

Final: concat(forward_hₙ, backward_h₁)
```

**Advantages:**

- Each feature sees both past and future context
- Better representation for batch processing
- Cannot be used for online/streaming predictions

**Trade-offs:**

- 2x parameters and compute
- Requires full sequence upfront
- Better accuracy when batch processing allowed

### Full Architecture

```
Input features [f₁, f₂, ..., fₙ]
        ↓
Optionally embed each feature
        ↓
Sequential processing:
        ↓
╔═══════════════════════════════╗
║ RNN Layer 1                   ║
║ f₁ → cell → h₁                ║
║ f₂, h₁ → cell → h₂            ║
║ ...                           ║
║ fₙ, hₙ₋₁ → cell → hₙ          ║
╚═══════════════════════════════╝
        ↓
╔═══════════════════════════════╗
║ RNN Layer 2                   ║
║ h₁⁽¹⁾ → cell → h₁⁽²⁾          ║
║ h₂⁽¹⁾, h₁⁽²⁾ → cell → h₂⁽²⁾  ║
║ ...                           ║
╚═══════════════════════════════╝
        ↓
    ... (L layers)
        ↓
Final hidden state hₙ⁽ᴸ⁾
  (or pooling over all states)
        ↓
Output head (task-specific)
        ↓
Predictions
```

### Feature Ordering Strategies

**If features naturally ordered:**

- Use chronological/temporal order
- Domain-specific ordering (e.g., medical tests by time)

**If features not naturally ordered:**

- Random order (baseline)
- Learn order via hyperparameter search
- Domain knowledge (hypothesize dependencies)
- Feature importance order (important first)
- Correlation-based order (cluster related features)

**Ordering experiment:**

```python
import numpy as np

# Try different feature orderings
orderings = [
    np.arange(n_features),  # Original
    np.random.permutation(n_features),  # Random 1
    np.random.permutation(n_features),  # Random 2
    feature_importance_order,  # By importance
]

for order in orderings:
    X_reordered = X[:, order]
    model = TabularNNClassifier()
    model.fit(X_reordered, y_train, max_epochs=50)
    # Check which ordering performs best
```

## Known Limitations

```{warning}
**Computational and applicability constraints:**
- **Sequential bottleneck:** Cannot parallelize across features (slow)
- **GPU underutilization:** Sequential processing limits GPU efficiency
- **Long sequences:** Gradients can vanish/explode with many features
- **Ordering sensitivity:** Performance depends on feature order
- **Unordered features:** Unnecessary overhead when order doesn't matter
- **Inference latency:** Sequential processing slower than parallel models
```

**When limitations matter:**

- Features unordered → Use FTTransformer or MLP (parallel processing)
- Speed critical → Use ResNet or MLP (faster)
- Many features (>100) → RNN becomes very slow
- Online inference needs → Unidirectional only (no bidirectional)
- GPU limited → CPU-based models may be faster

## Temporal Tabular Data Example

**Scenario:** Predicting patient outcome from lab tests over time

**Feature structure:**

```
Features: [test_1_day1, test_2_day1, test_3_day1,
           test_1_day2, test_2_day2, test_3_day2,
           ...
           test_1_dayN, test_2_dayN, test_3_dayN]
```

**Sequential ordering options:**

1. **By day (temporal):** All tests day 1, then day 2, etc.
   - Captures temporal progression
   - Each hidden state accumulates patient history

2. **By test (longitudinal):** All day 1 values, all day 2 values, etc.
   - Captures test-specific trends over time

**TabularNN advantage:** Naturally models temporal dependencies between tests and time points.

## Migration Path

**If TabularNN works but too slow:**

```python
# Start with TabularNN to validate sequential approach
model = TabularNNClassifier(model_config=TabularNNConfig(model_type="lstm"))
model.fit(X_train, y_train, max_epochs=50)
# Accuracy: 0.85, Training time: 100s

# Migrate to Mambular for similar benefits with better speed
from deeptab.models import MambularClassifier
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)
# Accuracy: 0.84-0.86, Training time: 60s (1.7x faster)
```

**If features unordered:**

```python
# If order doesn't matter, use parallel models
from deeptab.models import FTTransformerClassifier
model = FTTransformerClassifier()
model.fit(X_train, y_train, max_epochs=50)
# Better for unordered features
```

## References

**LSTM foundation:**

- Hochreiter, S., & Schmidhuber, J. (1997). _Long Short-Term Memory_. Neural Computation, 9(8). (Original LSTM)

**GRU variant:**

- Cho, K., et al. (2014). _Learning Phrase Representations using RNN Encoder-Decoder_. EMNLP 2014. (Introduces GRU)

**RNNs for tabular data:**

- Application of sequential models to structured data with temporal/ordered features

**Modern alternatives:**

- Gu, A., & Dao, T. (2024). _Mamba: Linear-Time Sequence Modeling_. (Better efficiency for long sequences)

## See Also

- [Mambular](mambular) — Better efficiency for sequential modeling
- [FTTransformer](fttransformer) — For unordered features (permutation invariant)
- [MLP](mlp) — Simple baseline for unordered features
- [Time Series Tutorial](../../tutorials/time_series) — Working with temporal data
- [Comparison Tables](../comparison_tables) — Performance across all models

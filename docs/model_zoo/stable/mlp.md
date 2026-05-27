# MLP (Multi-Layer Perceptron)

_Simple Feedforward Network for Tabular Data_

```{tip}
**Architecture Highlight**: Fastest baseline model with O(n·d²) complexity. Choose MLP when training speed is critical or as a strong baseline for comparison.
```

## Architecture Overview

MLP is a simple feedforward neural network that processes tabular data through successive linear transformations with non-linear activations. Each layer applies a learned weight matrix to all features simultaneously, making it the most straightforward deep learning approach for tabular data.

**Core Mechanism**: Sequential fully-connected layers with activation functions between each transformation, treating all features uniformly without specialized embedding or attention mechanisms.

**Computational Complexity**: O(n·d²) where n is samples and d is hidden dimension  
**Memory Scaling**: O(d²·L) where L is number of layers  
**Inductive Bias**: Smooth transformations, no assumptions about feature types or relationships

**Key Components**:

- Embedding layer for categorical/numerical features
- Stack of fully-connected (Linear) layers
- Non-linear activations (ReLU, GELU)
- Dropout regularization between layers
- Output head for task-specific predictions

### Architecture Comparison

| Aspect               | MLP              | ResNet           | Mambular            | FTTransformer        |
| -------------------- | ---------------- | ---------------- | ------------------- | -------------------- |
| Complexity           | O(n·d²)          | O(n·d²)          | O(n·f·d)            | O(n·f²·d)            |
| Training Speed       | **Fastest**      | Fast             | Moderate            | Moderate             |
| Memory Usage         | Lowest           | Low              | Medium              | Medium-High          |
| Feature Interactions | Implicit         | Skip connections | Sequential          | Global attention     |
| Best Use Case        | Baselines, speed | General purpose  | Sequential patterns | Complex interactions |

## When to Use

| Scenario                               | Recommendation            | Reasoning                                                     |
| -------------------------------------- | ------------------------- | ------------------------------------------------------------- |
| **Quick baseline needed**              | ✅ **Highly Recommended** | Fastest to train, establishes performance floor               |
| **Training time < 5 minutes**          | ✅ **Highly Recommended** | Trains 2-3x faster than transformers/SSMs                     |
| **CPU-only deployment**                | ✅ **Highly Recommended** | Minimal GPU requirements, efficient CPU inference             |
| **Simple feature relationships**       | ✅ **Recommended**        | No complex interactions needed                                |
| **Limited compute budget**             | ✅ **Recommended**        | Lowest memory and compute requirements                        |
| **Small datasets (<5K samples)**       | ✅ **Recommended**        | Simpler model reduces overfitting risk                        |
| **Need interpretability**              | ⚠️ **Use with caution**   | More interpretable than attention but less than linear models |
| **Complex feature interactions**       | ❌ **Not Recommended**    | Use FTTransformer or Mambular for better interaction modeling |
| **State-of-the-art accuracy required** | ❌ **Not Recommended**    | Typically 5-15% behind best models on complex tasks           |
| **Categorical-heavy datasets**         | ❌ **Not Recommended**    | TabTransformer better handles categorical embeddings          |

## Computational Characteristics

### Complexity Analysis

| Operation            | Time Complexity | Space Complexity | Notes                                      |
| -------------------- | --------------- | ---------------- | ------------------------------------------ |
| **Forward Pass**     | O(n·d²·L)       | O(n·d)           | Linear in samples, quadratic in hidden dim |
| **Backward Pass**    | O(n·d²·L)       | O(n·d)           | Same as forward pass                       |
| **Memory (weights)** | O(d²·L)         | O(d²·L)          | Dominated by weight matrices               |
| **Batch Processing** | O(b·d²·L)       | O(b·d)           | Scales linearly with batch size            |

Where: n = samples, d = hidden dimension, L = number of layers, b = batch size

### Training Efficiency Comparison

| Model         | Relative Training Time | Relative Memory | Convergence Speed |
| ------------- | ---------------------- | --------------- | ----------------- |
| **MLP**       | **1.0x (baseline)**    | **1.0x**        | **Fast**          |
| ResNet        | 1.1x                   | 1.1x            | Fast              |
| Mambular      | 1.5-2.0x               | 1.3x            | Moderate          |
| FTTransformer | 2.0-2.5x               | 1.5-2.0x        | Moderate          |
| SAINT         | 3.0-4.0x               | 2.0-2.5x        | Slow              |

### Memory Requirements (Approximate)

| Configuration        | Parameters | GPU Memory (batch=256) | Training Throughput |
| -------------------- | ---------- | ---------------------- | ------------------- |
| Small (d=64, L=4)    | ~50K       | ~200 MB                | ~10K samples/sec    |
| Medium (d=128, L=6)  | ~200K      | ~400 MB                | ~8K samples/sec     |
| Large (d=256, L=8)   | ~1.5M      | ~800 MB                | ~5K samples/sec     |
| XLarge (d=512, L=10) | ~10M       | ~2 GB                  | ~2K samples/sec     |

## Configuration Guidelines

### Parameter Reference

| Parameter    | Default     | Range               | Impact       | Description                                      |
| ------------ | ----------- | ------------------- | ------------ | ------------------------------------------------ |
| `d_model`    | 128         | 64-512              | **High**     | Hidden dimension size - primary capacity control |
| `n_layers`   | 8           | 4-12                | **High**     | Number of layers - depth of network              |
| `dropout`    | 0.1         | 0.0-0.3             | **Moderate** | Dropout rate for regularization                  |
| `activation` | "relu"      | relu/gelu/silu      | **Low**      | Non-linearity between layers                     |
| `norm`       | "layernorm" | layernorm/batchnorm | **Low**      | Normalization strategy                           |
| `residual`   | False       | True/False          | **Moderate** | Add skip connections (makes it ResNet-like)      |

### Recommended Settings by Dataset Size

| Dataset Size         | d_model | n_layers | dropout | Expected Training Time |
| -------------------- | ------- | -------- | ------- | ---------------------- |
| **<5K samples**      | 64      | 4        | 0.2     | <1 minute              |
| **5K-50K samples**   | 128     | 6        | 0.15    | 1-5 minutes            |
| **50K-500K samples** | 256     | 8        | 0.1     | 5-15 minutes           |
| **>500K samples**    | 512     | 10       | 0.05    | 15-60 minutes          |

```{note}
**Scaling Rule**: Increase `d_model` before `n_layers` when scaling up. Doubling `d_model` increases capacity more than adding 2 layers.
```

## Quick Start

### Classification Example

```python
from deeptab.models import MLPClassifier
from deeptab.configs import MLPConfig

# Configure model
config = MLPConfig(
    d_model=128,
    n_layers=8,
    dropout=0.1,
    activation="relu"
)

# Initialize and train
model = MLPClassifier(config=config)
model.fit(
    X_train, y_train,
    max_epochs=50,
    batch_size=256,
    learning_rate=1e-3
)

# Predict
predictions = model.predict(X_test)
```

### Regression Example

```python
from deeptab.models import MLPRegressor
from deeptab.configs import MLPConfig

config = MLPConfig(
    d_model=256,
    n_layers=6,
    dropout=0.15
)

model = MLPRegressor(config=config)
model.fit(X_train, y_train, max_epochs=100)

predictions = model.predict(X_test)
```

### Distributional Regression (LSS)

```python
from deeptab.models import MLPLSS
from deeptab.configs import MLPConfig

# Predict full distribution instead of point estimates
config = MLPConfig(d_model=128, n_layers=8)
model = MLPLSS(config=config, distribution="normal")

model.fit(X_train, y_train, max_epochs=50)
distribution_params = model.predict(X_test)  # Returns mean and std
```

## Performance Characteristics

### Comparative Analysis

| vs. Model          | Accuracy Gap | Speed Advantage | Memory    | When to Prefer MLP            | When to Prefer Alternative         |
| ------------------ | ------------ | --------------- | --------- | ----------------------------- | ---------------------------------- |
| **ResNet**         | -2% to -5%   | 10% faster      | Equal     | Need absolute fastest         | Complex patterns, better accuracy  |
| **Mambular**       | -5% to -15%  | **2x faster**   | 2x less   | Speed critical, baseline      | Sequential patterns, best accuracy |
| **FTTransformer**  | -5% to -15%  | **2.5x faster** | 2x less   | CPU deployment, fast training | Feature interactions, state-of-art |
| **TabTransformer** | -3% to -10%  | 1.8x faster     | 1.5x less | Few categoricals, speed       | Many categorical features          |
| **XGBoost**        | Similar      | Similar         | N/A       | Deep learning pipeline needed | No deep learning required          |

```{important}
**Performance Context**: MLP typically achieves 80-90% of the best model's performance while training 2-3x faster. It's an excellent choice when the marginal accuracy gain doesn't justify the computational cost.
```

### Strengths and Weaknesses

**Strengths**:

- ✅ Fastest training among all deep learning models
- ✅ Lowest memory footprint (d²·L parameters)
- ✅ Strong baseline performance (competitive with XGBoost)
- ✅ Simple architecture, easy to debug
- ✅ No special requirements (works on any hardware)
- ✅ Scales linearly with batch size

**Weaknesses**:

- ❌ No explicit feature interaction modeling
- ❌ Treats all features uniformly (no categorical specialization)
- ❌ Typically 5-15% behind state-of-the-art on complex tasks
- ❌ May underfit on very complex patterns
- ❌ Limited expressiveness compared to attention/SSM models

## Use Case Suitability

| Use Case                        | Suitability | Notes                                       |
| ------------------------------- | ----------- | ------------------------------------------- |
| **Rapid Prototyping**           | ⭐⭐⭐⭐⭐  | Perfect for quick experiments and baselines |
| **Production Deployment (CPU)** | ⭐⭐⭐⭐⭐  | Minimal requirements, fast inference        |
| **Small Datasets (<5K)**        | ⭐⭐⭐⭐    | Simple model reduces overfitting            |
| **Medium Datasets (5K-100K)**   | ⭐⭐⭐⭐    | Good balance of speed and accuracy          |
| **Large Datasets (>100K)**      | ⭐⭐⭐      | Can work but more complex models may help   |
| **Time Series Tabular**         | ⭐⭐        | No sequential modeling, consider Mambular   |
| **Categorical-Heavy Data**      | ⭐⭐⭐      | Works but TabTransformer better             |
| **High-Stakes Accuracy**        | ⭐⭐        | Use more sophisticated models               |
| **Research Baseline**           | ⭐⭐⭐⭐⭐  | Essential comparison point                  |
| **Real-time Inference (<1ms)**  | ⭐⭐⭐⭐⭐  | Fastest model for latency-critical apps     |

## Architecture Details

### Network Structure

```
Input Features (f dimensions)
    ↓
Embedding Layer → Numeric + Categorical Embeddings
    ↓
[Linear(d, d) → Activation → Dropout] × L layers
    ↓
Output Head (task-specific)
```

### Mathematical Formulation

For layer l, the transformation is:

$$h_l = \text{Dropout}(\sigma(W_l h_{l-1} + b_l))$$

Where:

- $h_l$ is the hidden state at layer l
- $W_l \in \mathbb{R}^{d \times d}$ is the weight matrix
- $b_l \in \mathbb{R}^d$ is the bias vector
- $\sigma$ is the activation function (ReLU, GELU, etc.)
- Dropout is applied for regularization

**Parameter Count**:
$$\text{params} = f \cdot d + L \cdot d^2 + L \cdot d + d \cdot c$$

Where f = input features, d = hidden dim, L = layers, c = output classes

### Key Design Choices

1. **Uniform Feature Processing**: All features pass through the same transformations, no specialized handling for categoricals vs numericals
2. **Fixed Width**: Hidden dimension stays constant across all layers (unlike encoder-decoder architectures)
3. **Dense Connections**: Every neuron connects to all neurons in next layer
4. **No Memory**: Processes each sample independently, no sequential dependencies

### Comparison to ResNet

MLP vs ResNet differ only by skip connections:

| Feature        | MLP                    | ResNet                           |
| -------------- | ---------------------- | -------------------------------- |
| Core Transform | $h_l = f(W_l h_{l-1})$ | $h_l = h_{l-1} + f(W_l h_{l-1})$ |
| Gradient Flow  | Direct                 | Residual paths help              |
| Depth Scaling  | Harder to train deep   | Easier to train deep             |
| Performance    | Slightly lower         | +2-5% accuracy                   |
| Speed          | Fastest                | Nearly as fast                   |

```{warning}
**Known Limitations**

1. **No Feature Interaction Modeling**: MLP learns interactions implicitly through layers, but this is less effective than explicit attention or cross-feature mechanisms
2. **Categorical Features**: Embeddings are learned but not contextualized like TabTransformer
3. **Depth Limitations**: Without skip connections, very deep MLPs (>12 layers) become hard to train
4. **Overfitting on Small Data**: High capacity relative to simple patterns can lead to overfitting
5. **No Sequential Awareness**: Cannot model temporal or sequential patterns in data
```

## References

1. **Rosenblatt, F. (1958)**. _The Perceptron: A Probabilistic Model for Information Storage and Retrieval in the Brain_. Psychological Review, 65(6):386-408.

2. **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986)**. _Learning Representations by Back-propagating Errors_. Nature, 323(6088):533-536.

3. **Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021)**. _Revisiting Deep Learning Models for Tabular Data_. NeurIPS 2021. [Establishes MLP as strong baseline for modern tabular learning]

4. **Shavitt, I., & Segal, E. (2018)**. _Regularization Learning Networks: Deep Learning for Tabular Datasets_. NeurIPS 2018.

5. **Kadra, A., et al. (2021)**. _Well-tuned Simple Nets Excel on Tabular Datasets_. NeurIPS 2021. [Shows properly tuned MLPs are competitive]

## See Also

- **[ResNet](resnet.md)** — MLP + skip connections for better gradient flow
- **[Mambular](mambular.md)** — State-space model for sequential patterns
- **[FTTransformer](fttransformer.md)** — Transformer with feature-wise attention
- **[TabTransformer](tabtransformer.md)** — Attention on categorical features only
- **[Model Selection Guide](../model_selection.md)** — Choose the right architecture for your task

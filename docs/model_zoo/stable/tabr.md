# TabR (Retrieval-Augmented Tabular Learning)

_Neural Network with k-Nearest Neighbors Retrieval_

```{tip}
**Architecture Highlight**: Combines neural network predictions with kNN retrieval for O(n·k·d + n·d²) complexity. Choose TabR when local similarity patterns matter and you have >50K training samples to retrieve from.
```

## Architecture Overview

TabR augments neural network predictions with k-nearest neighbor retrieval from the training set. During inference, it retrieves the k most similar training samples, processes them through a context encoder, and combines this context with the test sample's neural representation for final prediction. This non-parametric component enables the model to adapt predictions based on local data patterns.

**Core Mechanism**: For each test sample, retrieve k nearest training samples → encode context → combine with neural network embedding → predict. This allows the model to leverage local similarity beyond what the neural network alone learns.

**Computational Complexity**: O(n·k·d + n·d²) where k is neighbors, d is dimension  
**Memory Scaling**: O(N_train·d) must store all training embeddings for retrieval  
**Inductive Bias**: Local similarity is informative; similar training examples improve predictions

**Key Components**:

- Feature embedding network (like ResNet/MLP)
- Training data storage for retrieval (full dataset in memory)
- kNN search mechanism (approximate nearest neighbors)
- Context encoder for retrieved neighbors
- Fusion layer combining query + context

### Architecture Comparison

| Aspect                 | TabR           | ModernNCA                | Mambular   | FTTransformer    |
| ---------------------- | -------------- | ------------------------ | ---------- | ---------------- |
| Complexity (Train)     | O(n·d²)        | O(n·d²)                  | O(n·f·d)   | O(n·f²·d)        |
| Complexity (Inference) | O(k·d + d²)    | O(N·d)                   | O(f·d)     | O(f²·d)          |
| Memory (Inference)     | O(N_train·d)   | O(N_train·d)             | O(d²·L)    | O(f²·d)          |
| Retrieval              | kNN (k fixed)  | All neighbors            | None       | None             |
| Training Speed         | Moderate       | Slow                     | Moderate   | Moderate         |
| Best Use Case          | Local patterns | Distance metric learning | Sequential | Global attention |

## When to Use

| Scenario                         | Recommendation            | Reasoning                                                     |
| -------------------------------- | ------------------------- | ------------------------------------------------------------- |
| **Local similarity matters**     | ✅ **Highly Recommended** | Retrieval exploits local structure neural nets may miss       |
| **Large training sets (>50K)**   | ✅ **Highly Recommended** | More training data → better retrieval → stronger performance  |
| **Non-stationary distributions** | ✅ **Highly Recommended** | Can adapt to local regions without retraining                 |
| **Complex decision boundaries**  | ✅ **Recommended**        | kNN + neural net captures both smooth and local patterns      |
| **Sufficient inference memory**  | ✅ **Recommended**        | Must store N_train embeddings in memory/disk                  |
| **Moderate inference speed OK**  | ✅ **Recommended**        | kNN search adds latency but often worthwhile                  |
| **Need uncertainty estimates**   | ✅ **Recommended**        | Neighbor diversity can indicate prediction confidence         |
| **Online learning scenarios**    | ⚠️ **Use with caution**   | Can add new samples to index, but requires careful management |
| **Real-time inference (<10ms)**  | ❌ **Not Recommended**    | kNN search adds overhead; use pure neural models              |
| **Small datasets (<10K)**        | ❌ **Not Recommended**    | Retrieval less effective with limited training data           |
| **Limited memory budget**        | ❌ **Not Recommended**    | Must store O(N·d) training embeddings                         |
| **No local structure**           | ❌ **Not Recommended**    | Overhead not justified if global patterns dominate            |

## Computational Characteristics

### Complexity Analysis

| Operation                      | Time Complexity    | Space Complexity | Notes                               |
| ------------------------------ | ------------------ | ---------------- | ----------------------------------- |
| **Training (Forward)**         | O(n·d²·L)          | O(n·d)           | Standard neural network             |
| **Inference (kNN Search)**     | O(k·log(N) + k·d)  | O(N·d)           | Approximate NN with index           |
| **Inference (Context Encode)** | O(k·d²)            | O(k·d)           | Encode retrieved neighbors          |
| **Inference (Fusion)**         | O(d²)              | O(d)             | Combine query + context             |
| **Total Inference**            | O(k·log(N) + k·d²) | O(N·d)           | Dominated by kNN + context encoding |
| **Memory (Storage)**           | O(N·d + d²·L)      | O(N·d)           | Training embeddings + model weights |

Where: n = batch size, N = training set size, k = neighbors, d = dimension, L = layers

### Training Efficiency Comparison

| Model             | Training Time | Inference Time   | Memory (Inference) | Scalability to Large N |
| ----------------- | ------------- | ---------------- | ------------------ | ---------------------- |
| **MLP/ResNet**    | 1.0x          | 1.0x             | Low                | ✅ Excellent           |
| **Mambular**      | 1.5x          | 1.2x             | Medium             | ✅ Good                |
| **FTTransformer** | 2.0x          | 1.5x             | Medium             | ✅ Good                |
| **TabR**          | **1.3x**      | **2-3x slower**  | **High (O(N·d))**  | ⚠️ Moderate            |
| **ModernNCA**     | 2.5x          | **5-10x slower** | Very High          | ❌ Poor                |

```{note}
**Inference Tradeoff**: TabR's inference is 2-3x slower than pure neural models due to kNN search, but this overhead often yields 3-10% accuracy gains on tasks with strong local structure.
```

### Memory Requirements (Approximate)

| Training Set Size | Embedding Dim | Index Memory | Total Memory (inference) | kNN Search Time |
| ----------------- | ------------- | ------------ | ------------------------ | --------------- |
| **10K samples**   | 128           | ~5 MB        | ~50 MB                   | <5ms            |
| **50K samples**   | 128           | ~25 MB       | ~100 MB                  | ~10ms           |
| **100K samples**  | 256           | ~100 MB      | ~200 MB                  | ~15ms           |
| **500K samples**  | 256           | ~500 MB      | ~700 MB                  | ~30ms           |
| **1M samples**    | 512           | ~2 GB        | ~3 GB                    | ~50ms           |

```{important}
**Memory Constraint**: Unlike pure neural models that only need model weights at inference, TabR requires storing all training embeddings. For 1M samples with d=256, this is ~1GB of memory.
```

## Configuration Guidelines

### Parameter Reference

| Parameter                | Default     | Range          | Impact       | Description                                           |
| ------------------------ | ----------- | -------------- | ------------ | ----------------------------------------------------- |
| `d_model`                | 128         | 64-512         | **High**     | Embedding dimension (also determines retrieval space) |
| `n_layers`               | 4           | 3-8            | **High**     | Depth of embedding network                            |
| `k_neighbors`            | 32          | 8-128          | **High**     | Number of neighbors to retrieve                       |
| `context_encoder_layers` | 2           | 1-4            | **Moderate** | Depth of context encoder for neighbors                |
| `dropout`                | 0.1         | 0.0-0.3        | **Moderate** | Dropout regularization                                |
| `use_approx_nn`          | True        | True/False     | **High**     | Use approximate NN (HNSW) vs exact search             |
| `index_metric`           | "cosine"    | cosine/l2      | **Moderate** | Distance metric for retrieval                         |
| `context_aggregation`    | "attention" | mean/attention | **Moderate** | How to aggregate retrieved neighbors                  |

### Recommended Settings by Dataset Size

| Dataset Size | d_model | n_layers | k_neighbors | context_encoder | Expected Training | Expected Inference |
| ------------ | ------- | -------- | ----------- | --------------- | ----------------- | ------------------ |
| **<10K**     | 64      | 3        | 16          | 2 layers        | 5-10 min          | ~10ms/sample       |
| **10K-50K**  | 128     | 4        | 32          | 2 layers        | 10-30 min         | ~15ms/sample       |
| **50K-200K** | 192     | 4        | 48          | 3 layers        | 30-90 min         | ~20ms/sample       |
| **200K-1M**  | 256     | 5        | 64          | 3 layers        | 1-3 hours         | ~30ms/sample       |
| **>1M**      | 256     | 6        | 96          | 4 layers        | 3-6 hours         | ~50ms/sample       |

```{note}
**Scaling Rule**: Increase `k_neighbors` as training set grows. With more data, you can retrieve more neighbors while maintaining relevance. Typical: k ≈ 0.01% of training size.
```

## Quick Start

### Classification Example

```python
from deeptab.models import TabRClassifier
from deeptab.configs import TabRConfig

# Configure retrieval-augmented model
config = TabRConfig(
    d_model=128,
    n_layers=4,
    k_neighbors=32,
    context_encoder_layers=2,
    use_approx_nn=True
)

# Initialize and train
model = TabRClassifier(config=config)
model.fit(
    X_train, y_train,
    max_epochs=100,
    batch_size=256,
    learning_rate=1e-3
)

# Predict (automatically retrieves neighbors)
predictions = model.predict(X_test)
```

### Regression Example

```python
from deeptab.models import TabRRegressor
from deeptab.configs import TabRConfig

config = TabRConfig(
    d_model=256,
    n_layers=5,
    k_neighbors=48,
    context_encoder_layers=3,
    context_aggregation="attention"  # Weight neighbors by relevance
)

model = TabRRegressor(config=config)
model.fit(X_train, y_train, max_epochs=150)

predictions = model.predict(X_test)
```

### Distributional Regression (LSS) with Uncertainty

```python
from deeptab.models import TabRLSS
from deeptab.configs import TabRConfig

# Retrieval naturally provides uncertainty estimates via neighbor diversity
config = TabRConfig(
    d_model=192,
    n_layers=4,
    k_neighbors=64
)

model = TabRLSS(config=config, distribution="normal")
model.fit(X_train, y_train, max_epochs=100)

# Returns distributional parameters informed by retrieved neighbors
distribution_params = model.predict(X_test)
```

### Accessing Retrieved Neighbors

```python
# Get predictions along with retrieved neighbor information
predictions, neighbors = model.predict(X_test, return_neighbors=True)

# neighbors is a dict with:
# - 'indices': [batch_size, k] indices into training set
# - 'distances': [batch_size, k] distances to neighbors
# - 'labels': [batch_size, k] neighbor labels (for analysis)

# Use for interpretability or uncertainty quantification
```

## Performance Characteristics

### Comparative Analysis

| vs. Model         | Accuracy Gap   | Training Speed    | Inference Speed | Memory    | When to Prefer TabR        | When to Prefer Alternative          |
| ----------------- | -------------- | ----------------- | --------------- | --------- | -------------------------- | ----------------------------------- |
| **Mambular**      | +3% to +8%     | 15% slower        | **2x slower**   | 3x more   | Local patterns, large data | Sequential patterns, fast inference |
| **FTTransformer** | +2% to +10%    | 30% faster        | **2x slower**   | 2-3x more | Retrieval benefits clear   | Pure attention sufficient           |
| **ResNet**        | +5% to +15%    | Similar           | **3x slower**   | 5x more   | Complex boundaries         | Simple patterns, speed critical     |
| **ModernNCA**     | Similar to +5% | 2x faster         | **3x faster**   | Similar   | k is sufficient            | Need all neighbors                  |
| **XGBoost**       | +2% to +8%     | Context-dependent | Similar         | Less      | Deep embeddings valuable   | No deep learning needed             |

```{important}
**Performance Sweet Spot**: TabR excels on datasets with >50K samples where local similarity is predictive. Gains are most pronounced on complex tasks where global patterns are insufficient.
```

### Strengths and Weaknesses

**Strengths**:

- ✅ Captures local structure neural networks miss
- ✅ Non-parametric adaptation to local regions
- ✅ Strong performance on large datasets (>50K)
- ✅ Natural uncertainty quantification via neighbor diversity
- ✅ Can incorporate new data by updating index (no retraining)
- ✅ Interpretable via retrieved neighbors
- ✅ Robust to distribution shift in local regions

**Weaknesses**:

- ❌ High inference memory (must store N training embeddings)
- ❌ Slower inference (2-3x) due to kNN search overhead
- ❌ Less effective on small datasets (<10K)
- ❌ Requires careful index management (HNSW, FAISS)
- ❌ Training data must be retained (privacy/storage concerns)
- ❌ No benefit if local structure is weak
- ❌ Complexity in deployment (model + index + training data)

## Use Case Suitability

| Use Case                      | Suitability | Notes                                                |
| ----------------------------- | ----------- | ---------------------------------------------------- |
| **Large Datasets (>100K)**    | ⭐⭐⭐⭐⭐  | More data → better retrieval → stronger gains        |
| **Recommendation Systems**    | ⭐⭐⭐⭐⭐  | Local user/item similarity highly predictive         |
| **Medical Diagnosis**         | ⭐⭐⭐⭐    | Case-based reasoning via similar patient retrieval   |
| **Fraud Detection**           | ⭐⭐⭐⭐    | Detect patterns similar to known fraud cases         |
| **Anomaly Detection**         | ⭐⭐⭐⭐    | Neighbor distances indicate anomalies                |
| **Drug Discovery**            | ⭐⭐⭐⭐    | Molecular similarity drives predictions              |
| **Financial Forecasting**     | ⭐⭐⭐      | Historical similar contexts inform predictions       |
| **Real-time Systems (<10ms)** | ⭐⭐        | kNN overhead may be prohibitive                      |
| **Small Datasets (<10K)**     | ⭐⭐        | Insufficient training data for effective retrieval   |
| **Privacy-Sensitive**         | ⭐⭐        | Must store training data at inference time           |
| **Simple Patterns**           | ⭐⭐        | Overhead not justified if global patterns sufficient |

## Architecture Details

### Network Structure

```
Training Phase:
  Input Features
      ↓
  Embedding Network (ResNet-like) → Training Embeddings [N, d]
      ↓
  Store embeddings + labels → Retrieval Index (HNSW/FAISS)
      ↓
  Standard supervised loss

Inference Phase:
  Test Sample x
      ↓
  Embedding Network → Query Embedding q [d]
      ↓
  kNN Search in Index → k Neighbors [k, d] + distances
      ↓
  Context Encoder → Context Vector c [d]
      ↓
  Fusion Layer: Combine(q, c) → Prediction
```

### Mathematical Formulation

**Embedding**:
$$q = f_{\theta}(x) \in \mathbb{R}^d$$

Where $f_{\theta}$ is the neural embedding network.

**Retrieval**:
$$\mathcal{N}_k(q) = \{(x_i, y_i)\}_{i=1}^k \text{ where } x_i \text{ are } k \text{ nearest to } q$$

Using distance metric (e.g., cosine similarity):
$$d(q, e_i) = 1 - \frac{q \cdot e_i}{\|q\| \|e_i\|}$$

**Context Encoding** (attention-based aggregation):
$$\alpha_i = \frac{\exp(-\beta \cdot d(q, e_i))}{\sum_{j=1}^k \exp(-\beta \cdot d(q, e_j))}$$
$$c = \sum_{i=1}^k \alpha_i \cdot e_i$$

**Fusion**:
$$h = \text{MLP}([q; c; q \odot c])$$

Where $[;]$ is concatenation and $\odot$ is element-wise product.

**Final Prediction**:
$$\hat{y} = \text{Head}(h)$$

### Key Design Choices

1. **Why kNN retrieval?**
   - Captures local patterns complementary to global neural patterns
   - Non-parametric: adapts to local distribution without extra parameters
   - Empirically: 3-10% gains on complex tasks with local structure

2. **Approximate NN (HNSW)**:
   - Exact kNN is O(N·d) per query → prohibitive for large N
   - HNSW (Hierarchical Navigable Small World) reduces to O(log(N))
   - 95-99% recall with 10-100x speedup

3. **Context Aggregation**:
   - **Mean**: Simple average of neighbor embeddings
   - **Attention**: Weight by distance/similarity to query
   - Attention generally +1-2% better but slightly slower

4. **Fusion Strategy**:
   - Concatenate query, context, and their interaction
   - Allows model to learn when to trust retrieval vs neural prediction

### Comparison to ModernNCA

| Feature        | TabR                    | ModernNCA                         |
| -------------- | ----------------------- | --------------------------------- |
| Retrieval      | k neighbors (fixed)     | All training samples (weighted)   |
| Inference Cost | O(k·d²)                 | O(N·d²)                           |
| Training Focus | Embedding + prediction  | Distance metric learning          |
| Best For       | k sufficient            | Full training distribution needed |
| Speed          | 2-3x slower than neural | 5-10x slower than neural          |

```{warning}
**Known Limitations**

1. **High Inference Memory**: Must store O(N·d) training embeddings. For 1M samples with d=256, this is ~1GB. Cannot discard training data after training.

2. **Inference Latency**: kNN search adds 10-50ms per sample depending on N and index quality. Not suitable for real-time systems requiring <10ms latency.

3. **Small Data Ineffective**: With <10K training samples, retrieval provides limited benefit (not enough neighbors for robust local patterns).

4. **Index Management Complexity**: Requires maintaining HNSW/FAISS index, updating when new data arrives, and careful deployment (model + index + training data).

5. **No Benefit Without Local Structure**: If global patterns dominate (e.g., simple linear relationships), retrieval overhead is wasted. Test with simpler models first.

6. **Privacy Concerns**: Training data must be retained and accessible at inference time, which may violate privacy requirements in some domains.

7. **Distribution Shift**: If test distribution shifts significantly from training, retrieved neighbors may be irrelevant. Requires retraining or index updates.
```

## References

1. **Rubachev, I., Alekberov, A., Gorishniy, Y., & Babenko, A. (2023)**. _Retrieval-Augmented Tabular Deep Learning_. arXiv:2305.14379. [Original TabR paper]

2. **Malkov, Y. A., & Yashunin, D. A. (2018)**. _Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs_. IEEE TPAMI. [HNSW algorithm for fast kNN]

3. **Johnson, J., Douze, M., & Jégou, H. (2019)**. _Billion-scale Similarity Search with GPUs_. IEEE Transactions on Big Data. [FAISS library for efficient retrieval]

4. **Papernot, N., & McDaniel, P. (2018)**. _Deep k-Nearest Neighbors: Towards Confident, Interpretable and Robust Deep Learning_. arXiv:1803.04765. [Early work on combining deep learning + kNN]

5. **Goldberger, J., et al. (2004)**. _Neighbourhood Components Analysis_. NeurIPS 2004. [Foundation for metric learning with neighbors]

6. **Gorishniy, Y., Rubachev, I., & Babenko, A. (2022)**. _On Embeddings for Numerical Features in Tabular Deep Learning_. NeurIPS 2022. [Embedding strategies for tabular data]

## See Also

- **[Mambular](mambular.md)** — Linear complexity state-space model without retrieval overhead
- **[FTTransformer](fttransformer.md)** — Pure attention-based model, faster inference
- **[ResNet](resnet.md)** — Simple baseline without retrieval complexity
- **[ModernNCA](modernnca.md)** — Uses all training samples (slower but sometimes more accurate)
- **[Model Selection Guide](../model_selection.md)** — Choosing when retrieval is beneficial

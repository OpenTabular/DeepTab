# ModernNCA

**Modern Neighborhood Component Analysis for tabular learning** — Neural metric learning approach for tabular data.

```{warning}
**⚠️ EXPERIMENTAL MODEL:** API not semantically versioned. May change in minor releases. Pin exact DeepTab version (`deeptab==x.y.z`) when using in production. See [Model Tiers](../../core_concepts/model_tiers) for details.
```

## Architecture Overview

**Core mechanism:** Metric learning with neural embeddings  
**Complexity:** O(n·k·d) where k = number of neighbors considered  
**Inductive bias:** Local similarity in learned embedding space

### Key Components

1. **Embedding network:** Maps inputs to metric space
2. **Distance computation:** Learns appropriate distance metric
3. **Neighbor weighting:** Attention over nearest neighbors
4. **Prediction:** Weighted combination of neighbor labels

```{note}
**Research motivation:** Extends classical Neighborhood Component Analysis (NCA) with deep neural embeddings. Hypothesis: learned metric space better captures semantic similarity for tabular data than hand-crafted features + Euclidean distance.
```

## Experimental Status

| Aspect                  | Status            | Implications                                           |
| ----------------------- | ----------------- | ------------------------------------------------------ |
| **API stability**       | ⚠️ Not guaranteed | Pin version: `deeptab==2.0.0`                          |
| **Semantic versioning** | ❌ Not covered    | Breaking changes possible in minor releases            |
| **Promotion criteria**  | In evaluation     | Needs consistent outperformance + community validation |
| **Production use**      | Use with caution  | Pin version, monitor release notes                     |

````{important}
**Version pinning essential:** Always specify exact version in requirements:
```python
# requirements.txt
deeptab==2.0.0  # Exact version, not >=2.0.0
````

````

## When to Use

| Scenario | Recommendation | Reasoning |
| -------- | -------------- | --------- |
| **Research/experimentation** | ✅ Try ModernNCA | Cutting-edge metric learning approach |
| **Local similarity matters** | ✅ Try ModernNCA | Designed for similarity-based predictions |
| **Willing to handle API changes** | ✅ Try ModernNCA | Can pin versions and adapt |
| **Production deployment** | ❌ Use [Mambular](../stable/mambular) | Stable API, proven reliability |
| **Need stable API** | ❌ Use stable models | Experimental = no guarantees |
| **Cannot monitor updates** | ❌ Use stable models | API may break silently |

## Configuration

### Model Config (ModernNCAConfig)

```{warning}
**Config API may change:** Parameter names, defaults, and valid ranges subject to change in future releases without major version bump.
````

| Parameter         | Current Default | Description               | Status               |
| ----------------- | --------------- | ------------------------- | -------------------- |
| `d_model`         | 128             | Embedding dimension       | May change           |
| `n_layers`        | 6               | Encoder depth             | May change           |
| `k_neighbors`     | 32              | Number of neighbors       | May be added/renamed |
| `distance_metric` | "euclidean"     | Metric in embedding space | May change           |

### Example Configuration

```python
from deeptab.configs import ModernNCAConfig

# Check version!
import deeptab
print(f"DeepTab version: {deeptab.__version__}")  # Ensure matches pinned version

cfg = ModernNCAConfig(
    d_model=128,
    n_layers=6,
)
```

## Quick Start

```python
from deeptab.models.experimental import ModernNCAClassifier, ModernNCARegressor

# ⚠️ ALWAYS PIN VERSION IN PRODUCTION
# pip install deeptab==2.0.0

# Check version first
import deeptab
assert deeptab.__version__ == "2.0.0", "Version mismatch!"

# Standard usage
model = ModernNCAClassifier()
model.fit(X_train, y_train, max_epochs=50)
predictions = model.predict(X_test)

# Note: API may change - refer to release notes for current version
```

## Research Context

### Theoretical Foundation

**Classical NCA (Goldberger et al., 2004):**

- Linear transformation: x → Ax
- Euclidean distance in transformed space
- Optimizes k-NN classification accuracy

**ModernNCA extension:**

- Non-linear transformation: x → φ(x; θ) via neural network
- Learned distance metric
- Optimized via gradient descent

### Potential Advantages

| Aspect                   | Classical NCA                | ModernNCA           | Hypothesis                            |
| ------------------------ | ---------------------------- | ------------------- | ------------------------------------- |
| **Transformation**       | Linear                       | Non-linear (neural) | Better captures complex relationships |
| **Capacity**             | Limited by linear constraint | High (deep network) | Can learn more expressive embeddings  |
| **Optimization**         | Closed-form or iterative     | Gradient-based      | Scales to larger datasets             |
| **Feature interactions** | None                         | Implicit in network | Captures dependencies                 |

```{note}
**Research status:** Promising early results, but requires more extensive evaluation across diverse datasets before conclusions about systematic improvements.
```

## Performance Characteristics

### Preliminary Observations

```{warning}
**Limited benchmarking:** Results based on initial experiments. More comprehensive evaluation needed for robust conclusions.
```

| Aspect              | Observation                                     | Caveat                               |
| ------------------- | ----------------------------------------------- | ------------------------------------ |
| **Accuracy**        | Competitive with stable models on some datasets | High variance across datasets        |
| **Training speed**  | Moderate (similar to Mambular)                  | Neighbor computation adds overhead   |
| **Inference speed** | Moderate (k-NN search required)                 | Slower than pure feedforward models  |
| **Memory**          | Medium (stores embeddings)                      | Higher than models without retrieval |

### Comparison with Alternatives

| vs Model     | Status | When to Prefer ModernNCA | When to Prefer Alternative |
| ------------ | ------ | ------------------------ | -------------------------- |
| **Mambular** | Stable | Research/cutting-edge    | Production, stable API     |
| **TabR**     | Stable | Metric learning approach | Proven retrieval method    |
| **ResNet**   | Stable | Local similarity matters | Fast baseline, stability   |

## Known Limitations

```{warning}
**Current limitations (subject to change):**
- **Experimental status:** No API stability guarantees
- **Limited validation:** Fewer datasets/benchmarks than stable models
- **Neighbor overhead:** k-NN search adds inference latency
- **Memory requirements:** Must store training embeddings
- **Hyperparameter sensitivity:** Optimal settings not well-established
```

## Best Practices for Experimental Models

### Version Management

```python
# ✅ GOOD: Pin exact version
# requirements.txt
deeptab==2.0.0

# ❌ BAD: Allow any compatible version
# deeptab>=2.0.0  # Could break on 2.0.1!
```

### Monitoring for Changes

```{tip}
**Stay informed:**
1. Monitor DeepTab release notes
2. Join community discussions (GitHub issues)
3. Test thoroughly after any update
4. Have migration plan to stable models
```

### Production Deployment Checklist

- [ ] Version pinned in requirements.txt
- [ ] Tests verify exact version in CI/CD
- [ ] Monitoring for API deprecation warnings
- [ ] Fallback plan to stable model
- [ ] Alert system for DeepTab updates

## Migration to Stable Models

```{important}
**Exit strategy:** If ModernNCA doesn't work out or API changes are disruptive:

**Similar alternatives:**
- [TabR](../stable/tabr) — Stable retrieval-based model
- [Mambular](../stable/mambular) — Stable general-purpose model
- [FTTransformer](../stable/fttransformer) — Stable attention-based model
```

## References

**Classical NCA:**

- Goldberger, J., et al. (2004). _Neighbourhood Components Analysis_. NIPS 2004

**Related metric learning:**

- Weinberger, K., & Saul, L. (2009). _Distance Metric Learning for Large Margin Nearest Neighbor Classification_. JMLR

**ModernNCA implementation:**

- DeepTab-specific adaptation (check GitHub for implementation details)

## See Also

- [Model Tiers](../../core_concepts/model_tiers) — Understanding stable vs experimental
- [Experimental Models Tutorial](../../tutorials/experimental) — Best practices
- [TabR](../stable/tabr) — Stable alternative with retrieval
- [Mambular](../stable/mambular) — Stable general-purpose model

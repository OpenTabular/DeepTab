# Tangos

**Tangent-based Optimization for Tabular Learning** — Experimental architecture with novel gradient-based optimization approach.

```{warning}
**⚠️ EXPERIMENTAL MODEL:** API not semantically versioned. May change in minor releases. Pin exact DeepTab version (`deeptab==x.y.z`) when using in production. See [Model Tiers](../../core_concepts/model_tiers) for details.
```

## Architecture Overview

**Core mechanism:** Neural network with tangent-based gradient updates  
**Complexity:** O(n·d) per forward pass (similar to MLP)  
**Inductive bias:** Optimization-level innovation rather than architectural

### Key Components

1. **Standard feedforward layers:** MLP-like architecture
2. **Tangent-based updates:** Modified gradient computation
3. **Novel optimization:** Alternative to standard SGD/Adam
4. **Task-agnostic:** Can be applied to various architectures

```{note}
**Research motivation:** Explores whether alternative optimization strategies can improve tabular learning. Hypothesis: tangent-based updates may navigate loss landscape more effectively than standard gradients, particularly when standard optimization plateaus.
```

## Experimental Status

| Aspect                  | Status            | Implications                                           |
| ----------------------- | ----------------- | ------------------------------------------------------ |
| **API stability**       | ⚠️ Not guaranteed | Pin version: `deeptab==2.0.0`                          |
| **Semantic versioning** | ❌ Not covered    | Breaking changes possible in minor releases            |
| **Promotion criteria**  | In evaluation     | Needs consistent outperformance + community validation |
| **Production use**      | Use with caution  | Pin version, monitor release notes                     |
| **Research stage**      | Early validation  | Limited benchmarking across datasets                   |

```{important}
**Version pinning essential:** Always specify exact version in requirements:

    # requirements.txt
    deeptab==2.0.0  # Exact version, not >=2.0.0
```

## When to Use

| Scenario                           | Recommendation                        | Reasoning                               |
| ---------------------------------- | ------------------------------------- | --------------------------------------- |
| **Standard optimization plateaus** | ✅ Try Tangos                         | Designed for this scenario              |
| **Research/experimentation**       | ✅ Try Tangos                         | Cutting-edge optimization approach      |
| **Can handle API changes**         | ✅ Try Tangos                         | Version pinning and monitoring feasible |
| **Exploring novel methods**        | ✅ Try Tangos                         | Alternative optimization worth testing  |
| **Production deployment**          | ❌ Use [Mambular](../stable/mambular) | Stable API, proven reliability          |
| **Need stable API**                | ❌ Use stable models                  | Experimental = no guarantees            |
| **Cannot monitor updates**         | ❌ Use stable models                  | API may break silently                  |
| **Limited experimentation time**   | ❌ Use proven models                  | Tangos requires validation on your data |

## Configuration

### Model Config (TangosConfig)

```{warning}
**Config API may change:** Parameter names, defaults, and valid ranges subject to change in future releases without major version bump.
```

| Parameter    | Current Default | Description                    | Status               |
| ------------ | --------------- | ------------------------------ | -------------------- |
| `d_model`    | 128             | Hidden dimension               | May change           |
| `n_layers`   | 6               | Network depth                  | May change           |
| `dropout`    | 0.0             | Dropout rate                   | May change           |
| `tangent_lr` | Auto            | Tangent-specific learning rate | May be added/renamed |

### Example Configuration

```python
from deeptab.configs import TangosConfig

# Check version!
import deeptab
print(f"DeepTab version: {deeptab.__version__}")  # Ensure matches pinned version

cfg = TangosConfig(
    d_model=128,
    n_layers=6,
)
```

## Quick Start

```python
from deeptab.models.experimental import TangosClassifier, TangosRegressor

# ⚠️ ALWAYS PIN VERSION IN PRODUCTION
# pip install deeptab==2.0.0

# Check version first
import deeptab
assert deeptab.__version__ == "2.0.0", "Version mismatch!"

# Standard usage
model = TangosClassifier()
model.fit(X_train, y_train, max_epochs=50)
predictions = model.predict(X_test)

# Compare with standard optimization (Mambular)
from deeptab.models import MambularClassifier
baseline = MambularClassifier()
baseline.fit(X_train, y_train, max_epochs=50)
# Evaluate if Tangos provides improvement on your data

# Note: API may change - refer to release notes for current version
```

## Research Context

### Theoretical Foundation

**Standard gradient descent:**

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

**Tangent-based update (conceptual):**

$$
\theta_{t+1} = \theta_t - \eta \cdot \text{TangentOp}(\nabla_\theta \mathcal{L}(\theta_t))
$$

Where TangentOp modifies gradients based on loss surface tangent properties.

### Potential Advantages

| Aspect                        | Standard Optimization | Tangos              | Hypothesis                             |
| ----------------------------- | --------------------- | ------------------- | -------------------------------------- |
| **Gradient computation**      | Direct backprop       | Tangent-modified    | Better direction in complex landscapes |
| **Loss landscape navigation** | Standard descent      | Alternative paths   | May escape poor local minima           |
| **Plateau handling**          | Prone to stalling     | Alternative updates | Better progress when stuck             |
| **Convergence**               | Well-studied          | Under research      | May converge faster in some cases      |

```{note}
**Research status:** Preliminary experiments show promise in specific scenarios, but comprehensive evaluation across diverse datasets needed. Not yet clear when/why tangent-based updates help.
```

## Performance Characteristics

### Preliminary Observations

```{warning}
**Limited benchmarking:** Results based on initial experiments. More comprehensive evaluation needed for robust conclusions.
```

| Aspect                     | Observation                  | Caveat                        |
| -------------------------- | ---------------------------- | ----------------------------- |
| **Accuracy**               | Competitive on some datasets | High variance across datasets |
| **Training speed**         | Similar to MLP/ResNet        | Comparable to standard models |
| **Optimization stability** | Generally stable             | May require tuning            |
| **When it helps**          | Plateauing scenarios         | Not consistently identified   |

### Comparison with Alternatives

| vs Model     | Status | When to Prefer Tangos     | When to Prefer Alternative |
| ------------ | ------ | ------------------------- | -------------------------- |
| **Mambular** | Stable | Research/experimentation  | Production, stable API     |
| **ResNet**   | Stable | Novel optimization needed | Fast stable baseline       |
| **MLP**      | Stable | Optimization matters      | Simplest baseline          |

## Known Limitations

```{warning}
**Current limitations (subject to change):**
- **Experimental status:** No API stability guarantees
- **Limited validation:** Fewer datasets/benchmarks than stable models
- **Unclear advantage scenarios:** When tangent-based helps not well-characterized
- **Optimization understanding:** Theory less developed than standard methods
- **Hyperparameter sensitivity:** Optimal settings not well-established
- **Community experience:** Limited production usage for feedback
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
1. Monitor DeepTab release notes closely
2. Join community discussions (GitHub issues)
3. Test thoroughly after any update
4. Have migration plan to stable models
5. Set up alerts for new releases
```

### Evaluation Protocol

```python
# Systematic evaluation before production use
import deeptab
assert deeptab.__version__ == "2.0.0"

from deeptab.models.experimental import TangosClassifier
from deeptab.models import MambularClassifier

# Compare Tangos with stable baseline
tangos = TangosClassifier()
tangos.fit(X_train, y_train, max_epochs=50)
tangos_score = tangos.score(X_test, y_test)

mambular = MambularClassifier()
mambular.fit(X_train, y_train, max_epochs=50)
mambular_score = mambular.score(X_test, y_test)

# Only use Tangos if clear improvement
if tangos_score > mambular_score + 0.02:  # 2% threshold
    print("Tangos provides clear benefit on this dataset")
else:
    print("Stick with Mambular (stable)")
```

## Experimental Workflow

```{tip}
**Recommended approach:**
1. Start with stable model baseline ([Mambular](../stable/mambular))
2. If standard optimization plateaus, try Tangos
3. Validate improvement on held-out test set
4. Pin version if deploying
5. Monitor for updates and evaluate migration path
```

**Decision tree:**

```
Standard models (Mambular/ResNet) plateau?
    ↓ No → Stay with stable models
    ↓ Yes
Need cutting-edge optimization?
    ↓ No → Tune hyperparameters more
    ↓ Yes
Can handle API instability?
    ↓ No → Stay with stable
    ↓ Yes
→ Try Tangos (pin version!)
    ↓
Provides >2% improvement?
    ↓ No → Return to stable
    ↓ Yes
→ Deploy with version pinning and monitoring
```

## Migration to Stable Models

```{important}
**Exit strategy:** If Tangos doesn't work out or API changes are disruptive:

**Similar stable alternatives:**
- [Mambular](../stable/mambular) — Best general-purpose stable model
- [ResNet](../stable/resnet) — Fast stable baseline
- [MLP](../stable/mlp) — Simplest stable baseline

**Migration is seamless:**

    # Tangos (experimental)
    from deeptab.models.experimental import TangosClassifier
    model = TangosClassifier()

    # → Mambular (stable)
    from deeptab.models import MambularClassifier
    model = MambularClassifier()  # Same API!
```

## API Change Examples

```{warning}
**Past API changes (hypothetical examples):**

**v2.0.0 → v2.1.0:**
- Parameter `tangent_lr` → `tangent_learning_rate` (renamed)
- Added required parameter `tangent_mode` (breaking)
- Changed default `d_model` from 128 → 64 (behavior change)

**Impact:** Code using v2.0.0 breaks on v2.1.0 without modification.

**Protection:** Pin to `deeptab==2.0.0` exactly.
```

## Community Feedback

```{note}
**Help improve Tangos:** If you experiment with this model:

1. Share results (GitHub issues/discussions)
2. Report any issues or unexpected behavior
3. Suggest improvements
4. Document scenarios where it helps/doesn't help

Community validation essential for promotion to stable tier!
```

## References

**Tangent-based optimization:**

- Experimental approach under evaluation (check DeepTab documentation for implementation details)

**Alternative optimization methods:**

- Various second-order and adaptive methods in literature

**Related experimental approaches:**

- Research into optimization landscapes for tabular deep learning

## See Also

- [Model Tiers](../../core_concepts/model_tiers) — Understanding stable vs experimental
- [Experimental Models Tutorial](../../tutorials/experimental) — Best practices
- [Mambular](../stable/mambular) — Stable general-purpose alternative
- [ResNet](../stable/resnet) — Fast stable baseline
- [Version Pinning Guide](../../developer_guide/version_pinning) — Managing experimental dependencies

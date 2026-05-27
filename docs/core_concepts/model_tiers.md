# Model Tiers: Stable vs Experimental

DeepTab ships models at two tiers with different API stability guarantees. Understanding the difference helps you choose the right models for your project.

## Overview

| Tier             | Import path                                   | API guarantee                               | Use case                             |
| ---------------- | --------------------------------------------- | ------------------------------------------- | ------------------------------------ |
| **Stable**       | `from deeptab.models import ...`              | Public API frozen under semantic versioning | Production, long-term projects       |
| **Experimental** | `from deeptab.models.experimental import ...` | May change without deprecation cycle        | Research, prototyping, bleeding edge |

```{important}
**For production systems, always use stable models.** Experimental models may have breaking API changes between minor versions without deprecation warnings.
```

## Stable models

Stable models have a frozen public API that follows [semantic versioning](https://semver.org/):

- **Major version (X.0.0)**: Breaking changes allowed
- **Minor version (0.X.0)**: New features, no breaking changes
- **Patch version (0.0.X)**: Bug fixes only

### Import path

```python
from deeptab.models import MambularClassifier
```

All stable models are imported directly from `deeptab.models`.

### API stability

Once a model is stable, its public interface is frozen:

```python
# This API will not change within v2.x
model = MambularClassifier(
    model_config=MambularConfig(),
    preprocessing_config=PreprocessingConfig(),
    trainer_config=TrainerConfig(),
)

model.fit(X_train, y_train, max_epochs=100)
predictions = model.predict(X_test)
```

```{tip}
**Stable API guarantees:**
- ✅ Method signatures (`fit`, `predict`, `predict_proba`, `evaluate`) won't change
- ✅ Config parameters won't be removed or renamed
- ✅ Output formats stay consistent
- ✅ Deprecation warnings appear at least one minor version before removal
```

```{note}
**What can still change:**
- Internal implementation (for performance improvements)
- Default hyperparameter values (for better out-of-box performance)
- New parameters (added with backward-compatible defaults)
```

### Available stable models

As of v2.0:

**Sequential models:**

- `Mambular` — Mamba blocks for sequential feature processing
- `TabulaRNN` — Recurrent neural network for tabular data

**Attention-based:**

- `FTTransformer` — Feature tokenization + Transformer encoder
- `TabTransformer` — Transformer with categorical embeddings
- `SAINT` — Row attention with contrastive pre-training
- `MambAttention` — Mamba + Transformer hybrid

**Ensemble methods:**

- `TabM` — Batch ensembling for MLP
- `TabR` — Retrieval-augmented tabular model

**Tree-inspired:**

- `NODE` — Neural oblivious decision ensembles
- `NDTF` — Neural decision tree forest
- `ENODE` — Extended NODE variant

**Baselines:**

- `MLP` — Multi-layer perceptron
- `ResNet` — ResNet adapted for tabular data
- `MambaTab` — Mamba block on joint representation

**Others:**

- `AutoInt` — Automatic feature interaction via attention

All stable models are available as `*Classifier`, `*Regressor`, and `*LSS` variants.

## Experimental models

Experimental models are under active development and may change without warning between minor versions.

```{warning}
**Experimental models are NOT production-ready.** Always pin your DeepTab version (`deeptab==x.y.z`) if using experimental models to avoid unexpected breaking changes.
```

### Import path

Always use the explicit experimental import path:

```python
from deeptab.models.experimental import TromptClassifier
```

This signals that you accept the instability.

### What may change

- **Architecture**: Internal structure may be redesigned
- **Parameters**: Config parameters may be added, removed, or renamed
- **Defaults**: Default hyperparameters may change significantly
- **API**: Method signatures may evolve
- **Availability**: Models may be removed if they underperform

### Why experimental?

Models enter experimental status when:

1. **New research**: Based on recent papers, not yet proven in production
2. **Active development**: Architecture is still being tuned
3. **Limited testing**: Not yet thoroughly tested across diverse datasets
4. **Uncertain value**: Unclear if they provide advantages over stable models

### Graduation to stable

```{note}
**Promotion criteria:** Models graduate from experimental to stable when they demonstrate:

- ✅ Proven performance on diverse benchmarks
- ✅ Mature, well-designed API
- ✅ Comprehensive test coverage
- ✅ Community adoption and success stories
```

### Available experimental models

As of v2.0:

- `ModernNCA` — Modern neural classification architecture
- `Trompt` — Tabular-specific prompting model
- `Tangos` — Tabular model with graph-based structure

Check the [API reference](../../api/models/index) for the current list.

## Choosing between stable and experimental

### Use stable models when:

✅ Building production systems  
✅ Long-term projects (6+ months)  
✅ Need API stability guarantees  
✅ Deploying to critical environments  
✅ Collaborating with multiple teams  
✅ Require backward compatibility

### Use experimental models when:

✅ Research and prototyping  
✅ Exploring cutting-edge architectures  
✅ Short-term experiments  
✅ Personal projects  
✅ Willing to update code as models evolve  
✅ Seeking potential performance edge

## Version pinning

### For production with stable models

Pin to minor version:

```toml
# pyproject.toml
[tool.poetry.dependencies]
deeptab = "^2.0"  # Allows 2.0, 2.1, 2.2, ... but not 3.0
```

This ensures you get bug fixes and new features without breaking changes.

### For production with experimental models

Pin to exact version:

```toml
[tool.poetry.dependencies]
deeptab = "==2.0.0"  # Exact version only
```

This prevents unexpected changes when experimental models evolve.

### For development

Use latest:

```bash
pip install -U deeptab
```

## Deprecation policy

### Stable models

When a stable model feature needs to be removed:

1. **Deprecation warning**: Added in version N
2. **Continued support**: Feature still works in version N
3. **Removal**: Feature removed in version N+1 (next minor) or N+2 (if more time needed)

Example:

```python
# Version 2.1: Deprecation warning
model = OldFeatureModel()  # UserWarning: OldFeatureModel is deprecated...

# Version 2.2: Still works with warning
model = OldFeatureModel()  # UserWarning: OldFeatureModel will be removed in 2.3

# Version 2.3: Removed
model = OldFeatureModel()  # ImportError: OldFeatureModel removed. Use NewFeatureModel instead
```

### Experimental models

No deprecation warnings. Models may change or be removed in any version.

## Migration guides

When experimental models graduate to stable or stable models change significantly, migration guides are provided in the changelog.

Example migration:

```python
# Old (experimental in v2.0)
from deeptab.models.experimental import ProtoModel
model = ProtoModel(hidden_dim=128)

# New (stable in v2.1)
from deeptab.models import ProtoModel
from deeptab.configs import ProtoModelConfig

model = ProtoModel(
    model_config=ProtoModelConfig(d_model=128)  # Renamed parameter
)
```

## Promoting your own models

If you build custom models on top of DeepTab, you can apply the same tier system:

```python
# Your experimental model
from your_package.models.experimental import CustomClassifier

# After validation, promote to stable
from your_package.models import CustomClassifier
```

## Checking model tier at runtime

You can inspect the model tier programmatically:

```python
from deeptab.models import MambularClassifier
from deeptab.models.experimental import TromptClassifier

print(MambularClassifier._tier)  # "stable"
print(TromptClassifier._tier)    # "experimental"
```

This is useful for automated checks in CI/CD pipelines:

```python
def validate_models(models):
    for model in models:
        if model._tier == "experimental":
            raise ValueError(f"{model.__name__} is experimental. Use stable models for production.")
```

## FAQ

**Q: Can I use experimental models in production?**  
A: Technically yes, but not recommended. Pin to an exact version if you do.

**Q: Will experimental models ever be removed?**  
A: Yes, if they don't prove valuable or a better alternative emerges.

**Q: How often do experimental models change?**  
A: Varies. Some change in every minor release, others stabilize quickly.

**Q: Can stable models become experimental again?**  
A: No. Once stable, always stable (or deprecated if necessary).

**Q: What happens to v1 models in v2?**  
A: v1 is no longer supported after v2.0 release. See the [FAQ](../getting_started/faq) for details.

## Next steps

- **[Config System](config_system)** — Learn about the split-config API
- **[sklearn API](sklearn_api)** — Understand the scikit-learn interface
- **[Tutorials: Experimental](../../tutorials/experimental)** — See experimental models in action

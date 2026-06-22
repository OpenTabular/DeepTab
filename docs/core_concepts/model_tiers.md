# Model Tiers: Stable and Experimental

DeepTab separates production-ready models from research-stage models.

| Tier         | Import path                                   | API expectation                                                  | Best use                                            |
| ------------ | --------------------------------------------- | ---------------------------------------------------------------- | --------------------------------------------------- |
| Stable       | `from deeptab.models import ...`              | Public API intended to remain compatible within a major version. | Production, long-running projects, baseline suites. |
| Experimental | `from deeptab.models.experimental import ...` | May change as research implementations mature.                   | Prototyping, research comparisons, early feedback.  |

## Stable Models

Stable models live directly under `deeptab.models`:

```python
from deeptab.models import MambularClassifier, TabMRegressor, FTTransformerLSS
```

Stable model pages:

- [Stable Model Zoo](../model_zoo/stable/index)
- [Comparison Tables](../model_zoo/comparison_tables)
- [Recommended Configs](../model_zoo/recommended_configs)

Stable models include MLP/ResNet/TabM baselines, Transformer models, Mamba-family models, neural tree models, and retrieval models. All stable models are available as `*Classifier`, `*Regressor`, and `*LSS` variants unless noted in the API reference.

## Experimental Models

Experimental models use the explicit experimental import path:

```python
from deeptab.models.experimental import TromptClassifier, ModernNCARegressor
```

The explicit import is intentional: it makes research-stage dependency risk visible in code review and experiment records.

Experimental model pages:

- [Experimental Model Zoo](../model_zoo/experimental/index)
- [ModernNCA](../model_zoo/experimental/modernnca)
- [TANGOS](../model_zoo/experimental/tangos)
- [Trompt](../model_zoo/experimental/trompt)

## Custom Models

Beyond the stable and experimental tiers, you can plug in your own architecture
and use it through the same scikit-learn API, preprocessing pipeline, and
trainer as the built-in models. See [Custom Models](custom_models) for the full
guide.

## Choosing a Tier

| Consideration      | Stable                                 | Experimental                             |
| ------------------ | -------------------------------------- | ---------------------------------------- |
| Primary use        | Production and long-running projects   | Prototyping and research comparisons     |
| Reproducibility    | Stable across minor releases           | Requires pinning an exact version        |
| API stability      | Compatible within a major version      | May introduce breaking changes           |
| Maintenance burden | Lower; safe baseline for collaborators | Higher; tracks recent, evolving research |
| Goal               | Reliable deployment                    | Early evaluation and research feedback   |

```{note}
**Version pinning.** For stable-only projects, pin a compatible range such as
`deeptab>=2.0,<3.0`. For projects that use experimental models, pin the exact
version (`deeptab==2.0.0`), since their APIs may change between releases.
```

## Next Steps

- [Stable Models](../model_zoo/stable/index)
- [Experimental Models](../model_zoo/experimental/index)
- [Experimental Tutorial](../tutorials/experimental)

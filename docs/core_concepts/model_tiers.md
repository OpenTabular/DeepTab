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

Use stable models when:

- the code will run in production;
- experiments need long-term reproducibility;
- collaborators need a lower-maintenance baseline;
- APIs must remain stable across minor releases.

Use experimental models when:

- you are evaluating recent architectures;
- you can pin DeepTab to an exact version;
- breaking changes are acceptable;
- the goal is research feedback rather than deployment.

## Version Pinning

For stable-only projects, pin a compatible range:

```text
deeptab>=2.0,<3.0
```

For experimental-model projects, pin the exact version:

```text
deeptab==2.0.0
```

## Documentation Policy

Stable model docs should document both the paper idea and the actual DeepTab implementation. Experimental docs should be even more explicit about implementation differences, config limitations, and expected API volatility.

## Next Steps

- [Stable Models](../model_zoo/stable/index)
- [Experimental Models](../model_zoo/experimental/index)
- [Experimental Tutorial](../tutorials/experimental)

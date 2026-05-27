# Experimental Models

```{warning}
**Experimental tier:** These models are not covered by DeepTab's stable-model API guarantees. Pin the exact DeepTab version when using them in reproducible studies or production-like workflows.
```

Experimental models are research-facing architectures that are available for evaluation before they graduate to the stable model zoo. They are useful for benchmarking new inductive biases, studying architectural behavior, and contributing empirical evidence back to DeepTab.

## Available Models

| Model                  | Core Idea                                                                    | Best Research Use                                          | Main Cost Driver                           |
| ---------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------ |
| [ModernNCA](modernnca) | Differentiable nearest-neighbor prediction in a learned representation space | Testing whether local similarity structure helps a dataset | Pairwise distance to candidate rows        |
| [Tangos](tangos)       | MLP with gradient-attribution specialization and orthogonalization penalties | Studying regularization of dense tabular networks          | Jacobian computation during training       |
| [Trompt](trompt)       | Prompt-style recurrent tabular representation cells                          | Evaluating prompt-inspired tabular architectures           | Prompt-feature importance maps over cycles |

## Quick Usage

```python
from deeptab.configs import ModernNCAConfig, TangosConfig, TrainerConfig, TromptConfig
from deeptab.models.experimental import ModernNCAClassifier, TangosClassifier, TromptClassifier

trainer_cfg = TrainerConfig(max_epochs=100, batch_size=128, lr=3e-4, patience=15)

modern_nca = ModernNCAClassifier(
    model_config=ModernNCAConfig(dim=128, n_blocks=4, temperature=0.75),
    trainer_config=trainer_cfg,
)

tangos = TangosClassifier(
    model_config=TangosConfig(layer_sizes=[256, 128, 32], lamda1=0.5, lamda2=0.1),
    trainer_config=trainer_cfg,
)

trompt = TromptClassifier(
    model_config=TromptConfig(d_model=128, n_cycles=6, n_cells=4, P=128),
    trainer_config=trainer_cfg,
)
```

## Selection Guidance

| If your research question is...                                  | Start with       | Compare against                       |
| ---------------------------------------------------------------- | ---------------- | ------------------------------------- |
| Does a learned local-neighbor rule beat parametric prediction?   | ModernNCA        | TabR, TabM, ResNet                    |
| Can attribution-based regularization improve a plain MLP?        | Tangos           | MLP, ResNet, TabM                     |
| Do prompt-style latent records help tabular feature aggregation? | Trompt           | FTTransformer, Mambular, TabM         |
| Do I need a reliable model for production today?                 | Stable model zoo | Mambular, TabM, ResNet, FTTransformer |

```{important}
When benchmarking an experimental model, include at least one tuned simple baseline such as MLP, ResNet, or TabM. Otherwise it is hard to tell whether the experimental mechanism adds value beyond optimization and preprocessing.
```

## Stability Roadmap

Experimental models are candidates for stable promotion when they show:

- Competitive performance under a declared search budget.
- Reliable convergence across datasets and random seeds.
- Clear configuration defaults and failure modes.
- Documentation that explains both architecture and implementation details.
- Community feedback from real use cases.

See [Model Promotion Policy](../../developer_guide/model_promotion_policy) for the promotion criteria.

## See Also

- [Experimental Models Tutorial](../../tutorials/experimental) - end-to-end examples
- [Model Comparison](../comparison_tables) - architecture and complexity comparison
- [Recommended Configs](../recommended_configs) - general tuning guidance

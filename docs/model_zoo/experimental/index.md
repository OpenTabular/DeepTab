# Experimental Models

```{warning}
**Cutting-Edge Research — Use with Caution**

Experimental models are **not covered by semantic versioning**. APIs may change without deprecation warnings. Pin your DeepTab version (`deeptab==x.y.z`) if using in production.
```

## What Are Experimental Models?

Experimental models are **cutting-edge architectures** currently under evaluation for promotion to stable status. They represent the latest research in tabular deep learning but haven't yet undergone the rigorous stability testing required for production use.

```{tip}
**When to use experimental models:**

- Research projects and experimentation
- Exploring novel architectures
- Benchmarking against state-of-the-art
- Contributing to model evaluation
```

## Available Experimental Models

| Model                  | Description                                                        |
| ---------------------- | ------------------------------------------------------------------ |
| [ModernNCA](modernnca) | Neural metric learning approach for tabular data                   |
| [Trompt](trompt)       | Transformer with prompt-based learning for tabular data            |
| [Tangos](tangos)       | Graph-based neural architecture with learned feature relationships |

## Usage

Import experimental models from the `experimental` submodule:

```python
from deeptab.models.experimental import TromptClassifier, ModernNCARegressor, TangosClassifier

# Use like any stable model
model = TromptClassifier()
model.fit(X_train, y_train, max_epochs=50)
predictions = model.predict(X_test)
```

```{important}
**Version Pinning Required**

Always pin your DeepTab version when using experimental models:

\`\`\`bash
pip install deeptab==2.0.0  # Pin exact version
\`\`\`

This prevents breaking changes from affecting your code.
```

## Stability Roadmap

Experimental models are evaluated based on:

- **Performance** — Competitive accuracy across benchmarks
- **Stability** — Reliable training and convergence
- **Usability** — Clear configuration and good defaults
- **Community Feedback** — User reports and contributions

See **[Model Promotion Policy](../../developer_guide/model_promotion_policy)** for details on how models graduate to stable status.

## Examples and Best Practices

For detailed usage examples and tips:

- **[Experimental Models Tutorial](../../tutorials/experimental)** — Comprehensive guide
- **[Comparison Tables](../comparison_tables)** — Performance benchmarks
- **[Recommended Configs](../recommended_configs)** — Configuration guidance

## Contributing

Found a bug or have suggestions for experimental models? We welcome contributions!

- **[Contributing Guide](../../developer_guide/contributing)** — Get started
- **[GitHub Issues](https://github.com/OpenTabular/DeepTab/issues)** — Report bugs or request features

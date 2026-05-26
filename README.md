<div align="center">
    <img src="./docs/images/logo/deeptab-v1.png" width="550"/>

[![PyPI](https://img.shields.io/pypi/v/deeptab)](https://pypi.org/project/deeptab)
![PyPI - Downloads](https://img.shields.io/pypi/dm/deeptab)
[![docs build](https://readthedocs.org/projects/deeptab/badge/?version=latest)](https://deeptab.readthedocs.io/en/latest/?badge=latest)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://deeptab.readthedocs.io/en/latest/)
[![open issues](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/basf/deeptab/issues)

[📘 Documentation](https://deeptab.readthedocs.io) |
[🚀 Getting Started](https://deeptab.readthedocs.io/en/latest/getting_started/quickstart.html) |
[🎯 Model Zoo](https://deeptab.readthedocs.io/en/latest/model_zoo/index.html) |
[📖 Tutorials](https://deeptab.readthedocs.io/en/latest/tutorials/index.html) |
[🤔 Report Issues](https://github.com/basf/deeptab/issues)

</div>

<!-- START SHARED CONTENT -->

# DeepTab: Tabular Deep Learning Made Simple

**DeepTab** is a Python library for deep learning on tabular data. It features state-of-the-art architectures including Mamba (State Space Models), Transformers, and specialized tabular models—all with a familiar scikit-learn interface.

📄 **Papers:**

- [Mambular: A Sequential Model for Tabular Deep Learning](https://arxiv.org/abs/2408.06291)
- [TabulaRNN: Analyzing Efficiency of RNN Models for Tabular Data](https://arxiv.org/pdf/2411.17207)

## ⚡ What's New in v2.0

- **New Documentation**: [Getting Started](https://deeptab.readthedocs.io/en/latest/getting_started/index.html), [Core Concepts](https://deeptab.readthedocs.io/en/latest/core_concepts/index.html), [Tutorials with Colab](https://deeptab.readthedocs.io/en/latest/tutorials/index.html), [Model Zoo](https://deeptab.readthedocs.io/en/latest/model_zoo/index.html)
- **Typed Data Layer**: `TabularDataset`, `TabularDataModule`, `FeatureSchema`
- **Split-Config API**: Separate configs for model, preprocessing, and training
- **Enhanced Preprocessing**: Feature-specific transformations, PLE, pre-trained encodings
- **New Models**: AutoInt, ENODE, TabR
- **Experimental Models**: Tangos, Trompt, ModernNCA

## 🏃 Quickstart

```python
from deeptab.models import MambularClassifier

# Initialize and fit (sklearn-compatible)
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

**That's it!** DeepTab handles preprocessing, batching, and training automatically.

## 📖 Why DeepTab?

- **🔧 Familiar API**: Drop-in replacement for sklearn models
- **⚡ Auto-Preprocessing**: Automatic feature detection and transformation
- **🎯 State-of-the-Art Models**: 15+ proven architectures
- **📊 Distributional Regression**: Full distribution prediction (LSS)
- **🔍 Model Selection**: Comprehensive [Model Zoo](https://deeptab.readthedocs.io/en/latest/model_zoo/index.html) with guidance
- **📚 Complete Docs**: [Tutorials](https://deeptab.readthedocs.io/en/latest/tutorials/index.html), [examples](https://deeptab.readthedocs.io/en/latest/core_concepts/index.html), and [API reference](https://deeptab.readthedocs.io/en/latest/api/index.html)

## 🤖 Available Models

DeepTab includes 15 stable models + 3 experimental architectures:

### Stable Models

| Model              | Architecture                        | Best For                                  |
| ------------------ | ----------------------------------- | ----------------------------------------- |
| **Mambular**       | Multi-layer Mamba SSM               | General-purpose, best overall performance |
| **FTTransformer**  | Feature Tokenizer Transformer       | Strong baseline, feature interactions     |
| **ResNet**         | Residual MLP                        | Fast baseline, simple and effective       |
| **MambaTab**       | Single Mamba block                  | Small datasets, fast training             |
| **MambAttention**  | Hybrid Mamba + Attention            | Complex feature interactions              |
| **TabTransformer** | Transformer for categoricals        | Categorical-heavy data                    |
| **SAINT**          | Self-Attention + Intersample        | Semi-supervised learning                  |
| **TabM**           | Batch Ensembling MLP                | Efficient ensemble                        |
| **TabR**           | Retrieval-augmented                 | Large datasets (>50K samples)             |
| **MLP**            | Standard Multi-Layer Perceptron     | Fastest baseline                          |
| **NODE**           | Neural Oblivious Decision Ensembles | Interpretable tree-based                  |
| **ENODE**          | Enhanced NODE                       | Improved feature representations          |
| **NDTF**           | Neural Decision Tree Forest         | Differentiable tree ensemble              |
| **TabulaRNN**      | LSTM/GRU for tabular                | Sequential features                       |
| **AutoInt**        | Automatic Feature Interactions      | Feature engineering                       |

### Experimental Models ⚠️

- **ModernNCA**: Neighborhood Component Analysis
- **Tangos**: Gradient orthogonalization
- **Trompt**: Prompt-based learning

**See the [Model Zoo](https://deeptab.readthedocs.io/en/latest/model_zoo/index.html) for detailed comparisons, configuration recipes, and selection guidance.**

### Task Variants

All models come in three variants:

- `*Classifier` — Classification (binary & multi-class)
- `*Regressor` — Regression (point estimates)
- `*LSS` — Distributional regression (full distribution prediction)

## 📚 Documentation

**Full documentation:** [deeptab.readthedocs.io](https://deeptab.readthedocs.io)

### Quick Links

- **[Getting Started](https://deeptab.readthedocs.io/en/latest/getting_started/index.html)** — Installation, quickstart, FAQ
- **[Core Concepts](https://deeptab.readthedocs.io/en/latest/core_concepts/index.html)** — sklearn API, config system, preprocessing, training
- **[Tutorials](https://deeptab.readthedocs.io/en/latest/tutorials/index.html)** — Classification, regression, LSS (with Google Colab)
- **[Model Zoo](https://deeptab.readthedocs.io/en/latest/model_zoo/index.html)** — Model selection, comparisons, recommended configs
- **[API Reference](https://deeptab.readthedocs.io/en/latest/api/index.html)** — Complete API documentation

## 🛠️ Installation

**Basic installation:**

```bash
pip install deeptab
```

**With Mamba SSM (original implementation):**

```bash
pip install deeptab[mamba]
```

**Requirements:**

- Python 3.10+
- PyTorch 2.0+
- PyTorch Lightning 2.3.3+

See [installation guide](https://deeptab.readthedocs.io/en/latest/getting_started/installation.html) for GPU setup and troubleshooting.

## 🚀 Usage

### Basic Workflow

```python
from deeptab.models import MambularClassifier

# 1. Initialize with configuration
model = MambularClassifier(
    model_config={"d_model": 64, "n_layers": 6},
    preprocessing_config={"numerical_preprocessing": "quantile"},
    trainer_config={"max_epochs": 100, "lr": 1e-4}
)

# 2. Fit (X can be pandas DataFrame or numpy array)
model.fit(X_train, y_train)

# 3. Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# 4. Evaluate
metrics = model.evaluate(X_test, y_test)
```

### Hyperparameter Tuning

DeepTab models are sklearn-compatible, so you can use `GridSearchCV`:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model_config__d_model": [64, 128, 256],
    "model_config__n_layers": [4, 6, 8],
    "trainer_config__lr": [1e-4, 5e-4, 1e-3],
}

search = GridSearchCV(
    MambularClassifier(),
    param_grid,
    cv=5,
    scoring="accuracy"
)
search.fit(X_train, y_train)
print(search.best_params_)
```

Or use built-in Bayesian optimization:

```python
best_params = model.optimize_hparams(X_train, y_train)
```

### Distributional Regression (LSS)

Predict full distributions instead of point estimates:

```python
from deeptab.models import MambularLSS

# Fit with a distribution family
model = MambularLSS()
model.fit(X_train, y_train, family="normal")  # or "gamma", "poisson", "beta", etc.

# Predict distribution parameters
params = model.predict(X_test)  # Returns {"loc": ..., "scale": ...}

# Sample from predicted distributions
samples = model.sample(X_test, n_samples=1000)

# Get prediction intervals
lower, upper = model.predict_quantiles(X_test, quantiles=[0.025, 0.975])
```

**Available distributions:** normal, gamma, poisson, beta, studentt, negativebinom, dirichlet, quantile, and more.

See [Distributional Regression Tutorial](https://deeptab.readthedocs.io/en/latest/tutorials/distributional.html) for details.

## 🔧 Advanced Features

### Preprocessing

DeepTab includes comprehensive preprocessing powered by [PreTab](https://github.com/OpenTabular/PreTab):

- **Automatic detection**: Feature types detected automatically
- **Feature-specific**: Different preprocessing per feature
- **Methods**: PLE, quantile transform, spline encoding, polynomial features, pre-trained encodings

```python
from deeptab.configs import PreprocessingConfig

prep_config = PreprocessingConfig(
    numerical_preprocessing="quantile",
    use_ple=True,
    n_bins=50
)

model = MambularClassifier(preprocessing_config=prep_config)
```

### Custom Models

Implement your own architecture with DeepTab's base classes:

```python
from deeptab.base_models import BaseModel
from deeptab.models import SklearnBaseRegressor

class MyCustomModel(BaseModel):
    def __init__(self, feature_schema, num_classes, config, **kwargs):
        super().__init__(**kwargs)
        # Define your architecture

    def forward(self, batch):
        # Define forward pass
        return output

class MyRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(model=MyCustomModel, **kwargs)
```

See [Developer Guide](https://deeptab.readthedocs.io/en/latest/developer_guide/contributing.html) for details.

<!-- END SHARED CONTENT -->

## 🏷️ Citation

If you use DeepTab in your research, please cite:

```bibtex
@article{thielmann2024mambular,
  title={Mambular: A Sequential Model for Tabular Deep Learning},
  author={Thielmann, Anton and Weisser, Christoph and Kre{\ss}in, Arik and Reuter, Fabio and Kruse, Julius and Ben Amor, Farnoosh and Jungbluth, Tobias and dos Anjos, Antonia and Salkuti, Bhavya and S{\"a}fken, Benjamin},
  journal={arXiv preprint arXiv:2408.06291},
  year={2024}
}
```

## 📄 License

DeepTab is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please see [Contributing Guide](https://deeptab.readthedocs.io/en/latest/developer_guide/contributing.html).

## 📞 Support

- **Documentation:** [deeptab.readthedocs.io](https://deeptab.readthedocs.io)
- **Issues:** [GitHub Issues](https://github.com/basf/deeptab/issues)
- **Discussions:** [GitHub Discussions](https://github.com/basf/deeptab/discussions)

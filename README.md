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

> **💡 That's it!** DeepTab handles preprocessing, batching, and training automatically.

> **📊 Works with pandas & numpy:** Pass DataFrames or arrays—DeepTab auto-detects feature types.

## 📖 Why DeepTab?

- **🔧 Familiar API**: Drop-in replacement for sklearn models
- **⚡ Auto-Preprocessing**: Automatic feature detection and transformation
- **🎯 State-of-the-Art Models**: 15+ proven architectures
- **📊 Distributional Regression**: Full distribution prediction (LSS)
- **🔍 Model Selection**: Comprehensive [Model Zoo](https://deeptab.readthedocs.io/en/latest/model_zoo/index.html) with guidance
- **📚 Complete Docs**: [Tutorials](https://deeptab.readthedocs.io/en/latest/tutorials/index.html), [examples](https://deeptab.readthedocs.io/en/latest/core_concepts/index.html), and [API reference](https://deeptab.readthedocs.io/en/latest/api/index.html)

## 🤖 Available Models

DeepTab includes 15 stable models + 3 experimental architectures:

> **🎯 See the [Model Zoo](https://deeptab.readthedocs.io/en/latest/model_zoo/index.html) for detailed comparisons, complexity analysis, and selection guidance.**

### Stable Models

| Category               | Model              | Architecture                        | Best For                              |
| ---------------------- | ------------------ | ----------------------------------- | ------------------------------------- |
| **State Space Models** | **Mambular**       | Multi-layer Mamba SSM               | General-purpose, best overall         |
|                        | **MambaTab**       | Single Mamba block                  | Small datasets, fast training         |
|                        | **MambAttention**  | Hybrid Mamba + Attention            | Complex feature interactions          |
| **Transformers**       | **FTTransformer**  | Feature Tokenizer Transformer       | Strong baseline, feature interactions |
|                        | **TabTransformer** | Transformer for categoricals        | Categorical-heavy data (>60%)         |
|                        | **SAINT**          | Self-Attention + Intersample        | Small datasets, semi-supervised       |
|                        | **AutoInt**        | Automatic Feature Interactions      | Interaction discovery                 |
| **Residual Networks**  | **ResNet**         | Residual MLP                        | Fast baseline, simple and effective   |
|                        | **TabR**           | Retrieval-augmented ResNet          | Large datasets (>50K samples)         |
| **Tree-Based**         | **NODE**           | Neural Oblivious Decision Ensembles | Interpretable, tree inductive bias    |
|                        | **ENODE**          | Enhanced NODE                       | Better feature representations        |
|                        | **NDTF**           | Neural Decision Tree Forest         | Differentiable tree ensemble          |
| **Other**              | **MLP**            | Standard Multi-Layer Perceptron     | Fastest baseline                      |
|                        | **TabM**           | Batch Ensembling MLP                | Efficient ensemble, no tuning         |
|                        | **TabulaRNN**      | LSTM/GRU for tabular                | Sequential/temporal features          |

### Experimental Models ⚠️

> **⚠️ API Not Stable:** Experimental models may change in minor releases. Always pin exact version: `deeptab==x.y.z`

- **ModernNCA**: Neighborhood Component Analysis (metric learning)
- **Tangos**: Gradient orthogonalization approach
- **Trompt**: Prompt-based learning for tabular data

### Task Variants

All models come in three variants:

- `*Classifier` — Classification (binary & multi-class)
- `*Regressor` — Regression (point estimates)
- `*LSS` — Distributional regression (full distribution prediction)

> **🔄 Consistent API:** All models use the same interface—swap architectures without changing code!

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

**With Mamba SSM (recommended for best performance):**

```bash
pip install deeptab[mamba]
```

> **💻 Requirements:** Python 3.10+, PyTorch 2.0+, Lightning 2.3.3+

> **🚀 GPU Support:** See [installation guide](https://deeptab.readthedocs.io/en/latest/getting_started/installation.html) for CUDA setup.

## 🚀 Usage

### Basic Workflow

```python
from deeptab.models import MambularClassifier
from deeptab.configs import MambularConfig, PreprocessingConfig, TrainerConfig

# 1. Initialize with configuration (optional - defaults work well!)
model_config = MambularConfig(d_model=64, n_layers=6)
prep_config = PreprocessingConfig(numerical_preprocessing="quantile")
trainer_config = TrainerConfig(lr=1e-4, batch_size=256)

model = MambularClassifier(
    model_config=model_config,
    preprocessing_config=prep_config,
    trainer_config=trainer_config
)

# 2. Fit (X can be pandas DataFrame or numpy array)
model.fit(X_train, y_train, max_epochs=50)

# 3. Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# 4. Evaluate
metrics = model.evaluate(X_test, y_test)
```

> **💡 Tip:** Start with defaults (`MambularClassifier()`) and tune only if needed. See [Recommended Configs](https://deeptab.readthedocs.io/en/latest/model_zoo/recommended_configs.html) for guidance.

### Hyperparameter Tuning

DeepTab models are sklearn-compatible, so you can use `GridSearchCV`:

```python
from sklearn.model_selection import GridSearchCV
from deeptab.models import MambularClassifier

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
print(f"Best params: {search.best_params_}")
print(f"Best score: {search.best_score_}")
```

> **🔍 Built-in HPO:** DeepTab also supports Optuna for Bayesian optimization. See [HPO Tutorial](https://deeptab.readthedocs.io/en/latest/tutorials/hpo.html).

### Distributional Regression (LSS)

Predict full distributions instead of point estimates:

```python
from deeptab.models import MambularLSS

# Fit with a distribution family
model = MambularLSS()
model.fit(X_train, y_train, family="normal", max_epochs=50)

# Predict distribution parameters
params = model.predict(X_test)  # Returns dict with "loc", "scale", etc.

# Sample from predicted distributions
samples = model.sample(X_test, n_samples=1000)

# Get prediction intervals
intervals = model.predict_quantiles(X_test, quantiles=[0.025, 0.975])
```

> **📊 Available families:** `normal`, `gamma`, `poisson`, `beta`, `studentt`, `negativebinom`, `dirichlet`, `quantile`, and more.

> **📖 Learn more:** [Distributional Regression Tutorial](https://deeptab.readthedocs.io/en/latest/tutorials/distributional.html)

## 🔧 Advanced Features

### Preprocessing

DeepTab includes comprehensive preprocessing powered by [PreTab](https://github.com/OpenTabular/PreTab):

```python
from deeptab.configs import PreprocessingConfig
from deeptab.models import MambularClassifier

prep_config = PreprocessingConfig(
    numerical_preprocessing="quantile",  # Robust to outliers
    use_ple=True,                        # Piecewise linear encoding
    n_bins=50                            # Bins for PLE/quantile
)

model = MambularClassifier(preprocessing_config=prep_config)
model.fit(X_train, y_train, max_epochs=50)
```

> **✨ Features:**
>
> - **Automatic detection:** Feature types detected from data
> - **Feature-specific:** Different preprocessing per feature
> - **Methods:** PLE, quantile transform, spline encoding, polynomial features
> - **Pre-trained encodings:** Transfer learning for categorical features

> **📖 Learn more:** [Preprocessing Guide](https://deeptab.readthedocs.io/en/latest/core_concepts/preprocessing.html)

### Custom Models

Implement your own architecture with DeepTab's base classes:

```python
import torch.nn as nn
from deeptab.core import BaseModel
from deeptab.models import SklearnBaseRegressor
from deeptab.configs import PreprocessingConfig, TrainerConfig

class MyCustomConfig:
    def __init__(self, d_model=64, dropout=0.1):
        self.d_model = d_model
        self.dropout = dropout

class MyCustomModel(BaseModel):
    def __init__(
        self,
        feature_information: tuple,
        num_classes: int = 1,
        config: MyCustomConfig = MyCustomConfig(),
        **kwargs
    ):
        super().__init__(config=config, **kwargs)
        # feature_information = (num_feature_info, cat_feature_info, embedding_feature_info)

        # Define your architecture
        self.encoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, num_classes)
        )

    def forward(self, num_features, cat_features):
        # Implement forward pass
        x = num_features  # Process features as needed
        return self.encoder(x)

class MyRegressor(SklearnBaseRegressor):
    def __init__(
        self,
        model_config: MyCustomConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        random_state: int | None = None,
    ):
        super().__init__(
            model=MyCustomModel,
            config=MyCustomConfig,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=random_state,
        )

# Use like any other DeepTab model
model = MyRegressor()
model.fit(X_train, y_train, max_epochs=50)
```

> **🛠️ Developer Guide:** See [Contributing](https://deeptab.readthedocs.io/en/latest/developer_guide/contributing.html) for architecture guidelines.

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

@article{thielmann2024efficiency,
  title={On the Efficiency of NLP-Inspired Methods for Tabular Deep Learning},
  author={Thielmann, Anton Frederik and Samiee, Soheila},
  journal={arXiv preprint arXiv:2411.17207},
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

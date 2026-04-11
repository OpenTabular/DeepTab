<div align="center">
  <img src="./docs/images/logo/mamba_tabular.jpg" width="400"/>

[![PyPI](https://img.shields.io/pypi/v/deeptab)](https://pypi.org/project/deeptab)
![PyPI - Downloads](https://img.shields.io/pypi/dm/deeptab)
[![docs build](https://readthedocs.org/projects/deeptab/badge/?version=latest)](https://deeptab.readthedocs.io/en/latest/?badge=latest)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://deeptab.readthedocs.io/en/latest/)
[![open issues](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/OpenTabular/deeptab/issues)

[📘Documentation](https://deeptab.readthedocs.io/en/latest/index.html) |
[🛠️Installation](https://deeptab.readthedocs.io/en/latest/installation.html) |
[Models](https://deeptab.readthedocs.io/en/latest/api/models/index.html) |
[🤔Report Issues](https://github.com/OpenTabular/deeptab/issues)

</div>

<div style="text-align: center;">
    <h1>deeptab: Tabular Deep Learning Made Simple</h1>
</div>

deeptab is a Python library for tabular deep learning. It includes models that leverage the Mamba (State Space Model) architecture, as well as other popular models like TabTransformer, FTTransformer, TabM and tabular ResNets. Check out our paper `Mambular: A Sequential Model for Tabular Deep Learning`, available [here](https://arxiv.org/abs/2408.06291). Also check out our paper introducing [TabulaRNN](https://arxiv.org/pdf/2411.17207) and analyzing the efficiency of NLP inspired tabular models.

<h3>⚡ What's New ⚡</h3>
<ul>
  <li>New Models: `Tangos`, `AutoInt`, `Trompt`, `ModernNCA`</li>
  <li>Pretraining optionality for suitable models.</li>
  <li>Individual preprocessing: preprocess each feature differently, use pre-trained models for categorical encoding</li>
  <li>Extract latent representations of tables</li>
  <li>Use embeddings as inputs</li>
  <li>Define custom training metrics</li>
</ul>

<h3> Table of Contents </h3>

- [🏃 Quickstart](#-quickstart)
- [📖 Introduction](#-introduction)
- [🤖 Models](#-models)
- [📚 Documentation](#-documentation)
- [🛠️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [💻 Implement Your Own Model](#-implement-your-own-model)
- [🏷️ Citation](#️-citation)
- [License](#license)

# 🏃 Quickstart

Similar to any sklearn model, deeptab models can be fit as easy as this:

```python
from deeptab.models import MambularClassifier
# Initialize and fit your model
model = MambularClassifier()

# X can be a dataframe or something that can be easily transformed into a pd.DataFrame as a np.array
model.fit(X, y, max_epochs=150, lr=1e-04)
```

# 📖 Introduction

deeptab is a Python package that brings the power of advanced deep learning architectures to tabular data, offering a suite of models for regression, classification, and distributional regression tasks. Designed with ease of use in mind, deeptab models adhere to scikit-learn's `BaseEstimator` interface, making them highly compatible with the familiar scikit-learn ecosystem. This means you can fit, predict, and evaluate using deeptab models just as you would with any traditional scikit-learn model, but with the added performance and flexibility of deep learning.

# 🤖 Models

| Model            | Description                                                                                                                                                                                                 |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Mambular`       | A sequential model using Mamba blocks specifically designed for various tabular data tasks introduced [here](https://arxiv.org/abs/2408.06291).                                                             |
| `TabM`           | Batch Ensembling for a MLP as introduced by [Gorishniy et al.](https://arxiv.org/abs/2410.24210)                                                                                                            |
| `NODE`           | Neural Oblivious Decision Ensembles as introduced by [Popov et al.](https://arxiv.org/abs/1909.06312)                                                                                                       |
| `FTTransformer`  | A model leveraging transformer encoders, as introduced by [Gorishniy et al.](https://arxiv.org/abs/2106.11959), for tabular data.                                                                           |
| `MLP`            | A classical Multi-Layer Perceptron (MLP) model for handling tabular data tasks.                                                                                                                             |
| `ResNet`         | An adaptation of the ResNet architecture for tabular data applications.                                                                                                                                     |
| `TabTransformer` | A transformer-based model for tabular data introduced by [Huang et al.](https://arxiv.org/abs/2012.06678), enhancing feature learning capabilities.                                                         |
| `MambaTab`       | A tabular model using a Mamba-Block on a joint input representation described [here](https://arxiv.org/abs/2401.08867) . Not a sequential model.                                                            |
| `TabulaRNN`      | A Recurrent Neural Network for Tabular data, introduced [here](https://arxiv.org/pdf/2411.17207).                                                                                                           |
| `MambAttention`  | A combination between Mamba and Transformers, also introduced [here](https://arxiv.org/pdf/2411.17207).                                                                                                     |
| `NDTF`           | A neural decision forest using soft decision trees. See [Kontschieder et al.](https://openaccess.thecvf.com/content_iccv_2015/html/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.html) for inspiration. |
| `SAINT`          | Improve neural networs via Row Attention and Contrastive Pre-Training, introduced [here](https://arxiv.org/pdf/2106.01342).                                                                                 |
| `AutoInt`        | Automatic Feature Interaction Learning via Self-Attentive Neural Networks introduced [here](https://arxiv.org/abs/1810.11921).                                                                              |
| `Trompt`         | Trompt: Towards a Better Deep Neural Network for Tabular Data introduced [here](https://arxiv.org/abs/2305.18446).                                                                                          |
| `Tangos`         | Tangos: Regularizing Tabular Neural Networks through Gradient Orthogonalization and Specialization introduced [here](https://openreview.net/pdf?id=n6H86gW8u0d).                                            |
| `ModernNCA`      | Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later introduced [here](https://arxiv.org/abs/2407.03257).                                                                |
| `TabR`           | TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023 [here](https://arxiv.org/abs/2307.14338)                                                                                                        |

All models are available for `regression`, `classification` and distributional regression, denoted by `LSS`.
Hence, they are available as e.g. `MambularRegressor`, `MambularClassifier` or `MambularLSS`

# 📚 Documentation

You can find the deeptab API documentation [here](https://deeptab.readthedocs.io/en/latest/).

# 🛠️ Installation

Install deeptab using pip:

```sh
pip install deeptab
```

If you want to use the original mamba and mamba2 implementations, additionally install mamba-ssm via:

```sh
pip install mamba-ssm
```

Be careful to use the correct torch and cuda versions:

```sh
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
pip install mamba-ssm
```

# 🚀 Usage

<h2> Preprocessing </h2>

deeptab uses pretab preprocessing: https://github.com/OpenTabular/PreTab

Hence, datatypes etc. are detected automatically and all preprocessing methods from pretab as well as from Sklearn.preprocessing are available.
Additionally, you can specify that each feature is preprocessed differently, according to your requirements, by setting the `feature_preprocessing={}`argument during model initialization.
For an overview over all available methods: [pretab](https://github.com/OpenTabular/PreTab)

<h3> Data Type Detection and Transformation </h3>

- **Ordinal & One-Hot Encoding**: Automatically transforms categorical data into numerical formats using continuous ordinal encoding or one-hot encoding. Includes options for transforming outputs to `float` for compatibility with downstream models.
- **Binning**: Discretizes numerical features into bins, with support for both fixed binning strategies and optimal binning derived from decision tree models.
- **MinMax**: Scales numerical data to a specific range, such as [-1, 1], using Min-Max scaling or similar techniques.
- **Standardization**: Centers and scales numerical features to have a mean of zero and unit variance for better compatibility with certain models.
- **Quantile Transformations**: Normalizes numerical data to follow a uniform or normal distribution, handling distributional shifts effectively.
- **Spline Transformations**: Captures nonlinearity in numerical features using spline-based transformations, ideal for complex relationships.
- **Piecewise Linear Encodings (PLE)**: Captures complex numerical patterns by applying piecewise linear encoding, suitable for data with periodic or nonlinear structures.
- **Polynomial Features**: Automatically generates polynomial and interaction terms for numerical features, enhancing the ability to capture higher-order relationships.
- **Box-Cox & Yeo-Johnson Transformations**: Performs power transformations to stabilize variance and normalize distributions.
- **Custom Binning**: Enables user-defined bin edges for precise discretization of numerical data.
- **Pre-trained Encoding**: Use sentence transformers to encode categorical features.

<h2> Fit a Model </h2>
Fitting a model in deeptab is as simple as it gets. All models in deeptab are sklearn BaseEstimators. Thus the `.fit` method is implemented for all of them. Additionally, this allows for using all other sklearn inherent methods such as their built in hyperparameter optimization tools.

```python
from deeptab.models import MambularClassifier
# Initialize and fit your model
model = MambularClassifier(
    d_model=64,
    n_layers=4,
    numerical_preprocessing="ple",
    n_bins=50,
    d_conv=8
)

# X can be a dataframe or something that can be easily transformed into a pd.DataFrame as a np.array
model.fit(X, y, max_epochs=150, lr=1e-04)
```

Predictions are also easily obtained:

```python
# simple predictions
preds = model.predict(X)

# Predict probabilities
preds = model.predict_proba(X)
```

Get latent representations for each feature:

```python
# simple encoding
model.encode(X)
```

Use unstructured data:

```python
# load pretrained models
image_model = ...
nlp_model = ...

# create embeddings
img_embs = image_model.encode(images)
txt_embs = nlp_model.encode(texts)

# fit model on tabular data and unstructured data
model.fit(X_train, y_train, embeddings=[img_embs, txt_embs])
```

<h3> Hyperparameter Optimization</h3>
Since all of the models are sklearn base estimators, you can use the built-in hyperparameter optimizatino from sklearn.

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'd_model': randint(32, 128),
    'n_layers': randint(2, 10),
    'lr': uniform(1e-5, 1e-3)
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,  # Number of parameter settings sampled
    cv=5,       # 5-fold cross-validation
    scoring='accuracy',  # Metric to optimize
    random_state=42
)

fit_params = {"max_epochs":5, "rebuild":False}

# Fit the model
random_search.fit(X, y, **fit_params)

# Best parameters and score
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)
```

Note, that using this, you can also optimize the preprocessing. Just specify the necessary parameters when specifying the preprocessor arguments you want to optimize:

```python
param_dist = {
    'd_model': randint(32, 128),
    'n_layers': randint(2, 10),
    'lr': uniform(1e-5, 1e-3),
    "numerical_preprocessing": ["ple", "standardization", "box-cox"]
}

```

Since we have early stopping integrated and return the best model with respect to the validation loss, setting max_epochs to a large number is sensible.

Or use the built-in bayesian hpo simply by running:

```python
best_params = model.optimize_hparams(X, y)
```

This automatically sets the search space based on the default config from `deeptab.configs`. See the documentation for all params with regard to `optimize_hparams()`. However, the preprocessor arguments are fixed and cannot be optimized here.

<h2> ⚖️ Distributional Regression with MambularLSS </h2>

MambularLSS allows you to model the full distribution of a response variable, not just its mean. This is crucial when understanding variability, skewness, or kurtosis is important. All deeptab models are available as distributional models.

<h3> Key Features of MambularLSS: </h3>

- **Full Distribution Modeling**: Predicts the entire distribution, not just a single value, providing richer insights.
- **Customizable Distribution Types**: Supports various distributions (e.g., Gaussian, Poisson, Binomial) for different data types.
- **Location, Scale, Shape Parameters**: Predicts key distributional parameters for deeper insights.
- **Enhanced Predictive Uncertainty**: Offers more robust predictions by modeling the entire distribution.

<h3> Available Distribution Classes: </h3>

- **normal**: For continuous data with a symmetric distribution.
- **poisson**: For count data within a fixed interval.
- **gamma**: For skewed continuous data, often used for waiting times.
- **beta**: For data bounded between 0 and 1, like proportions.
- **dirichlet**: For multivariate data with correlated components.
- **studentt**: For data with heavier tails, useful with small samples.
- **negativebinom**: For over-dispersed count data.
- **inversegamma**: Often used as a prior in Bayesian inference.
- **johnsonsu**: Four parameter distribution defining location, scale, kurtosis and skewness.
- **categorical**: For data with more than two categories.
- **Quantile**: For quantile regression using the pinball loss.

These distribution classes make MambularLSS versatile in modeling various data types and distributions.

<h3> Getting Started with MambularLSS: </h3>

To integrate distributional regression into your workflow with `MambularLSS`, start by initializing the model with your desired configuration, similar to other deeptab models:

```python
from deeptab.models import MambularLSS

# Initialize the MambularLSS model
model = MambularLSS(
    dropout=0.2,
    d_model=64,
    n_layers=8,

)

# Fit the model to your data
model.fit(
    X,
    y,
    max_epochs=150,
    lr=1e-04,
    patience=10,
    family="normal" # define your distribution
    )

```

# 💻 Implement Your Own Model

deeptab allows users to easily integrate their custom models into the existing logic. This process is designed to be straightforward, making it simple to create a PyTorch model and define its forward pass. Instead of inheriting from `nn.Module`, you inherit from deeptab's `BaseModel`. Each deeptab model takes three main arguments: the number of classes (e.g., 1 for regression or 2 for binary classification), `cat_feature_info`, and `num_feature_info` for categorical and numerical feature information, respectively. Additionally, you can provide a config argument, which can either be a custom configuration or one of the provided default configs.

One of the key advantages of using deeptab is that the inputs to the forward passes are lists of tensors. While this might be unconventional, it is highly beneficial for models that treat different data types differently. For example, the TabTransformer model leverages this feature to handle categorical and numerical data separately, applying different transformations and processing steps to each type of data.

Here's how you can implement a custom model with deeptab:

1. **First, define your config:**  
   The configuration class allows you to specify hyperparameters and other settings for your model. This can be done using a simple dataclass.

   ```python
   from dataclasses import dataclass
   from deeptab.configs import BaseConfig

   @dataclass
   class MyConfig(BaseConfig):
       lr: float = 1e-04
       lr_patience: int = 10
       weight_decay: float = 1e-06
       n_layers: int = 4
       pooling_method:str = "avg

   ```

2. **Second, define your model:**  
   Define your custom model just as you would for an `nn.Module`. The main difference is that you will inherit from `BaseModel` and use the provided feature information to construct your layers. To integrate your model into the existing API, you only need to define the architecture and the forward pass.

   ```python
   from deeptab.base_models.utils import BaseModel
   from deeptab.utils.get_feature_dimensions import get_feature_dimensions
   import torch
   import torch.nn

   class MyCustomModel(BaseModel):
       def __init__(
           self,
           feature_information: tuple,
           num_classes: int = 1,
           config=None,
           **kwargs,
       ):
            super().__init__(**kwargs)
            self.save_hyperparameters(ignore=["feature_information"])
            self.returns_ensemble = False

            # embedding layer
            self.embedding_layer = EmbeddingLayer(
                *feature_information,
                config=config,
            )

           input_dim = np.sum(
                [len(info) * self.hparams.d_model for info in feature_information]
            )

           self.linear = nn.Linear(input_dim, num_classes)

       def forward(self, *data) -> torch.Tensor:
            x = self.embedding_layer(*data)
            B, S, D = x.shape
            x = x.reshape(B, S * D)


           # Pass through linear layer
           output = self.linear(x)
           return output
   ```

3. **Leverage the deeptab API:**  
   You can build a regression, classification, or distributional regression model that can leverage all of deeptab's built-in methods by using the following:

   ```python
   from deeptab.models.utils import SklearnBaseRegressor

   class MyRegressor(SklearnBaseRegressor):
       def __init__(self, **kwargs):
           super().__init__(model=MyCustomModel, config=MyConfig, **kwargs)
   ```

4. **Train and evaluate your model:**  
   You can now fit, evaluate, and predict with your custom model just like with any other deeptab model. For classification or distributional regression, inherit from `SklearnBaseClassifier` or `SklearnBaseLSS` respectively.

   ```python
   regressor = MyRegressor(numerical_preprocessing="ple")
   regressor.fit(X_train, y_train, max_epochs=50)

   regressor.evaluate(X_test, y_test)
   ```

# 🤝 Contributing

We welcome contributions! This project uses [Conventional Commits](https://www.conventionalcommits.org/) and automated semantic versioning.

**Quick Start for Contributors:**

```bash
# Install dependencies with pre-commit hooks
just install

# Make your changes and commit using the interactive tool
just commit

# Or commit manually following conventional commits format
git commit -m "feat(models): add new model architecture"
```

See our [Contributing Guide](docs/contributing.md) for detailed guidelines and [Conventional Commits Reference](CONVENTIONAL_COMMITS.md) for commit message formatting.

# 🏷️ Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@article{thielmann2024mambular,
  title={Mambular: A Sequential Model for Tabular Deep Learning},
  author={Thielmann, Anton Frederik and Kumar, Manish and Weisser, Christoph and Reuter, Arik and S{\"a}fken, Benjamin and Samiee, Soheila},
  journal={arXiv preprint arXiv:2408.06291},
  year={2024}
}
```

If you use TabulaRNN please consider to cite:

```BibTeX
@article{thielmann2024efficiency,
  title={On the Efficiency of NLP-Inspired Methods for Tabular Deep Learning},
  author={Thielmann, Anton Frederik and Samiee, Soheila},
  journal={arXiv preprint arXiv:2411.17207},
  year={2024}
}
```

# License

The entire codebase is under MIT license.

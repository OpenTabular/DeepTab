# DeepTab: Tabular Deep Learning Made Simple

DeepTab is a Python library for deep learning on tabular data, built on PyTorch and Lightning with a scikit-learn compatible API. It provides 15 neural architectures, including State Space Models (Mamba), Transformers, attention, retrieval, and tree-inspired networks, each available as a classifier, a regressor, and a distributional (`LSS`) model for uncertainty estimation. Preprocessing, training, and evaluation run through the standard `fit`/`predict`/`evaluate` workflow, so one code path serves day-to-day modeling, architecture research (via runtime registries and custom-model base classes), and production deployment (via schema-validated, read-only inference artifacts).

```python
from deeptab.models import MambularClassifier

model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## Why DeepTab

- **Familiar interface.** A scikit-learn `fit`/`predict`/`evaluate` API that drops into existing pipelines, including `GridSearchCV`.
- **Automatic preprocessing.** Feature-type detection, encoding, scaling, and missing-value handling are built in.
- **One model, three tasks.** Every architecture ships as a classifier, a regressor, and a distributional (`LSS`) variant for uncertainty quantification.
- **A broad model zoo.** 15 stable architectures plus experimental models, all behind the same interface, with [selection guidance](model_zoo/comparison_tables).
- **Built for real data.** Mixed feature types, class imbalance, GPU acceleration, and early stopping work out of the box.

## Installation

```bash
pip install deeptab
```

DeepTab requires Python 3.10+ and installs PyTorch automatically. See [Installation](getting_started/installation) for GPU setup and the optional Mamba CUDA kernels.

## What's New in v2.0

v2.0 is a ground-up restructuring of DeepTab. The high-level estimator API stays familiar, while the package layout, configuration objects, and import paths have moved.

- **Split-config API**: separate model, preprocessing, and training configuration objects, so each concern can be tuned on its own. This is the first thing you reach for in v2.
- **New models**: AutoInt, ENODE, and TabR (stable); Tangos, Trompt, and ModernNCA (experimental).
- **Observability**: `ObservabilityConfig` adds structured lifecycle logging and one-line MLflow or TensorBoard tracking, opt-in and silent by default.
- **Deployment-safe inference**: `InferenceModel` exposes a read-only prediction surface with schema validation, so a served model cannot be re-fitted by accident.
- **Self-describing artifacts**: a single `.deeptab` save format bundles the architecture, feature schema, preprocessing, and versions with the weights.
- **Registry-driven training**: optimizers, schedulers, and losses are selectable by name through `TrainerConfig` and extensible at runtime.
- **Unified metrics**: 25+ metric classes auto-selected per task across regression, classification, and distributional models.
- **Typed data layer**: `TabularDataset`, `TabularDataModule`, and `FeatureSchema` give the pipeline an inspectable contract.
- **Reproducibility**: cross-platform seeding across CPU, CUDA, and MPS.
- **Rebuilt docs and tutorials**: refreshed guides plus end-to-end, Colab-ready tutorials for [classification](tutorials/imbalance_classification), [regression](tutorials/skewed_regression), and [uncertainty quantification](tutorials/uncertainty_quantification).

```{warning}
Upgrading from v1 requires changes. Packages were reorganised, the `Default<Arch>Config` classes were renamed to `<Arch>Config`, and the data modules became `TabularDataModule` / `TabularDataset`. See the [FAQ](getting_started/faq) for v1 support and upgrade notes.
```

See the [Overview](getting_started/overview) for the full picture.

## Available Models

DeepTab provides 15 stable architectures across five families: State Space Models (Mambular, MambaTab, MambAttention), Transformers (FTTransformer, TabTransformer, SAINT, AutoInt), residual networks (ResNet, TabR), tree-inspired models (NODE, ENODE, NDTF), and general baselines (MLP, TabM, TabulaRNN). Three experimental models (ModernNCA, Tangos, Trompt) are under evaluation for promotion.

Each architecture comes in three variants, `*Classifier`, `*Regressor`, and `*LSS`, sharing one interface so you can swap models without changing code. See the [Model Zoo](model_zoo/comparison_tables) for comparisons and selection guidance.

---

## Documentation

### Getting Started

- [Overview](getting_started/overview): What DeepTab is and when to use it
- [Installation](getting_started/installation): Setup, GPU support, and optional kernels
- [Quickstart](getting_started/quickstart): Train your first models in a few minutes
- [FAQ](getting_started/faq): Common questions and troubleshooting

### Core Concepts

- [sklearn API](core_concepts/sklearn_api): The fit/predict/evaluate interface
- [Model Tiers](core_concepts/model_tiers): Stable versus experimental models
- [Config System](core_concepts/config_system): Split configuration for model, preprocessing, and training
- [Training and Evaluation](core_concepts/training_and_evaluation): The fit pipeline, metrics, and reproducibility
- [Observability](core_concepts/observability): Lifecycle events, structured logging, and experiment tracking
- [Model Operations](core_concepts/model_operations): Serialisation and inspection
- [Inference](core_concepts/inference): Deployment-safe prediction with `InferenceModel`

### Tutorials

- [Imbalanced Classification](tutorials/imbalance_classification): An end-to-end classification workflow
- [Skewed-Target Regression](tutorials/skewed_regression): Regression on a right-skewed target
- [Uncertainty Quantification](tutorials/uncertainty_quantification): Prediction intervals with LSS models
- [Hyperparameter Optimisation](tutorials/hpo): Tuning models efficiently
- [Advanced Training and Inference](tutorials/advanced_training): Optimizers, schedulers, and production inference
- [Observability and Logging](tutorials/observability): Run directories and experiment trackers
- [Model Efficiency](tutorials/model_efficiency): Runtime and memory benchmarking
- [Experimental Models](tutorials/experimental): Working with cutting-edge architectures

### Model Zoo

- [Comparison Tables](model_zoo/comparison_tables): Selection guidance and performance across dimensions
- [Stable Models](model_zoo/stable/index): Production-ready architectures
- [Experimental Models](model_zoo/experimental/index): Models under evaluation
- [Efficiency and Benchmarking](model_zoo/efficiency): Runtime and memory guidance
- [Recommended Configs](model_zoo/recommended_configs): Hyperparameter recipes

### API Reference

- [Models](api/models/index): Classifier, Regressor, and LSS classes
- [Configs](api/configs/index): Configuration dataclasses
- [Data](api/data/index): Datasets, data modules, and schemas
- [Distributions](api/distributions/index): LSS distribution families
- [Metrics](api/metrics/index): Task-aware metric classes
- [Training](api/training/index): Lightning modules for advanced use

### Developer Guide

- [Contributing](developer_guide/contributing): How to contribute
- [Testing](developer_guide/testing): Test suite and coverage
- [Documentation](developer_guide/documentation): Building the docs locally
- [Versioning](developer_guide/versioning): Semantic versioning policy
- [CI/CD](developer_guide/ci_cd): Continuous integration
- [Release Process](developer_guide/release): Release workflow
- [Model Promotion Policy](developer_guide/model_promotion_policy): From experimental to stable
- [Support Matrix](developer_guide/support_matrix): Supported Python and PyTorch versions

---

## Citation

If you use DeepTab in your research, please cite:

```bibtex
@article{thielmann2024mambular,
  title={Mambular: A Sequential Model for Tabular Deep Learning},
  author={Thielmann, Anton Frederik and Kumar, Manish and Weisser, Christoph and Reuter, Arik and S{\"a}fken, Benjamin and Samiee, Soheila},
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

## License

DeepTab is licensed under the MIT License. See [LICENSE](https://github.com/OpenTabular/DeepTab/blob/main/LICENSE) for details.

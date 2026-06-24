# DeepTab: Tabular Deep Learning Made Simple

**DeepTab** is a Python library for deep learning on tabular data, built on PyTorch and Lightning with a scikit-learn compatible API. It offers 15 neural architectures, from Mamba-inspired state space models and Transformers to tree ensembles and MLP baselines, each available as a classifier, regressor, or distributional (`LSS`) model. One `fit`/`predict`/`evaluate` workflow covers everyday modeling, architecture research, and production deployment.

```python
from deeptab.models import MambularClassifier

model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## Why DeepTab

- **Familiar interface.** A scikit-learn `fit`/`predict`/`evaluate` API that drops into existing pipelines, including `GridSearchCV`.
- **Automatic preprocessing.** Feature-type detection, encoding, scaling, and missing-value handling are powered by [PreTab](https://github.com/OpenTabular/PreTab) and applied for you.
- **One model, three tasks.** Every architecture ships as a classifier, a regressor, and a distributional (`LSS`) variant for uncertainty quantification.
- **A broad model zoo.** 15 stable architectures plus experimental models, all behind the same interface, with [selection guidance](model_zoo/comparison_tables).
- **Built for real data.** Mixed feature types, class imbalance, GPU acceleration, and early stopping work out of the box.

## Installation

```bash
pip install deeptab
```

DeepTab requires Python 3.10+ and installs PyTorch automatically. See [Installation](getting_started/installation) for GPU setup and the optional Mamba CUDA kernels.

## What's New in v2.0

v2.0 is a ground-up restructuring of DeepTab. The high-level estimator API stays familiar, while the package layout, configuration objects, and import paths have been updated.

- **Split-config API**: separate model, preprocessing, and training configuration objects, so each concern can be tuned on its own. This is the first thing you reach for in v2.
- **New models**: AutoInt, ENODE, and TabR (stable); Tangos, Trompt, and ModernNCA (experimental).
- **Observability**: `ObservabilityConfig` adds structured lifecycle logging via `structlog` and one-line MLflow or TensorBoard tracking, opt-in and silent by default.
- **Deployment-safe inference**: `InferenceModel` exposes a read-only prediction surface with schema validation, so a served model cannot be re-fitted by accident.
- **Self-describing artifacts**: a single `.deeptab` save format bundles the architecture, feature schema, preprocessing, and versions with the weights.
- **Registry-driven training**: optimizers, schedulers, and losses are selectable by name through `TrainerConfig` and extensible at runtime.
- **Unified metrics**: 25+ metric classes auto-selected per task across regression, classification, and distributional models.
- **Typed data layer**: `TabularDataset`, `TabularDataModule`, and `FeatureSchema` give the pipeline an inspectable contract.
- **Reproducibility**: cross-platform seeding across CPU, CUDA, and MPS.
- **Rebuilt docs and tutorials**: refreshed guides plus end-to-end, Colab-ready tutorials for [classification](tutorials/imbalance_classification), [regression](tutorials/skewed_regression), and [uncertainty quantification](tutorials/uncertainty_quantification).

```{warning}
**v2.0 is not backward compatible with v1, and v1 is no longer maintained.** Three
things changed that affect existing code:

1. **Import paths** were reorganised under the `deeptab` namespace.
2. **Config classes** lost their `Default` prefix, so `DefaultMambularConfig` is now
   `MambularConfig`. Settings are also split across `MambularConfig` (architecture),
   `PreprocessingConfig` (feature handling), and `TrainerConfig` (optimisation).
3. **Data modules** were renamed to `TabularDataModule` and `TabularDataset`; the old
   `Mambular*` aliases are deprecated.

If you need to stay on v1 for now, pin `deeptab<2.0`. Note that the v1 branch receives
no bug fixes or security updates. See the [migration guide](getting_started/migration)
for a step-by-step upgrade walkthrough, and the [FAQ](getting_started/faq) for the
full support policy.
```

The high-level `fit`/`predict`/`evaluate` workflow is unchanged. In most cases only
the imports and config construction need updating:

```python
# v1: settings passed as flat keyword arguments on the estimator
from deeptab.models import MambularClassifier

model = MambularClassifier(d_model=128, n_layers=4, numerical_preprocessing="ple")
model.fit(X_train, y_train, max_epochs=50)
```

```python
# v2: settings grouped into focused config objects
from deeptab.models import MambularClassifier
from deeptab.configs import MambularConfig, PreprocessingConfig

model = MambularClassifier(
    model_config=MambularConfig(d_model=128, n_layers=4),          # architecture
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="ple"),  # features
)
model.fit(X_train, y_train, max_epochs=50)
```

```{note}
You only pass the configs you want to customise. `MambularClassifier()` with no
arguments uses sensible defaults for the architecture, preprocessing, and training.
The flat keyword-argument style from v1 is no longer accepted, so settings must go
through the relevant config object.
```

See the [Overview](getting_started/overview) for the full picture.

## Available Models

DeepTab provides 15 stable architectures across five families: State Space Models (Mambular, MambaTab, MambAttention), Transformers (FTTransformer, TabTransformer, SAINT, AutoInt), residual networks (ResNet, TabR), tree-inspired models (NODE, ENODE, NDTF), and general baselines (MLP, TabM, TabulaRNN). Three experimental models (ModernNCA, Tangos, Trompt) are under evaluation for promotion.

Each architecture comes in three variants, `*Classifier`, `*Regressor`, and `*LSS`, sharing one interface so you can swap models without changing code. See the [Model Zoo](model_zoo/comparison_tables) for comparisons and selection guidance.

---

## Documentation

```{note}
These are starting points for each area. The sidebar navigation is the source of
truth, so please review the individual documentation sections for the latest
updates and further documentation.
```

- **Getting Started**: Begin with the [Overview](getting_started/overview) and [Quickstart](getting_started/quickstart), then see [Installation](getting_started/installation) for GPU setup.
- **Core Concepts**: How DeepTab works, including the [sklearn API](core_concepts/sklearn_api), the [Config System](core_concepts/config_system), and [Training and Evaluation](core_concepts/training_and_evaluation).
- **Tutorials**: End-to-end, Colab-ready walkthroughs for [regression](tutorials/skewed_regression), [classification](tutorials/imbalance_classification), and [uncertainty quantification](tutorials/uncertainty_quantification).
- **Model Zoo**: Browse the [Stable Models](model_zoo/stable/index) and [Experimental Models](model_zoo/experimental/index), or use the [Comparison Tables](model_zoo/comparison_tables) for selection guidance.
- **API Reference**: Full reference for [Models](api/models/index), [Configs](api/configs/index), and the rest of the public API.
- **Developer Guide**: [Contributing](developer_guide/contributing), [Testing](developer_guide/testing), and the [Release Process](developer_guide/release) for maintainers.

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

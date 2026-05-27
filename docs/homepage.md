```{include} ../README.md
:start-after: <!-- START SHARED CONTENT -->
:end-before: <!-- END SHARED CONTENT -->
```

---

## 📚 Documentation Navigation

### 🚀 Getting Started

New to DeepTab? Start here:

- **[Overview](getting_started/overview)** — What is DeepTab?
- **[Why DeepTab?](getting_started/why_deeptab)** — Key features and advantages
- **[Installation](getting_started/installation)** — Setup and dependencies
- **[Quickstart](getting_started/quickstart)** — Your first model in 5 minutes
- **[FAQ](getting_started/faq)** — Common questions answered

### 📖 Core Concepts

Understand DeepTab's design:

- **[sklearn API](core_concepts/sklearn_api)** — Familiar fit/predict/evaluate interface
- **[Model Tiers](core_concepts/model_tiers)** — Stable vs experimental models
- **[Config System](core_concepts/config_system)** — Split-config for model, preprocessing, training
- **[Preprocessing](core_concepts/preprocessing)** — Automatic feature handling
- **[Classification](core_concepts/classification)** — Binary and multi-class classification
- **[Regression](core_concepts/regression)** — Point estimation regression
- **[Distributional Regression](core_concepts/distributional_regression)** — Full distribution prediction (LSS)
- **[Training & Evaluation](core_concepts/training_and_evaluation)** — Deep dive into training

### 🎯 Interactive Tutorials

Hands-on examples with Google Colab:

- **[Classification Tutorial](tutorials/classification)** — Multi-class classification workflow
- **[Regression Tutorial](tutorials/regression)** — Standard regression with TabR
- **[Distributional Regression (LSS)](tutorials/distributional)** — Full distribution prediction
- **[Experimental Models](tutorials/experimental)** — Using cutting-edge architectures
- **[Model Efficiency Benchmarking](tutorials/model_efficiency)** — Runtime and memory workflow

### 🤖 Model Zoo

Choose the right model for your task:

- **[Model Selection Guide](model_zoo/comparison_tables)** — Quick start and decision tree
- **[Comparison Tables](model_zoo/comparison_tables)** — Performance across dimensions
- **[Efficiency & Benchmarking](model_zoo/efficiency)** — Runtime and memory benchmarking guidance
- **[Recommended Configs](model_zoo/recommended_configs)** — Hyperparameter recipes

**Browse by category:**

- [State Space Models](model_zoo/stable/index) — Mambular, MambaTab, MambAttention
- [Transformer-Based](model_zoo/stable/index) — FTTransformer, TabTransformer, SAINT
- [MLP-Based](model_zoo/stable/index) — ResNet, MLP, TabM, AutoInt
- [Tree-Based](model_zoo/stable/index) — NODE, ENODE, NDTF
- [Specialized](model_zoo/stable/index) — TabR, TabulaRNN
- [Experimental](model_zoo/experimental/index) — ModernNCA, Tangos, Trompt

### 📖 API Reference

Complete API documentation:

- **[Models API](api/models/index)** — All model classes (Classifier, Regressor, LSS)
- **[Configs API](api/configs/index)** — Configuration dataclasses
- **[Data API](api/data/index)** — TabularDataset, TabularDataModule, schemas
- **[Distributions API](api/distributions/index)** — LSS distribution families
- **[Training API](api/training/index)** — Lightning modules for advanced use

### 🛠️ Developer Guide

Contributing to DeepTab:

- **[Contributing Guidelines](developer_guide/contributing)** — How to contribute
- **[Testing](developer_guide/testing)** — Test suite and coverage
- **[Documentation](developer_guide/documentation)** — Building docs locally
- **[Release Process](developer_guide/release)** — Release workflow
- **[Versioning](developer_guide/versioning)** — Semantic versioning policy
- **[CI/CD](developer_guide/ci_cd)** — Continuous integration
- **[Model Promotion Policy](developer_guide/model_promotion_policy)** — Experimental to stable
- **[Support Matrix](developer_guide/support_matrix)** — Python/PyTorch versions

---

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

DeepTab is licensed under the MIT License. See [LICENSE](https://github.com/basf/deeptab/blob/main/LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please see [Contributing Guide](developer_guide/contributing) for details.

## 📞 Support

- **Documentation:** [deeptab.readthedocs.io](https://deeptab.readthedocs.io)
- **GitHub Issues:** [Report bugs or request features](https://github.com/basf/deeptab/issues)
- **GitHub Discussions:** [Ask questions and share ideas](https://github.com/basf/deeptab/discussions)
- **Papers:**
  - [Mambular: A Sequential Model for Tabular Deep Learning](https://arxiv.org/abs/2408.06291)
  - [TabulaRNN: Analyzing Efficiency of RNN Models](https://arxiv.org/pdf/2411.17207)

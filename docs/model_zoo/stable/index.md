# Stable Models

```{important}
**Production-Ready Architectures**

All stable models have frozen APIs covered by semantic versioning. Safe for production use with guaranteed backward compatibility.
```

DeepTab provides **15 battle-tested deep learning architectures** for tabular data, each optimized for different use cases. All models support:

- **Classification** (binary and multiclass)
- **Regression** (continuous targets)
- **Distributional Regression** (uncertainty quantification)

## Model Categories

### 🧬 State Space Models (SSMs)

Modern sequence models that efficiently capture feature dependencies:

- **[Mambular](mambular)** — Sequential processing with Mamba blocks
- **[MambaTab](mambatab)** — Joint processing variant
- **[MambAttention](mambattention)** — Hybrid Mamba-Attention architecture

### 🤖 Transformer Architectures

Attention-based models excelling at complex feature interactions:

- **[FTTransformer](fttransformer)** — Feature Tokenizer + Transformer
- **[TabTransformer](tabtransformer)** — Categorical feature embeddings
- **[SAINT](saint)** — Self-Attention + Intersample Attention

### 🏗️ Residual Networks

Deep feedforward architectures with skip connections:

- **[ResNet](resnet)** — Classic residual architecture for tabular data
- **[MLP](mlp)** — Multi-layer perceptron baseline

### 🌲 Tree-Based Neural Models

Neural networks that mimic decision tree behavior:

- **[NODE](node)** — Neural Oblivious Decision Ensembles
- **[ENODE](enode)** — Enhanced NODE with feature selection
- **[NDTF](ndtf)** — Neural Decision Tree Forest

### 📊 Other Architectures

Specialized designs for specific use cases:

- **[TabM](tabm)** — Efficient architecture for large-scale data
- **[TabR](tabr)** — Retrieval-augmented predictions
- **[AutoInt](autoint)** — Automatic feature interaction learning
- **[TabulaRNN](tabularnn)** — Recurrent architecture for sequential features

## Quick Start

```python
from deeptab.models import MambularClassifier

# Import any stable model
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)
predictions = model.predict(X_test)
```

## Choosing a Model

```{tip}
Start with **Mambular** for most tasks. It's our most robust general-purpose model.
```

Not sure which model to use? See:

- **[Comparison Tables](../comparison_tables)** — Performance and complexity analysis
- **[Recommended Configs](../recommended_configs)** — Dataset-specific guidance

## Complete Model List

```{toctree}
:maxdepth: 1

mambular
mambatab
mambattention
fttransformer
tabtransformer
saint
mlp
resnet
tabm
node
enode
autoint
ndtf
tabr
tabularnn
```

# Hyperparameter Configuration Guidelines

This guide gives research-oriented and developer-oriented starting points for DeepTab hyperparameter tuning. The goal is not to prescribe universal optima. Tabular datasets vary strongly in sample size, feature cardinality, signal-to-noise ratio, missingness, and feature interactions, so the right configuration should be selected with a validation protocol.

```{note}
**Use this as a protocol, not a leaderboard.** Start with a defensible baseline, tune the smallest set of high-impact parameters, and report the search budget together with results. Deep tabular models are sensitive to preprocessing, optimization, and evaluation design.
```

## Configuration Layers

DeepTab separates model structure, preprocessing, and training into independent config objects.

| Config | Controls | Examples |
| ------ | -------- | -------- |
| `<Model>Config` | Architecture | `d_model`, `n_layers`, `dropout`, `layer_sizes`, `depth` |
| `PreprocessingConfig` | Feature transforms | `numerical_preprocessing`, `categorical_preprocessing`, `n_bins` |
| `TrainerConfig` | Optimization/runtime | `lr`, `batch_size`, `max_epochs`, `patience`, `weight_decay` |

```python
from deeptab.configs import MambularConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MambularRegressor

model = MambularRegressor(
    model_config=MambularConfig(d_model=128, n_layers=6, dropout=0.1),
    preprocessing_config=PreprocessingConfig(numerical_preprocessing="quantile"),
    trainer_config=TrainerConfig(lr=5e-4, batch_size=256, max_epochs=150),
    random_state=101,
)
```

```{important}
Examples in this page use the current split-config API. Architecture parameters belong in `<Model>Config`; training parameters belong in `TrainerConfig`; preprocessing parameters belong in `PreprocessingConfig`.
```

## Experimental Protocol

For research comparisons, keep the protocol as explicit as the model configuration.

| Decision | Recommendation | Why it matters |
| -------- | -------------- | -------------- |
| Data split | Use a fixed train/validation/test split or repeated cross-validation | Avoids test-set tuning and reduces split noise |
| Search budget | Report the number of trials, epochs, and early-stopping rule | Hyperparameter budget can change model rankings |
| Baselines | Include at least MLP/ResNet or TabM, plus a tree baseline when relevant | Tabular deep learning should be compared to strong simple baselines |
| Metrics | Report task metric and validation loss; for LSS also report NLL/calibration | Point accuracy and uncertainty quality can disagree |
| Seeds | Run multiple seeds for final candidates | Many tabular datasets are small enough for seed variance to matter |
| Preprocessing | Tune preprocessing jointly with model family | Numerical embeddings and transforms can dominate architecture effects |

```{tip}
For papers and internal benchmark reports, prefer "best validation model selected from a declared search space" over "single default run". Also report wall-clock time or number of trials when comparing architectures.
```

## High-Impact Knobs

Tune these before searching large architecture grids.

| Priority | Parameter | Typical Search Values | Applies To | Notes |
| -------- | --------- | --------------------- | ---------- | ----- |
| 1 | `trainer_config__lr` | `[1e-4, 3e-4, 1e-3]` | All models | Usually the highest-impact optimizer parameter |
| 2 | `model_config__dropout`, `attn_dropout`, `ff_dropout` | `[0.0, 0.1, 0.2, 0.3]` | Most neural models | Increase for small/noisy data |
| 3 | Width | `d_model=[64,128,256]` or layer sizes | Mamba/attention/MLP-like models | Width affects capacity and quadratic projection costs |
| 4 | Depth | `n_layers=[1,2,4,6,8]`, model-dependent | Sequence and attention models | More depth is not always better on small tables |
| 5 | Preprocessing | `standard`, `quantile`, `ple` | Numerical-heavy data | Often changes results as much as architecture |
| 6 | Batch size | `[64,128,256,512]` | All models | Constrained by memory and row-attention/retrieval behavior |

### Learning Rate

| Family | Starting Range | Practical Notes |
| ------ | -------------- | --------------- |
| MLP, ResNet, TabM | `3e-4` to `1e-3` | Usually robust; lower LR if loss is unstable |
| Mambular, MambaTab, TabulaRNN | `1e-4` to `1e-3` | Use lower LR for wider/deeper variants |
| FTTransformer, TabTransformer, AutoInt, SAINT | `1e-4` to `5e-4` | Attention models often need conservative updates |
| NODE/ENODE/NDTF | `3e-4` to `1e-3` | Tune with depth/layer dimension; soft tree models can be initialization-sensitive |
| TabR | `1e-4` to `5e-4` | Retrieval and candidate encoding make validation cost higher |

DeepTab currently uses `ReduceLROnPlateau` in the training module. Control it with `lr_patience` and `lr_factor`.

```python
trainer_cfg = TrainerConfig(
    lr=3e-4,
    lr_patience=10,
    lr_factor=0.1,
    weight_decay=1e-6,
    patience=20,
)
```

### Regularization

| Dataset Regime | Dropout Starting Point | Weight Decay Starting Point | Notes |
| -------------- | ---------------------- | --------------------------- | ----- |
| `<1K` rows | `0.2` to `0.5` | `1e-5` to `1e-4` | Prefer smaller models and repeated CV |
| `1K-10K` rows | `0.1` to `0.3` | `1e-6` to `1e-4` | Tune dropout and preprocessing first |
| `10K-100K` rows | `0.0` to `0.2` | `1e-6` to `1e-5` | Capacity starts to help if signal is complex |
| `>100K` rows | `0.0` to `0.1` | `1e-7` to `1e-5` | Watch compute bottlenecks more than overfitting |

```{warning}
Do not assume that large neural models automatically improve with more rows. Dataset difficulty, uninformative features, target smoothness, and feature orientation are central in tabular learning.
```

### Batch Size

| Model Family | Starting Batch Size | Constraint |
| ------------ | ------------------- | ---------- |
| MLP, ResNet, MambaTab, Mambular, TabM | `128` to `512` | Increase until GPU utilization is good or validation degrades |
| FTTransformer, AutoInt, TabTransformer | `128` to `256` | Attention memory grows with feature-token count |
| SAINT | `32` to `128` | Row attention is quadratic in batch size |
| TabR | `128` to `256` | Candidate encoding/search can dominate runtime |
| NODE/ENODE/NDTF | `256` to `512` | Larger batches can stabilize tree/path initialization |

## Model Family Recommendations

### Strong Baseline Stack

Start here unless the research question specifically targets a model family.

```python
from deeptab.configs import MLPConfig, ResNetConfig, TabMConfig, TrainerConfig

mlp_cfg = MLPConfig(layer_sizes=[256, 128, 32], dropout=0.1)
resnet_cfg = ResNetConfig(layer_sizes=[256, 128, 32], num_blocks=3, dropout=0.2)
tabm_cfg = TabMConfig(layer_sizes=[256, 256, 128], ensemble_size=32, dropout=0.1)

trainer_cfg = TrainerConfig(lr=1e-3, batch_size=256, max_epochs=100, patience=15)
```

**Research use:** MLP/ResNet/TabM provide useful controls for whether a more complex architecture is actually adding value. Recent TabM results also make parameter-efficient ensembling a strong baseline, not just a fallback.

### Mambular and MambaTab

Use when you want a sequence-style inductive bias over features without quadratic feature attention.

| Regime | MambularConfig | TrainerConfig |
| ------ | -------------- | ------------- |
| Small data | `d_model=64`, `n_layers=2-4`, `dropout=0.1-0.3` | `lr=5e-4`, `batch_size=128` |
| Medium data | `d_model=128`, `n_layers=4-6`, `dropout=0.0-0.2` | `lr=3e-4` to `5e-4`, `batch_size=256` |
| Large data | `d_model=128-256`, `n_layers=6-8`, `dropout=0.0-0.1` | `lr=1e-4` to `3e-4`, `batch_size=512` |

```python
from deeptab.configs import MambaTabConfig, MambularConfig, TrainerConfig

# Lightweight Mamba baseline
mambatab_cfg = MambaTabConfig(
    d_model=64,
    n_layers=1,
    d_conv=16,
    dropout=0.05,
)

# Higher-capacity tabular sequence model
mambular_cfg = MambularConfig(
    d_model=128,
    n_layers=6,
    d_state=128,
    expand_factor=2,
    dropout=0.1,
    pooling_method="avg",
)

trainer_cfg = TrainerConfig(lr=3e-4, batch_size=256, max_epochs=150, patience=20)
```

**Tune first:** `d_model`, `n_layers`, `dropout`, `pooling_method`, and `use_learnable_interaction`.

**Research notes:** Report feature ordering and preprocessing because feature-sequence models can be affected by how columns are presented. Mambular and MambaTab are motivated by Mamba-style selective state spaces, but their tabular behavior should be validated against dense and tree baselines.

### FTTransformer, TabTransformer, AutoInt, and SAINT

Use when feature interactions are central and the feature-token count is not too large.

| Model | Good Starting Config | When to Prefer |
| ----- | -------------------- | -------------- |
| FTTransformer | `d_model=128`, `n_layers=4`, `n_heads=8`, `attn_dropout=0.1`, `ff_dropout=0.1` | General feature-token attention |
| TabTransformer | `d_model=128`, `n_layers=4`, `n_heads=8`, `attn_dropout=0.1` | Categorical-heavy tables |
| AutoInt | `d_model=128`, `n_layers=3-4`, `n_heads=4-8`, `kv_compression=0.5` | Interaction modeling with optional compression |
| SAINT | `d_model=128`, `n_layers=1-2`, `n_heads=2-4`, `batch_size=32-128` | Row-context or semi-supervised-style experiments |

```python
from deeptab.configs import AutoIntConfig, FTTransformerConfig, SAINTConfig, TabTransformerConfig

ft_cfg = FTTransformerConfig(
    d_model=128,
    n_layers=4,
    n_heads=8,
    attn_dropout=0.1,
    ff_dropout=0.1,
)

tabtransformer_cfg = TabTransformerConfig(
    d_model=128,
    n_layers=4,
    n_heads=8,
    attn_dropout=0.1,
    ff_dropout=0.1,
)

autoint_cfg = AutoIntConfig(
    d_model=128,
    n_layers=4,
    n_heads=8,
    attn_dropout=0.1,
    kv_compression=0.5,
)

saint_cfg = SAINTConfig(
    d_model=128,
    n_layers=1,
    n_heads=2,
    attn_dropout=0.1,
    ff_dropout=0.1,
)
```

**Tune first:** `d_model`, `n_layers`, `n_heads`, `attn_dropout`, and `ff_dropout` where available.

```{tip}
Choose `n_heads` so that `d_model` is divisible by `n_heads`. Common pairs are `(64, 4)`, `(128, 8)`, and `(256, 8 or 16)`.
```

**Research notes:** Attention models can be strong but expensive when feature-token count grows. For SAINT, report batch size because row attention changes both memory use and the effective context available to each row.

### ResNet and MLP

Use as fast baselines and as practical production candidates when the dataset does not justify attention/retrieval overhead.

```python
from deeptab.configs import MLPConfig, ResNetConfig

mlp_cfg = MLPConfig(
    layer_sizes=[256, 128, 32],
    dropout=0.1,
    use_glu=False,
    skip_connections=False,
)

resnet_cfg = ResNetConfig(
    layer_sizes=[256, 128, 32],
    num_blocks=3,
    dropout=0.2,
    norm=False,
)
```

**Tune first:** `layer_sizes`, `dropout`, `num_blocks` for ResNet, and `use_glu` for MLP.

**Research notes:** These models are essential controls. If an advanced architecture does not beat a tuned MLP/ResNet/TabM under the same budget, the added complexity needs justification.

### TabM

Use as a strong parameter-efficient ensemble baseline.

```python
from deeptab.configs import TabMConfig

tabm_cfg = TabMConfig(
    layer_sizes=[256, 256, 128],
    ensemble_size=32,
    model_type="mini",
    dropout=0.1,
    average_ensembles=False,
)
```

**Tune first:** `ensemble_size`, `layer_sizes`, `dropout`, `model_type`, and `average_embeddings`.

**Research notes:** TabM is a useful modern baseline because it tests whether ensemble-like diversity helps without training many independent models. Use a batch size large enough that ensemble outputs are statistically meaningful and memory-safe.

### TabR

Use when nearest-neighbor context is expected to carry target signal.

```python
from deeptab.configs import TabRConfig, TrainerConfig

tabr_cfg = TabRConfig(
    d_main=256,
    context_size=96,
    predictor_n_blocks=1,
    encoder_n_blocks=0,
    context_dropout=0.2,
    dropout0=0.2,
    dropout1=0.0,
    memory_efficient=False,
)

trainer_cfg = TrainerConfig(lr=3e-4, batch_size=256, max_epochs=150, patience=20)
```

**Tune first:** `context_size`, `d_main`, `dropout0`, `context_dropout`, `predictor_n_blocks`, and `candidate_encoding_batch_size`.

**Research notes:** Report candidate pool construction, whether validation/test rows retrieve from training candidates only, and the value of `context_size`. Retrieval leakage can invalidate results.

### NODE, ENODE, and NDTF

Use when you want differentiable tree-inspired models.

```python
from deeptab.configs import ENODEConfig, NDTFConfig, NODEConfig

node_cfg = NODEConfig(
    num_layers=4,
    layer_dim=128,
    depth=6,
    tree_dim=1,
)

enode_cfg = ENODEConfig(
    d_model=8,
    num_layers=4,
    layer_dim=64,
    depth=6,
    tree_dim=1,
)

ndtf_cfg = NDTFConfig(
    min_depth=4,
    max_depth=12,
    n_ensembles=12,
    temperature=0.1,
)
```

**Tune first:** `depth`, `num_layers`, `layer_dim`, `tree_dim`, and for NDTF `n_ensembles`, `min_depth`, `max_depth`, `temperature`.

**Research notes:** NODE-style models evaluate differentiable soft paths rather than performing logarithmic hard-tree traversal. Depth increases leaf/path complexity quickly, so treat `depth` as a high-impact compute and regularization parameter.

## Preprocessing Search

Preprocessing is part of the model in tabular deep learning. Tune it explicitly.

| Data Condition | Candidate Setting | Notes |
| -------------- | ----------------- | ----- |
| Roughly symmetric numerical features | `numerical_preprocessing="standard"` | Fast, simple, and easy to audit |
| Heavy tails/outliers/skew | `numerical_preprocessing="quantile"` | Often robust for real-world tables |
| Bounded features | `numerical_preprocessing="minmax"` | Use when scale bounds are meaningful |
| Nonlinear numeric effects | `numerical_preprocessing="ple"`, tune `n_bins` | Connects to numerical feature embedding work |
| Many integer IDs | `treat_all_integers_as_numerical=True` or tune `cat_cutoff` | Prevents accidental categorical treatment |
| Categorical features | `categorical_preprocessing="int"` or project default | Use model `d_model`/embeddings for representation capacity |

```python
from deeptab.configs import PreprocessingConfig

# Conservative baseline
standard_prep = PreprocessingConfig(
    numerical_preprocessing="standard",
    categorical_preprocessing="int",
)

# Robust numeric preprocessing
quantile_prep = PreprocessingConfig(
    numerical_preprocessing="quantile",
    categorical_preprocessing="int",
)

# Numerical feature embedding/binning experiment
ple_prep = PreprocessingConfig(
    numerical_preprocessing="ple",
    n_bins=64,
    categorical_preprocessing="int",
)
```

```{important}
`PreprocessingConfig` does not own model width. Set representation size with model fields such as `d_model` or `layer_sizes`, not with an `embedding_dim` preprocessing argument.
```

## Search Spaces

Use small spaces first. Expand only after the baseline protocol is stable.

### Mambular

```python
param_grid = {
    "preprocessing_config__numerical_preprocessing": ["standard", "quantile", "ple"],
    "preprocessing_config__n_bins": [32, 64],
    "model_config__d_model": [64, 128, 256],
    "model_config__n_layers": [2, 4, 6],
    "model_config__dropout": [0.0, 0.1, 0.2],
    "model_config__pooling_method": ["avg", "max"],
    "trainer_config__lr": [1e-4, 3e-4, 1e-3],
    "trainer_config__batch_size": [128, 256, 512],
}
```

### FTTransformer

```python
param_grid = {
    "preprocessing_config__numerical_preprocessing": ["standard", "quantile", "ple"],
    "model_config__d_model": [64, 128, 256],
    "model_config__n_layers": [2, 4, 6],
    "model_config__n_heads": [4, 8],
    "model_config__attn_dropout": [0.0, 0.1, 0.2],
    "model_config__ff_dropout": [0.0, 0.1, 0.2],
    "trainer_config__lr": [1e-4, 3e-4, 5e-4],
    "trainer_config__batch_size": [128, 256],
}
```

### TabM

```python
param_grid = {
    "preprocessing_config__numerical_preprocessing": ["standard", "quantile", "ple"],
    "model_config__layer_sizes": [[256, 128], [256, 256, 128], [512, 256, 128]],
    "model_config__ensemble_size": [8, 16, 32],
    "model_config__dropout": [0.0, 0.1, 0.2],
    "model_config__model_type": ["mini", "full"],
    "trainer_config__lr": [3e-4, 1e-3],
    "trainer_config__batch_size": [128, 256, 512],
}
```

### TabR

```python
param_grid = {
    "preprocessing_config__numerical_preprocessing": ["standard", "quantile", "ple"],
    "model_config__d_main": [128, 256],
    "model_config__context_size": [32, 64, 96],
    "model_config__dropout0": [0.0, 0.2, 0.4],
    "model_config__context_dropout": [0.0, 0.2, 0.4],
    "model_config__predictor_n_blocks": [1, 2],
    "trainer_config__lr": [1e-4, 3e-4, 5e-4],
}
```

### NODE

```python
param_grid = {
    "preprocessing_config__numerical_preprocessing": ["standard", "quantile"],
    "model_config__num_layers": [2, 4, 6],
    "model_config__layer_dim": [64, 128, 256],
    "model_config__depth": [4, 6, 8],
    "trainer_config__lr": [3e-4, 1e-3],
    "trainer_config__batch_size": [256, 512],
}
```

## Research Reporting Checklist

Use this checklist when presenting DeepTab results.

- Report model, preprocessing, and trainer configs separately.
- Report DeepTab version/commit, PyTorch version, device, and random seeds.
- State whether hyperparameters were chosen by validation, cross-validation, or fixed defaults.
- Include the trial budget and early-stopping patience.
- Include tuned MLP/ResNet/TabM baselines when evaluating a new architecture.
- For attention models, report feature-token count and batch size.
- For retrieval models, report candidate-pool construction and context size.
- For distributional regression, report NLL and at least one calibration or coverage metric.

## References

The recommendations above are grounded in DeepTab's current config API and in the tabular deep learning literature:

- Ahamed, M. A., & Cheng, Q. (2024). _MambaTab: A Plug-and-Play Model for Learning Tabular Data_. [arXiv:2401.08867](https://arxiv.org/abs/2401.08867)
- Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). _Revisiting Deep Learning Models for Tabular Data_. NeurIPS 2021. [arXiv:2106.11959](https://arxiv.org/abs/2106.11959)
- Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2022). _On Embeddings for Numerical Features in Tabular Deep Learning_. NeurIPS 2022. [arXiv:2203.05556](https://arxiv.org/abs/2203.05556)
- Gorishniy, Y., Rubachev, I., Kartashev, N., Shlenskii, D., Kotelnikov, A., & Babenko, A. (2023). _TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023_. [arXiv:2307.14338](https://arxiv.org/abs/2307.14338)
- Gorishniy, Y., Kotelnikov, A., & Babenko, A. (2024). _TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling_. ICLR 2025. [arXiv:2410.24210](https://arxiv.org/abs/2410.24210)
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). _Why do tree-based models still outperform deep learning on tabular data?_ NeurIPS 2022. [arXiv:2207.08815](https://arxiv.org/abs/2207.08815)
- Gu, A., & Dao, T. (2024). _Mamba: Linear-Time Sequence Modeling with Selective State Spaces_. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Huang, X., Khetan, A., Cvitkovic, M., & Karnin, Z. (2020). _TabTransformer: Tabular Data Modeling Using Contextual Embeddings_. [arXiv:2012.06678](https://arxiv.org/abs/2012.06678)
- Popov, S., Morozov, S., & Babenko, A. (2019). _Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data_. ICLR 2020. [arXiv:1909.06312](https://arxiv.org/abs/1909.06312)
- Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C. B., & Goldstein, T. (2021). _SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training_. [arXiv:2106.01342](https://arxiv.org/abs/2106.01342)
- Song, W., Shi, C., Xiao, Z., Duan, Z., Xu, Y., Zhang, M., & Tang, J. (2019). _AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks_. CIKM 2019. [arXiv:1810.11921](https://arxiv.org/abs/1810.11921)
- Thielmann, A. F., Kumar, M., Weisser, C., Reuter, A., Säfken, B., & Samiee, S. (2024). _Mambular: A Sequential Model for Tabular Deep Learning_. [arXiv:2408.06291](https://arxiv.org/abs/2408.06291)

## See Also

- [Model Comparison](comparison_tables) — Architecture and complexity comparison
- [Config System](../core_concepts/config_system) — Configuration API details

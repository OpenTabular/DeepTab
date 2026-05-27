# Model Comparison

Architectural comparison and computational characteristics of DeepTab's model zoo.

```{note}
**Focus on architecture:** This document emphasizes computational complexity, architectural design, and qualitative comparisons. Quantitative performance benchmarks will be added when systematic experiments are completed.
```

## Computational Characteristics

**Theoretical complexity and architectural properties:**

| Category               | Model          | Parameters (typical) | Inference Complexity | Memory Scaling    | Time Complexity |
| ---------------------- | -------------- | -------------------- | -------------------- | ----------------- | --------------- |
| **State Space Models** | Mambular       | 100K-500K            | O(n·d)               | Linear            | O(n·d)          |
|                        | MambaTab       | 50K-200K             | O(n·d)               | Linear            | O(n·d)          |
|                        | MambAttention  | 200K-1M              | O(n·f²·d)            | Quadratic (f)     | O(n·f²·d)       |
| **Transformers**       | FTTransformer  | 150K-800K            | O(n·f²·d)            | Quadratic (f)     | O(n·f²·d)       |
|                        | TabTransformer | 100K-600K            | O(n·f_cat²·d)        | Quadratic (f_cat) | O(n·f_cat²·d)   |
|                        | SAINT          | 300K-1.5M            | O(n²·f·d)            | Quadratic (n)     | O(n²·f·d)       |
|                        | AutoInt        | 150K-700K            | O(n·f²·d)            | Quadratic (f)     | O(n·f²·d)       |
| **Residual Networks**  | ResNet         | 50K-300K             | O(n·d)               | Linear            | O(n·d)          |
|                        | TabR           | 200K-1M              | O(n·k·d)             | Linear            | O(n·k·d)        |
| **Tree-Based**         | NODE           | 100K-500K            | O(n·d·log n)         | Log-linear        | O(n·d·log n)    |
|                        | ENODE          | 150K-700K            | O(n·d·log n)         | Log-linear        | O(n·d·log n)    |
|                        | NDTF           | 200K-1M              | O(n·d·log n)         | Log-linear        | O(n·d·log n)    |
| **Other**              | MLP            | 30K-200K             | O(n·d)               | Linear            | O(n·d)          |
|                        | TabM           | 80K-400K             | O(n·d)               | Linear            | O(n·d)          |
|                        | TabulaRNN      | 100K-600K            | O(n·l·d)             | Linear            | O(n·l·d)        |

**Notation:** n=samples, d=hidden_dim, f=features, f_cat=categorical features, k=neighbors, l=sequence length.

```{important}
**Parameter count assumptions:** The ranges above assume a **baseline dataset** with:
- **~10 numerical features** + **~5 categorical features** (with ~10 categories each)
- **d_model = 64** (hidden dimension)
- **Default architecture configs** (layers, heads, depth as per model defaults)

Parameter counts scale with:
- **Input features:** More features → larger embedding layers (especially for transformers)
- **Hidden dimension (d_model):** Larger d → quadratic growth (weight matrices are d×d)
- **Architecture depth:** More layers → linear growth
- **Categorical cardinality:** More categories → larger embedding tables
```

```{tip}
**Practical implications:**
- **Linear O(n·d):** Scales well with data size (MLP, ResNet, Mamba variants, TabM)
- **Quadratic O(n·f²):** Attention over features, slower with many features (Transformers)
- **Quadratic O(n²):** Attention over samples, impractical for large datasets (SAINT)
- **Log-linear O(n·log n):** Tree routing, good middle ground (NODE family)
```

```{note}
**Category guide:**
- **State Space Models:** Linear-time selective SSMs (Mamba architecture family)
- **Transformers:** Self-attention mechanisms for feature/sample interactions
- **Residual Networks:** Deep feedforward MLPs with skip connections
- **Tree-Based:** Differentiable decision trees with gradient optimization
- **Other:** Standard architectures (MLP, ensembles, RNNs)
```

## Architecture Categories

### State Space Models (SSMs)

**Linear complexity, efficient long-range dependencies**

| Model         | Layers | Hidden Dim | Key Feature              | Best Use Case         |
| ------------- | ------ | ---------- | ------------------------ | --------------------- |
| Mambular      | 4-12   | 64-512     | Stacked Mamba blocks     | General-purpose       |
| MambaTab      | 1      | 64-256     | Single Mamba block       | Small datasets, speed |
| MambAttention | Hybrid | 128-512    | Mamba + Attention fusion | Complex interactions  |

**References:**

- Gu & Dao (2024). _Mamba: Linear-Time Sequence Modeling_. arXiv:2312.00752

### Transformer-Based

**Attention mechanisms for feature interactions**

| Model          | Attention | Hidden Dim | Key Feature                | Best Use Case          |
| -------------- | --------- | ---------- | -------------------------- | ---------------------- |
| FTTransformer  | Full      | 64-512     | Feature tokenization       | Feature interactions   |
| TabTransformer | Partial   | 64-256     | Categorical-only attention | Categorical-heavy data |
| SAINT          | Row+Col   | 128-512    | Intersample attention      | Semi-supervised        |

**References:**

- Gorishniy et al. (2021). _Revisiting Deep Learning Models for Tabular Data_. NeurIPS 2021
- Huang et al. (2020). _TabTransformer_. arXiv:2012.06678
- Somepalli et al. (2021). _SAINT_. arXiv:2106.01342

### Tree-Inspired

**Neural networks with tree-like structure**

| Model | Tree Type        | Layers | Key Feature     | Best Use Case       |
| ----- | ---------------- | ------ | --------------- | ------------------- |
| NODE  | Oblivious trees  | 6-8    | Soft routing    | Interpretability    |
| ENODE | Extended routing | 6-10   | Enhanced splits | Better than NODE    |
| NDTF  | Forest ensemble  | 8-12   | Multiple trees  | Tree ensemble boost |

**References:**

- Popov et al. (2019). _Neural Oblivious Decision Ensembles_. arXiv:1909.06312

### Residual Networks

**Deep feedforward with skip connections**

| Model  | Blocks | Hidden Dim | Key Feature     | Best Use Case |
| ------ | ------ | ---------- | --------------- | ------------- |
| ResNet | 4-12   | 64-512     | Residual blocks | Fast baseline |
| TabR   | Hybrid | 128-512    | + Retrieval     | Large data    |

**References:**

- He et al. (2016). _Deep Residual Learning_. CVPR 2016
- Gorishniy et al. (2023). _TabR_. arXiv:2307.14338

### Other Architectures

| Model     | Type        | Key Feature           | Best Use Case          |
| --------- | ----------- | --------------------- | ---------------------- |
| MLP       | Feedforward | Simple MLP            | Fastest baseline       |
| TabM      | Ensemble    | Batch ensembling      | Budget ensemble        |
| TabulaRNN | RNN         | Sequential processing | Sequential features    |
| AutoInt   | Attention   | Feature interactions  | Automatic interactions |

## Model Selection by Use Case

```{note}
**General pattern:** Simpler models (MLP, ResNet) work well on small datasets with proper regularization. More complex models (Transformers, SSMs) excel on medium-to-large datasets where their capacity is justified.
```

### By Dataset Size

| Dataset Size       | Recommended Models                     | Reasoning                           | Key Consideration                   | Avoid                                         |
| ------------------ | -------------------------------------- | ----------------------------------- | ----------------------------------- | --------------------------------------------- |
| **<5K samples**    | MambaTab, ResNet, MLP, TabM            | Lower capacity reduces overfitting  | Use high dropout (0.3-0.4)          | Deep Transformers (SAINT, deep FTTransformer) |
| **5K-50K samples** | Mambular, FTTransformer, MambAttention | Architecture complexity pays off    | Balance capacity vs training time   | Very high capacity if data is simple          |
| **>50K samples**   | Mambular, TabR, FTTransformer          | Complex patterns benefit from depth | Watch quadratic scaling bottlenecks | SAINT (O(n²) impractical)                     |

**Alternatives:** MambaTab for speed, NODE/ENODE for interpretability, ResNet for very fast training

### By Feature Type

| Feature Composition  | Best Choice             | Good Alternatives       | Reasoning                                     | Avoid          |
| -------------------- | ----------------------- | ----------------------- | --------------------------------------------- | -------------- |
| **>60% categorical** | TabTransformer          | FTTransformer, Mambular | Categorical-only attention optimized for this | -              |
| **>80% numerical**   | Mambular                | ResNet, NODE            | SSM/dense layers excel on continuous          | TabTransformer |
| **Balanced mixed**   | Mambular, FTTransformer | MambAttention           | Unified feature processing                    | -              |

### By Computational Constraints

| Constraint                | Recommended Models                    | Reasoning                             | Avoid                                   |
| ------------------------- | ------------------------------------- | ------------------------------------- | --------------------------------------- |
| **Memory <8GB GPU**       | MLP, ResNet, MambaTab, Mambular, TabM | O(n·d) linear memory scaling          | FTTransformer, SAINT (quadratic memory) |
| **Fast training needed**  | MLP (fastest), ResNet, MambaTab, TabM | Simple architectures or single blocks | FTTransformer, TabR, SAINT (slow)       |
| **Low inference latency** | MLP, ResNet, Mamba variants, TabM     | O(n) complexity per sample            | Transformers (O(n·f²)), SAINT (O(n²))   |

**Training speed tiers:** Fastest (MLP, ResNet) → Fast (MambaTab, TabM) → Moderate (Mambular, NODE) → Slow (FTTransformer, TabR, SAINT)

### By Task Requirements

| Task                     | General Purpose                            | Fast/Efficient   | Interpretable     | Notes                             |
| ------------------------ | ------------------------------------------ | ---------------- | ----------------- | --------------------------------- |
| **Classification**       | Mambular, FTTransformer, MambAttention     | MambaTab, ResNet | NODE, ENODE, NDTF | All models support multi-class    |
| **Regression**           | Mambular, FTTransformer, TabR (large data) | MambaTab, ResNet | NODE              | Tree models resistant to outliers |
| **LSS (Distributional)** | Mambular, FTTransformer, MambAttention     | MambaTab         | ENODE             | All models support LSS mode       |

**Special cases:** For quantile regression, use any model in LSS mode with appropriate distribution family

## Recommended Decision Tree

```
Start Here
│
├─ Dataset size <5K? → Use MambaTab or ResNet + high dropout (0.3-0.4)
│
├─ Need interpretability? → Use NODE, ENODE, or NDTF
│
├─ Memory constrained (<8GB)? → Avoid Transformers, use Mambular or ResNet
│
├─ Inference latency critical? → Use O(n) models: MLP, ResNet, Mamba variants
│
├─ >60% categorical features? → Consider TabTransformer
│
└─ General purpose → **Mambular** (recommended default)
   └─ Alternative → FTTransformer (if GPU memory available)
```

## References

Complete citations in individual model pages. Key papers:

- Gu & Dao (2024). Mamba: Linear-Time Sequence Modeling. arXiv:2312.00752
- Gorishniy et al. (2021). Revisiting Deep Learning Models for Tabular Data. NeurIPS 2021
- Popov et al. (2019). Neural Oblivious Decision Ensembles. arXiv:1909.06312
- Gorishniy et al. (2023). TabR: Tabular Deep Learning with Retrieval. arXiv:2307.14338

## See Also

- [Recommended Configs](recommended_configs) — Hyperparameter guidelines
- [Model Tiers](../core_concepts/model_tiers) — Stable vs experimental

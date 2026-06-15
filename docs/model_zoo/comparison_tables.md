# Model Comparison

Architectural comparison and computational characteristics of DeepTab's model zoo.

```{note}
**Focus on architecture:** This document emphasizes computational complexity, architectural design, and qualitative comparisons. Quantitative performance benchmarks will be added when systematic experiments are completed.
```

```{seealso}
For practical timing and memory measurement guidance, see [Model Efficiency and Benchmarking](efficiency). For a runnable workflow, use the [Model Efficiency Benchmarking tutorial](../tutorials/model_efficiency) and its notebook at `docs/tutorials/notebooks/model_efficiency.ipynb`.
```

## Computational Characteristics

The table below reports dominant forward-pass scaling for a batch. It is a practical guide, not a FLOP-count benchmark.

| Category               | Model          | DeepTab Default Shape                                  | Dominant Forward-Time Terms                                                                        | Memory Driver                                              | Primary References                                                                                                                      |
| ---------------------- | -------------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **State Space Models** | Mambular       | `d_model=64`, `n_layers=4`                             | Linear in feature sequence: O(B·L·P·D) plus projection constants                                   | O(B·P·D) activations                                       | [Mambular](https://arxiv.org/abs/2408.06291), [Mamba](https://arxiv.org/abs/2312.00752)                                                 |
|                        | MambaTab       | `d_model=64`, `n_layers=1`                             | Linear in feature sequence: O(B·L·P·D) plus projection constants                                   | O(B·P·D) activations                                       | [MambaTab](https://arxiv.org/abs/2401.08867), [Mamba](https://arxiv.org/abs/2312.00752)                                                 |
|                        | MambAttention  | `d_model=64`, Mamba blocks + attention                 | Mamba term O(B·L_m·P·D) plus feature attention O(B·L_a·P²·D)                                       | Attention maps O(B·P²) when attention layers are active    | [Mambular](https://arxiv.org/abs/2408.06291), [Mamba](https://arxiv.org/abs/2312.00752)                                                 |
| **Transformers**       | FTTransformer  | `d_model=128`, `n_layers=4`, `n_heads=8`               | Feature self-attention O(B·L·P²·D) plus feed-forward blocks                                        | O(B·L·P²) attention maps                                   | [Gorishniy et al. 2021](https://arxiv.org/abs/2106.11959)                                                                               |
|                        | TabTransformer | `d_model=128`, `n_layers=4`, `n_heads=8`               | Categorical-token self-attention O(B·L·P_cat²·D) plus numerical MLP head                           | O(B·L·P_cat²) attention maps                               | [Huang et al. 2020](https://arxiv.org/abs/2012.06678)                                                                                   |
|                        | SAINT          | `d_model=128`, `n_layers=1`, `n_heads=2`               | Column attention O(B·P²·D) plus row attention O(B²·P·D) within a batch                             | O(B·P² + B²) attention maps                                | [Somepalli et al. 2021](https://arxiv.org/abs/2106.01342)                                                                               |
|                        | AutoInt        | `d_model=128`, `n_layers=4`, `n_heads=8`               | Feature self-attention O(B·L·P²·D); key-value compression reduces constants                        | O(B·L·P²) attention maps                                   | [Song et al. 2019](https://arxiv.org/abs/1810.11921)                                                                                    |
| **Residual Networks**  | ResNet         | `layer_sizes=[256,128,32]`, `num_blocks=3`             | Dense layers: O(B·sum layer matrix costs)                                                          | Linear in batch and hidden width                           | [He et al. 2016](https://arxiv.org/abs/1512.03385), [Gorishniy et al. 2021](https://arxiv.org/abs/2106.11959)                           |
|                        | TabR           | `d_main=256`, `context_size=96`                        | Candidate encoding plus exact/FAISS nearest-neighbor search O(B·N_c·D) and context mixing O(B·C·D) | Candidate cache O(N_c·D)                                   | [Gorishniy et al. 2023](https://arxiv.org/abs/2307.14338)                                                                               |
| **Tree-Based**         | NODE           | `num_layers=4`, `layer_dim=128`, `depth=6`             | Soft oblivious trees evaluate all splits/leaves: O(B·L·T·(P·D_t + D_t·2^D_t))                      | Path/leaf activations O(B·T·2^D_t)                         | [Popov et al. 2019](https://arxiv.org/abs/1909.06312)                                                                                   |
|                        | ENODE          | `d_model=8`, `num_layers=4`, `layer_dim=64`, `depth=6` | NODE-style soft tree evaluation with learned embeddings                                            | Path/leaf activations O(B·T·2^D_t)                         | [Popov et al. 2019](https://arxiv.org/abs/1909.06312)                                                                                   |
|                        | NDTF           | `n_ensembles=12`, random depths 4-16                   | Neural decision forest evaluates internal nodes and leaf probabilities for each tree               | Leaf probabilities scale with O(B·E·2^D_t)                 | [Kontschieder et al. 2015](https://openaccess.thecvf.com/content_iccv_2015/html/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.html) |
| **Other**              | MLP            | `layer_sizes=[256,128,32]`                             | Dense layers: O(B·sum layer matrix costs)                                                          | Linear in batch and hidden width                           | Standard MLP baseline                                                                                                                   |
|                        | TabM           | `layer_sizes=[256,256,128]`, `ensemble_size=32`        | MLP-style dense compute with parameter-efficient batch ensembling                                  | Linear in batch, hidden width, and active ensemble outputs | [Gorishniy et al. 2024](https://arxiv.org/abs/2410.24210), [Wen et al. 2020](https://arxiv.org/abs/2002.06715)                          |
|                        | TabulaRNN      | `d_model=128`, `n_layers=4`                            | Recurrent feature-sequence processing O(B·L·P·D²) for standard RNN-style cells                     | O(B·P·D) activations                                       | [Thielmann & Samiee 2024](https://arxiv.org/abs/2411.17207)                                                                             |

**Notation:** B=batch size, P=feature tokens after preprocessing/embedding, P_cat=categorical tokens, D=hidden dimension, L=layers, L_m=Mamba layers, L_a=attention layers, C=retrieved context size, N_c=candidate rows for retrieval, T=trees per layer, E=forest ensemble size, D_t=tree depth.

```{important}
**Parameter count assumptions:** Parameter counts are not listed because they depend strongly on dataset schema and preprocessing:
- **Input features:** More features increase embedding, tokenizer, and first-layer parameters.
- **Categorical cardinality:** More categories increase embedding-table parameters.
- **Hidden width:** Dense projections usually scale with width squared.
- **Depth and ensembles:** Additional layers, trees, or ensemble members increase parameters and activations.

The "DeepTab Default Shape" column is taken from the current model config defaults in `deeptab/configs/models/`.
```

```{tip}
**Practical implications:**
- **Linear in feature sequence:** Mamba variants, RNNs, MLPs, ResNets, and TabM avoid feature-attention matrices.
- **Quadratic in features:** FTTransformer, AutoInt, MambAttention attention layers, and TabTransformer become expensive as the number of feature tokens grows.
- **Quadratic in batch rows:** SAINT's row-attention term is controlled by mini-batch size, not by the total dataset size directly.
- **Retrieval-based:** TabR can be strong on larger data, but it needs candidate encoding/search memory and depends on the retrieval index.
- **Soft tree-based:** NODE-style models are not logarithmic at inference; differentiable trees evaluate soft paths/leaves, so tree depth matters.
```

```{note}
**Category guide:**
- **State Space Models:** Selective SSM/Mamba-style sequence models adapted to tabular features.
- **Transformers:** Self-attention mechanisms for feature and/or row interactions.
- **Residual Networks:** Deep feedforward MLPs with skip connections.
- **Tree-Based:** Differentiable decision trees with gradient optimization.
- **Other:** Standard architectures (MLP, parameter-efficient ensembles, RNNs).
```

## Architecture Categories

### State Space Models (SSMs)

**Feature-sequence models with linear sequence-length scaling in the Mamba blocks**

| Model         | Default Layers | Default Hidden Dim | Key Feature                              | Best Use Case                             |
| ------------- | -------------- | ------------------ | ---------------------------------------- | ----------------------------------------- |
| Mambular      | 4 Mamba layers | 64                 | Stacked Mamba blocks over feature tokens | General-purpose tabular sequence modeling |
| MambaTab      | 1 Mamba layer  | 64                 | Lightweight Mamba block                  | Small datasets, speed                     |
| MambAttention | Hybrid         | 64                 | Mamba blocks plus feature attention      | Complex feature interactions              |

**References:**

- Thielmann et al. (2024). _Mambular: A Sequential Model for Tabular Deep Learning_. [arXiv:2408.06291](https://arxiv.org/abs/2408.06291)
- Ahamed & Cheng (2024). _MambaTab: A Plug-and-Play Model for Learning Tabular Data_. [arXiv:2401.08867](https://arxiv.org/abs/2401.08867)
- Gu & Dao (2024). _Mamba: Linear-Time Sequence Modeling with Selective State Spaces_. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

### Transformer-Based

**Attention mechanisms for feature and row interactions**

| Model          | Attention Scope    | Default Hidden Dim | Key Feature                                       | Best Use Case                           |
| -------------- | ------------------ | ------------------ | ------------------------------------------------- | --------------------------------------- |
| FTTransformer  | All feature tokens | 128                | Feature tokenization                              | Feature interactions                    |
| TabTransformer | Categorical tokens | 128                | Contextual categorical embeddings                 | Categorical-heavy data                  |
| SAINT          | Row + column       | 128                | Intersample attention and contrastive pretraining | Semi-supervised or row-context settings |
| AutoInt        | All feature tokens | 128                | Self-attentive feature interaction learning       | Automatic interaction modeling          |

**References:**

- Gorishniy et al. (2021). _Revisiting Deep Learning Models for Tabular Data_. NeurIPS 2021. [arXiv:2106.11959](https://arxiv.org/abs/2106.11959)
- Huang et al. (2020). _TabTransformer: Tabular Data Modeling Using Contextual Embeddings_. [arXiv:2012.06678](https://arxiv.org/abs/2012.06678)
- Somepalli et al. (2021). _SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training_. [arXiv:2106.01342](https://arxiv.org/abs/2106.01342)
- Song et al. (2019). _AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks_. CIKM 2019. [arXiv:1810.11921](https://arxiv.org/abs/1810.11921)

### Tree-Inspired

**Differentiable tree and forest structures**

| Model | Tree Type                      | Default Shape                      | Key Feature                                 | Best Use Case                          |
| ----- | ------------------------------ | ---------------------------------- | ------------------------------------------- | -------------------------------------- |
| NODE  | Oblivious differentiable trees | 4 layers, 128 trees/layer, depth 6 | Soft routing over oblivious trees           | Interpretable tree-inspired modeling   |
| ENODE | Embedded NODE variant          | 4 layers, 64 trees/layer, depth 6  | Feature embeddings before NODE-style blocks | Tree-inspired modeling with embeddings |
| NDTF  | Neural decision tree forest    | 12 trees, random depths 4-16       | Multiple neural decision trees              | Tree ensemble-style experiments        |

**References:**

- Popov et al. (2019). _Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data_. ICLR 2020. [arXiv:1909.06312](https://arxiv.org/abs/1909.06312)
- Kontschieder et al. (2015). _Deep Neural Decision Forests_. ICCV 2015. [CVF Open Access](https://openaccess.thecvf.com/content_iccv_2015/html/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.html)

### Residual Networks

**Deep feedforward networks with skip connections**

| Model  | Default Shape                                   | Key Feature                    | Best Use Case                                  |
| ------ | ----------------------------------------------- | ------------------------------ | ---------------------------------------------- |
| ResNet | 3 residual blocks, `[256, 128, 32]` layer sizes | Residual blocks                | Fast baseline                                  |
| TabR   | `d_main=256`, `context_size=96`                 | Retrieval-augmented prediction | Larger datasets with useful neighbor structure |

**References:**

- He et al. (2016). _Deep Residual Learning for Image Recognition_. CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Gorishniy et al. (2021). _Revisiting Deep Learning Models for Tabular Data_. NeurIPS 2021. [arXiv:2106.11959](https://arxiv.org/abs/2106.11959)
- Gorishniy et al. (2023). _TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023_. [arXiv:2307.14338](https://arxiv.org/abs/2307.14338)

### Other Architectures

| Model     | Type                         | Default Shape                                      | Key Feature                   | Best Use Case               |
| --------- | ---------------------------- | -------------------------------------------------- | ----------------------------- | --------------------------- |
| MLP       | Feedforward                  | `[256, 128, 32]` layer sizes                       | Simple dense baseline         | Fastest baseline            |
| TabM      | Parameter-efficient ensemble | `[256, 256, 128]` layer sizes, 32 ensemble members | Batch ensembling              | Strong efficient baseline   |
| TabulaRNN | RNN                          | `d_model=128`, 4 recurrent layers                  | Sequential feature processing | Sequential feature modeling |
| AutoInt   | Attention                    | `d_model=128`, 4 attention layers                  | Feature interactions          | Automatic interactions      |

**References:**

- Gorishniy et al. (2024). _TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling_. ICLR 2025. [arXiv:2410.24210](https://arxiv.org/abs/2410.24210)
- Wen et al. (2020). _BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning_. [arXiv:2002.06715](https://arxiv.org/abs/2002.06715)
- Thielmann & Samiee (2024). _On the Efficiency of NLP-Inspired Methods for Tabular Deep Learning_. [arXiv:2411.17207](https://arxiv.org/abs/2411.17207)
- Song et al. (2019). _AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks_. CIKM 2019. [arXiv:1810.11921](https://arxiv.org/abs/1810.11921)

## Model Selection by Use Case

```{note}
**General pattern:** Simpler models (MLP, ResNet, TabM) are strong practical baselines and often work well on small or medium datasets with proper regularization. More complex models (Transformers, SSMs, retrieval models) are most useful when their inductive bias matches the data or when the dataset is large enough to justify the extra capacity and compute.
```

### By Dataset Size

| Dataset Size       | Recommended Models                           | Reasoning                                                 | Key Consideration                                       | Avoid                                                   |
| ------------------ | -------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------- |
| **<5K samples**    | MambaTab, ResNet, MLP, TabM                  | Lower capacity and fast iteration reduce overfitting risk | Use regularization and validation-driven early stopping | Deep Transformers (SAINT, deep FTTransformer)           |
| **5K-50K samples** | Mambular, FTTransformer, TabM, MambAttention | More capacity can pay off when features interact strongly | Balance capacity vs training time                       | Very high capacity if data is simple                    |
| **>50K samples**   | Mambular, TabM, TabR, FTTransformer          | Larger data can support complex patterns and retrieval    | Watch attention/retrieval bottlenecks                   | SAINT with large batches unless row attention is needed |

**Alternatives:** MambaTab for speed, NODE/ENODE for tree-inspired interpretability, ResNet/MLP for very fast training.

### By Feature Type

| Feature Composition  | Best Choice             | Good Alternatives       | Reasoning                                                                  | Avoid          |
| -------------------- | ----------------------- | ----------------------- | -------------------------------------------------------------------------- | -------------- |
| **>60% categorical** | TabTransformer          | FTTransformer, Mambular | TabTransformer's attention is focused on categorical contextual embeddings | -              |
| **>80% numerical**   | Mambular, TabM          | ResNet, NODE            | SSM/dense baselines avoid categorical-only assumptions                     | TabTransformer |
| **Balanced mixed**   | Mambular, FTTransformer | MambAttention, TabM     | Unified feature processing supports mixed feature interactions             | -              |

### By Computational Constraints

| Constraint                | Recommended Models                    | Reasoning                                                   | Avoid                                                                    |
| ------------------------- | ------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Memory <8GB GPU**       | MLP, ResNet, MambaTab, Mambular, TabM | No full feature-attention matrix in the main path           | FTTransformer/AutoInt with many feature tokens, SAINT with large batches |
| **Fast training needed**  | MLP, ResNet, MambaTab, TabM           | Simple dense or short sequence paths                        | FTTransformer, TabR, SAINT if retrieval/row attention dominates          |
| **Low inference latency** | MLP, ResNet, Mamba variants, TabM     | Avoids retrieval search and full attention over many tokens | TabR with large candidate pools, wide Transformers                       |

**Training speed tiers:** Fastest (MLP, ResNet) -> Fast (MambaTab, TabM) -> Moderate (Mambular, NODE) -> Slower or workload-dependent (FTTransformer, TabR, SAINT).

### By Task Requirements

| Task                     | General Purpose                            | Fast/Efficient         | Interpretable     | Notes                                                        |
| ------------------------ | ------------------------------------------ | ---------------------- | ----------------- | ------------------------------------------------------------ |
| **Classification**       | Mambular, FTTransformer, MambAttention     | MambaTab, ResNet, TabM | NODE, ENODE, NDTF | All models support multi-class                               |
| **Regression**           | Mambular, FTTransformer, TabR (large data) | MambaTab, ResNet, TabM | NODE              | Tree models can be useful when tree-like splits fit the data |
| **LSS (Distributional)** | Mambular, FTTransformer, MambAttention     | MambaTab               | ENODE             | All models support LSS mode                                  |

**Special cases:** For quantile regression, use any model in LSS mode with an appropriate distribution family.

## Recommended Decision Tree

```
Start Here
|
|- Dataset size <5K? -> Use MambaTab, ResNet, MLP, or TabM with regularization
|
|- Need tree-inspired interpretability? -> Use NODE, ENODE, or NDTF
|
|- Memory constrained (<8GB)? -> Prefer Mambular, MambaTab, MLP, ResNet, or TabM
|
|- Inference latency critical? -> Avoid retrieval/large attention; use MLP, ResNet, TabM, or Mamba variants
|
|- >60% categorical features? -> Consider TabTransformer
|
|- Need retrieval from similar training examples? -> Consider TabR
|
`- General purpose -> Mambular or TabM
   `- Alternative -> FTTransformer when GPU memory and feature count permit
```

## Hardware Requirements by Model

The table below gives practical guidance on whether each model trains comfortably on a **CPU-only machine** or requires a **GPU (CUDA, MPS, or other accelerator)**. Thresholds are rough estimates based on architecture cost, and the actual boundary depends on the number of features, hidden width, and depth used.

```{important}
**Features matter as much as rows.** Transformer-style models grow quadratically with feature-token count, so 20 features with a default FTTransformer config can require as much compute as 50 features with an MLP. The estimates below assume the default DeepTab config for each model and a moderate feature count (10 to 30 columns). Wide datasets shift the GPU threshold lower.
```

| Model              | Family      | CPU-only comfortable up to | GPU strongly recommended above | Primary cost driver                             | Notes                                                                                   |
| ------------------ | ----------- | -------------------------- | ------------------------------ | ----------------------------------------------- | --------------------------------------------------------------------------------------- |
| **MLP**            | Baseline    | ~500K rows                 | ~500K rows                     | Dense layers (cache-friendly)                   | Fastest CPU model; scales well on CPU even for large data                               |
| **ResNet**         | Residual    | ~200K rows                 | ~200K rows                     | Dense + skip-connection blocks                  | Marginally heavier than MLP per step                                                    |
| **TabM**           | Ensemble    | ~100K rows                 | ~100K rows                     | MLP ensemble paths per batch                    | Ensemble overhead is constant; CPU stays competitive                                    |
| **MambaTab**       | State space | ~100K rows                 | ~100K rows                     | Single lightweight Mamba block                  | Lightest SSM variant; GPU advantage modest                                              |
| **Mambular**       | State space | ~20K rows                  | ~20K rows                      | Stacked Mamba blocks over feature tokens        | Mamba CUDA kernels give large GPU speedup; CPU inference still fine                     |
| **MambAttention**  | Hybrid      | ~10K rows                  | ~10K rows                      | Mamba blocks + feature attention                | Attention term adds O(P²) per layer; GPU needed at scale                                |
| **TabulaRNN**      | Recurrent   | ~20K rows                  | ~20K rows                      | Sequential RNN cell over feature tokens         | CPU viable for small datasets; large batches need GPU                                   |
| **TabTransformer** | Transformer | ~20K rows                  | ~20K rows                      | Categorical-token attention                     | Cheaper than full-feature attention; depends on categorical count                       |
| **FTTransformer**  | Transformer | ~10K rows                  | ~10K rows                      | O(P²) full-feature self-attention               | Becomes expensive quickly as feature count grows                                        |
| **AutoInt**        | Transformer | ~10K rows                  | ~10K rows                      | O(P²) feature self-attention                    | Similar profile to FTTransformer                                                        |
| **SAINT**          | Transformer | ~2K rows                   | ~2K rows                       | Column attention + row attention per batch      | Batch size is part of the architecture; CPU impractically slow past a few thousand rows |
| **NODE**           | Tree-based  | ~20K rows                  | ~20K rows                      | Soft-path evaluation exponential in depth       | Depth 6 evaluates 64 leaf activations per tree                                          |
| **ENODE**          | Tree-based  | ~10K rows                  | ~10K rows                      | NODE + learned feature embeddings               | Embedding layer adds compute before tree blocks                                         |
| **NDTF**           | Tree-based  | ~10K rows                  | ~10K rows                      | Forest of soft neural decision trees            | Multiple trees compound the depth-exponential cost                                      |
| **TabR**           | Retrieval   | ~10K rows                  | ~10K rows                      | Candidate encoding + nearest-neighbor retrieval | Retrieval index and candidate encoding scale with training set size                     |

**Legend:**

- _CPU-only comfortable up to_: training at default config typically completes in a reasonable wall-clock time on a modern CPU.
- _GPU strongly recommended above_: training time on CPU becomes a bottleneck; a CUDA, MPS, or similar accelerator provides meaningful speedup.

```{tip}
**Apple Silicon (MPS):** All models run on MPS via PyTorch's MPS backend. Set `accelerator="mps"` in `TrainerConfig`. MPS provides meaningful speedup for most models except those with Mamba CUDA kernels, which fall back to CPU on MPS unless a dedicated MPS implementation is available.
```

```{note}
**Inference vs training:** Inference (predict) is cheaper than training because there is no backward pass or optimizer state. A model that needs a GPU for training can often run inference on CPU in production for moderate batch sizes. Use `InferenceModel` to load artifacts for CPU-only inference environments.
```

### Minimum practical dataset sizes

The thresholds above are about training speed. Deep learning models also have minimum data requirements to learn meaningfully:

| Model family                                          | Minimum rows for useful learning | Reasoning                                                                 |
| ----------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------- |
| MLP, ResNet, TabM                                     | ~500 rows                        | Dense models regularize well; few parameters in shallow configs           |
| Mamba variants, TabulaRNN                             | ~1K rows                         | Sequence inductive bias adds capacity; still workable with early stopping |
| Transformers (FTTransformer, TabTransformer, AutoInt) | ~2K rows                         | Attention needs enough examples to learn meaningful feature interactions  |
| SAINT                                                 | ~2K rows                         | Row attention requires diverse mini-batches                               |
| NODE, ENODE, NDTF                                     | ~2K rows                         | Soft trees need enough samples to define splits                           |
| TabR                                                  | ~5K rows                         | Retrieval quality depends on having a meaningful candidate pool           |

---

## References

Key papers used for the comparison:

- Ahamed, M. A., & Cheng, Q. (2024). _MambaTab: A Plug-and-Play Model for Learning Tabular Data_. [arXiv:2401.08867](https://arxiv.org/abs/2401.08867), [DOI:10.1109/MIPR62202.2024.00065](https://doi.org/10.1109/MIPR62202.2024.00065)
- Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). _Revisiting Deep Learning Models for Tabular Data_. NeurIPS 2021. [arXiv:2106.11959](https://arxiv.org/abs/2106.11959)
- Gorishniy, Y., Rubachev, I., Kartashev, N., Shlenskii, D., Kotelnikov, A., & Babenko, A. (2023). _TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023_. [arXiv:2307.14338](https://arxiv.org/abs/2307.14338)
- Gorishniy, Y., Kotelnikov, A., & Babenko, A. (2024). _TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling_. ICLR 2025. [arXiv:2410.24210](https://arxiv.org/abs/2410.24210)
- Gu, A., & Dao, T. (2024). _Mamba: Linear-Time Sequence Modeling with Selective State Spaces_. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). _Deep Residual Learning for Image Recognition_. CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Huang, X., Khetan, A., Cvitkovic, M., & Karnin, Z. (2020). _TabTransformer: Tabular Data Modeling Using Contextual Embeddings_. [arXiv:2012.06678](https://arxiv.org/abs/2012.06678)
- Kontschieder, P., Fiterau, M., Criminisi, A., & Rota Bulo, S. (2015). _Deep Neural Decision Forests_. ICCV 2015. [CVF Open Access](https://openaccess.thecvf.com/content_iccv_2015/html/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.html)
- Popov, S., Morozov, S., & Babenko, A. (2019). _Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data_. ICLR 2020. [arXiv:1909.06312](https://arxiv.org/abs/1909.06312)
- Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C. B., & Goldstein, T. (2021). _SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training_. [arXiv:2106.01342](https://arxiv.org/abs/2106.01342)
- Song, W., Shi, C., Xiao, Z., Duan, Z., Xu, Y., Zhang, M., & Tang, J. (2019). _AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks_. CIKM 2019. [arXiv:1810.11921](https://arxiv.org/abs/1810.11921)
- Thielmann, A. F., Kumar, M., Weisser, C., Reuter, A., Säfken, B., & Samiee, S. (2024). _Mambular: A Sequential Model for Tabular Deep Learning_. [arXiv:2408.06291](https://arxiv.org/abs/2408.06291)
- Thielmann, A. F., & Samiee, S. (2024). _On the Efficiency of NLP-Inspired Methods for Tabular Deep Learning_. [arXiv:2411.17207](https://arxiv.org/abs/2411.17207)
- Wen, Y., Tran, D., & Ba, J. (2020). _BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning_. [arXiv:2002.06715](https://arxiv.org/abs/2002.06715)

## See Also

- [Recommended Configs](recommended_configs): Hyperparameter guidelines
- [Model Efficiency and Benchmarking](efficiency): Runtime and memory benchmarking protocol
- [Model Tiers](../core_concepts/model_tiers): Stable vs experimental

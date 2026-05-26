# Model Comparison Tables

Systematic comparison of all DeepTab models across key dimensions.

## Quick Reference

| Model                                   | Speed      | Accuracy   | Memory     | Interpretability | Best For                    |
| --------------------------------------- | ---------- | ---------- | ---------- | ---------------- | --------------------------- |
| [Mambular](stable/mambular)             | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐             | General-purpose, large data |
| [FTTransformer](stable/fttransformer)   | ⭐⭐⭐     | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   | ⭐⭐             | Feature interactions        |
| [ResNet](stable/resnet)                 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ | ⭐⭐⭐           | Fast baseline               |
| [MambaTab](stable/mambatab)             | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ | ⭐⭐             | Small datasets, speed       |
| [MambAttention](stable/mambattention)   | ⭐⭐⭐     | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   | ⭐⭐             | Complex interactions        |
| [TabTransformer](stable/tabtransformer) | ⭐⭐⭐⭐   | ⭐⭐⭐⭐   | ⭐⭐⭐⭐   | ⭐⭐             | Categorical-heavy           |
| [SAINT](stable/saint)                   | ⭐⭐       | ⭐⭐⭐⭐⭐ | ⭐⭐⭐     | ⭐⭐             | Semi-supervised             |
| [TabM](stable/tabm)                     | ⭐⭐⭐⭐   | ⭐⭐⭐⭐   | ⭐⭐⭐⭐   | ⭐⭐             | Ensemble on budget          |
| [TabR](stable/tabr)                     | ⭐⭐⭐     | ⭐⭐⭐⭐⭐ | ⭐⭐⭐     | ⭐⭐             | Large data, locality        |
| [MLP](stable/mlp)                       | ⭐⭐⭐⭐⭐ | ⭐⭐⭐     | ⭐⭐⭐⭐⭐ | ⭐⭐⭐           | Fastest baseline            |
| [NODE](stable/node)                     | ⭐⭐⭐     | ⭐⭐⭐⭐   | ⭐⭐⭐⭐   | ⭐⭐⭐⭐         | Tree inductive bias         |
| [ENODE](stable/enode)                   | ⭐⭐⭐     | ⭐⭐⭐⭐   | ⭐⭐⭐     | ⭐⭐⭐⭐         | Enhanced NODE               |
| [NDTF](stable/ndtf)                     | ⭐⭐⭐     | ⭐⭐⭐⭐   | ⭐⭐⭐⭐   | ⭐⭐⭐⭐         | Tree ensemble               |
| [TabulaRNN](stable/tabularnn)           | ⭐⭐       | ⭐⭐⭐     | ⭐⭐⭐⭐   | ⭐⭐             | Sequential features         |
| [AutoInt](stable/autoint)               | ⭐⭐⭐     | ⭐⭐⭐⭐   | ⭐⭐⭐⭐   | ⭐⭐             | Feature interactions        |

## Training Speed Comparison

Relative training time on a typical dataset (lower is better):

| Model          | Relative Time | GPU Utilization | Scales to Large Data |
| -------------- | ------------- | --------------- | -------------------- |
| MLP            | 1.0x          | Good            | ✅                   |
| ResNet         | 1.2x          | Good            | ✅                   |
| MambaTab       | 1.5x          | Good            | ✅                   |
| TabM           | 1.8x          | Good            | ✅                   |
| Mambular       | 2.0x          | Excellent       | ✅                   |
| NODE           | 2.2x          | Moderate        | ⚠️                   |
| TabTransformer | 2.5x          | Good            | ✅                   |
| MambAttention  | 2.8x          | Good            | ✅                   |
| AutoInt        | 3.0x          | Good            | ✅                   |
| FTTransformer  | 3.2x          | Good            | ⚠️                   |
| NDTF           | 3.5x          | Moderate        | ⚠️                   |
| TabR           | 3.8x          | Good            | ✅                   |
| TabulaRNN      | 4.0x          | Moderate        | ⚠️                   |
| SAINT          | 4.5x          | Moderate        | ⚠️                   |

## Accuracy by Dataset Size

Recommended models for different dataset sizes:

### Small Datasets (<5K samples)

1. **MambaTab** — Fast, prevents overfitting
2. **TabM** — Ensemble benefits at low cost
3. **ResNet** — Simple and effective
4. **MLP** — Fastest baseline

### Medium Datasets (5K-50K samples)

1. **Mambular** — Best overall
2. **FTTransformer** — Strong baseline
3. **MambAttention** — Complex interactions
4. **TabTransformer** — If categorical-heavy

### Large Datasets (>50K samples)

1. **Mambular** — Scales excellently
2. **TabR** — Leverages large training set
3. **FTTransformer** — Still competitive
4. **ResNet** — Fast alternative

## Task-Specific Recommendations

### Classification

**Top performers:**

1. Mambular
2. FTTransformer
3. MambAttention
4. SAINT (if semi-supervised)

**Fast alternatives:**

- ResNet
- MambaTab
- TabM

### Regression

**Top performers:**

1. Mambular
2. FTTransformer
3. TabR (large datasets)
4. MambAttention

**Fast alternatives:**

- ResNet
- MLP
- NODE

### LSS (Distributional Regression)

**Top performers:**

1. Mambular
2. FTTransformer
3. MambAttention
4. ENODE

**Fast alternatives:**

- ResNet
- MambaTab

## Data Type Recommendations

### Categorical-Heavy (>50% categorical features)

1. **TabTransformer** — Specialized for categoricals
2. **FTTransformer** — Handles all feature types
3. **Mambular** — General-purpose strong performance

### Numerical-Heavy (>80% numerical features)

1. **Mambular** — Excellent on numerical
2. **ResNet** — Simple and effective
3. **FTTransformer** — Still works well

### Mixed Data (balanced numerical/categorical)

1. **Mambular** — Best overall
2. **FTTransformer** — Strong baseline
3. **MambAttention** — Complex patterns

## Computational Budget

### Limited Compute

**Best choices:**

1. MLP — Fastest
2. ResNet — Fast + good accuracy
3. MambaTab — Efficient modern architecture

### Moderate Compute

**Best choices:**

1. Mambular — Best balance
2. TabM — Ensemble benefits
3. TabTransformer — If categorical-heavy

### High Compute Available

**Best choices:**

1. FTTransformer — Maximum accuracy
2. SAINT — If semi-supervised
3. MambAttention — Complex modeling

## Memory Requirements

### Low Memory (<4GB GPU)

Compatible models:

- MLP
- ResNet
- MambaTab
- TabM
- NODE

### Medium Memory (4-16GB GPU)

All models work, optimal:

- Mambular
- FTTransformer
- TabTransformer
- MambAttention

### High Memory (>16GB GPU)

Best utilization:

- SAINT (large batches)
- TabR (large retrieval sets)
- FTTransformer (many features)

## Interpretability vs Performance

| Interpretability Tier | Models                                | Trade-off             |
| --------------------- | ------------------------------------- | --------------------- |
| High                  | NODE, ENODE, NDTF                     | Some accuracy loss    |
| Medium                | ResNet, MLP                           | Simpler architectures |
| Low                   | Mambular, FTTransformer, Transformers | Maximum performance   |

## Feature Count Considerations

### Few Features (<10)

- MLP, ResNet work well
- Mambular still competitive
- Avoid over-parameterization

### Medium Features (10-50)

- All models perform well
- Mambular, FTTransformer excel
- Choose based on other criteria

### Many Features (>50)

- Mambular scales well
- FTTransformer may struggle (attention complexity)
- TabR handles large feature sets
- Consider feature selection

## Summary Decision Tree

```
Need maximum accuracy?
├─ Yes → Mambular or FTTransformer
└─ No
   ├─ Need speed?
   │  ├─ Yes → ResNet or MLP
   │  └─ No → Continue
   ├─ Categorical-heavy?
   │  ├─ Yes → TabTransformer
   │  └─ No → Continue
   ├─ Need interpretability?
   │  ├─ Yes → NODE or NDTF
   │  └─ No → Continue
   ├─ Small dataset (<5K)?
   │  ├─ Yes → MambaTab or TabM
   │  └─ No → Mambular
```

## Benchmark Results

See [GitHub repository](https://github.com/basf/DeepTab) for detailed benchmark results on:

- OpenML-CC18 datasets
- Kaggle competition datasets
- Custom evaluation benchmarks

## See Also

- [Recommended Configs](recommended_configs) — Hyperparameter settings
- [Model Zoo Index](index) — Individual model pages
- [Tutorials](../tutorials/index) — Usage examples

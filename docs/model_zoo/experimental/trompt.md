# Trompt

**Transformer with Prompting for Tabular Data** — Experimental architecture using prompt-based learning paradigm.

```{warning}
**⚠️ EXPERIMENTAL MODEL:** API not semantically versioned. May change in minor releases. Pin exact DeepTab version (`deeptab==x.y.z`) when using in production. See [Model Tiers](../../core_concepts/model_tiers) for details.
```

## Architecture Overview

**Core mechanism:** Transformer with learnable prompts for task conditioning  
**Complexity:** O(n·f·d) per forward pass where f = feature count  
**Inductive bias:** Prompt-based conditioning guides feature processing

### Key Components

1. **Feature embeddings:** Maps inputs to representation space
2. **Learnable prompts:** Task-specific tokens prepended to inputs
3. **Transformer layers:** Self-attention over prompts + features
4. **Prompt-conditioned output:** Predictions influenced by learned prompts

```{note}
**Research motivation:** Explores prompt-based learning (successful in NLP) for tabular data. Hypothesis: learnable prompts can capture task-specific patterns and improve feature representations through attention-based conditioning.
```

## Experimental Status

| Aspect                  | Status            | Implications                                           |
| ----------------------- | ----------------- | ------------------------------------------------------ |
| **API stability**       | ⚠️ Not guaranteed | Pin version: `deeptab==2.0.0`                          |
| **Semantic versioning** | ❌ Not covered    | Breaking changes possible in minor releases            |
| **Promotion criteria**  | In evaluation     | Needs consistent outperformance + community validation |
| **Production use**      | Use with caution  | Pin version, monitor release notes                     |
| **Research stage**      | Early validation  | Limited benchmarking, unclear when prompts help        |

```{important}
**Version pinning essential:** Always specify exact version in requirements:

    # requirements.txt
    deeptab==2.0.0  # Exact version, not >=2.0.0
```

## When to Use

| Scenario                           | Recommendation                                  | Reasoning                               |
| ---------------------------------- | ----------------------------------------------- | --------------------------------------- |
| **Exploring prompt-based methods** | ✅ Try Trompt                                   | Cutting-edge paradigm                   |
| **Research/experimentation**       | ✅ Try Trompt                                   | Novel approach worth testing            |
| **Can handle API changes**         | ✅ Try Trompt                                   | Version pinning and monitoring feasible |
| **Task conditioning hypothesis**   | ✅ Try Trompt                                   | Learnable prompts may help              |
| **Production deployment**          | ❌ Use [FTTransformer](../stable/fttransformer) | Stable transformer alternative          |
| **Need stable API**                | ❌ Use stable models                            | Experimental = no guarantees            |
| **Cannot monitor updates**         | ❌ Use stable models                            | API may break silently                  |
| **Limited experimentation time**   | ❌ Use proven models                            | Trompt requires validation              |

## Configuration

### Model Config (TromptConfig)

```{warning}
**Config API may change:** Parameter names, defaults, and valid ranges subject to change in future releases without major version bump.
```

| Parameter    | Current Default | Description                 | Status               |
| ------------ | --------------- | --------------------------- | -------------------- |
| `d_model`    | 128             | Embedding dimension         | May change           |
| `n_heads`    | 8               | Attention heads             | May change           |
| `n_layers`   | 6               | Transformer layers          | May change           |
| `n_prompts`  | 4               | Number of learnable prompts | May be added/renamed |
| `prompt_dim` | d_model         | Prompt dimension            | May change           |

### Example Configuration

```python
from deeptab.configs import TromptConfig

# Check version!
import deeptab
print(f"DeepTab version: {deeptab.__version__}")  # Ensure matches pinned version

cfg = TromptConfig(
    d_model=128,
    n_heads=8,
    n_layers=6,
    n_prompts=4,  # May change in future versions
)
```

## Quick Start

```python
from deeptab.models.experimental import TromptClassifier, TromptRegressor

# ⚠️ ALWAYS PIN VERSION IN PRODUCTION
# pip install deeptab==2.0.0

# Check version first
import deeptab
assert deeptab.__version__ == "2.0.0", "Version mismatch!"

# Standard usage
model = TromptClassifier()
model.fit(X_train, y_train, max_epochs=50)
predictions = model.predict(X_test)

# Compare with stable transformer (FTTransformer)
from deeptab.models import FTTransformerClassifier
baseline = FTTransformerClassifier()
baseline.fit(X_train, y_train, max_epochs=50)
# Evaluate if prompts provide improvement

# Note: API may change - refer to release notes for current version
```

## Research Context

### Theoretical Foundation

**Standard transformer (FTTransformer):**

```
Input: [feature₁, feature₂, ..., featureₙ]
       ↓ self-attention
Output: predictions
```

**Prompt-based transformer (Trompt):**

```
Input: [prompt₁, prompt₂, ..., promptₘ, feature₁, feature₂, ..., featureₙ]
       ↓ self-attention (prompts attend to features, features attend to prompts)
Output: prompt-conditioned predictions
```

**Learnable prompts:**

$$
\mathbf{P} = [\mathbf{p}_1, \mathbf{p}_2, ..., \mathbf{p}_m] \in \mathbb{R}^{m \times d}
$$

Optimized during training to capture task-specific patterns.

### Potential Advantages

| Aspect                   | Standard Transformer | Trompt               | Hypothesis               |
| ------------------------ | -------------------- | -------------------- | ------------------------ |
| **Task conditioning**    | Implicit in weights  | Explicit via prompts | More flexible adaptation |
| **Feature processing**   | Direct attention     | Prompt-mediated      | Better guidance          |
| **Multi-task potential** | Single task          | Prompt per task      | Could generalize         |
| **Interpretability**     | Attention weights    | Prompt + attention   | Prompt analysis possible |

```{note}
**Research status:** Promising concept from NLP domain. Unclear if prompt-based learning advantages transfer to tabular data. Requires extensive evaluation to determine when/why prompts help.
```

## Performance Characteristics

### Preliminary Observations

```{warning}
**Limited benchmarking:** Results based on initial experiments. More comprehensive evaluation needed for robust conclusions.
```

| Aspect                   | Observation                                     | Caveat                 |
| ------------------------ | ----------------------------------------------- | ---------------------- |
| **Accuracy**             | Competitive with FTTransformer on some datasets | High variance          |
| **Training speed**       | Similar to FTTransformer                        | Comparable overhead    |
| **Prompt effectiveness** | Unclear when prompts help                       | Needs characterization |
| **Memory**               | Slightly higher (prompts)                       | ~10-20% overhead       |

### Comparison with Alternatives

| vs Model          | Status | When to Prefer Trompt         | When to Prefer Alternative |
| ----------------- | ------ | ----------------------------- | -------------------------- |
| **FTTransformer** | Stable | Research/experimentation      | Production, stable API     |
| **Mambular**      | Stable | Prompt hypothesis interesting | General purpose            |
| **ResNet**        | Stable | Exploring attention + prompts | Fast baseline              |

## Known Limitations

```{warning}
**Current limitations (subject to change):**
- **Experimental status:** No API stability guarantees
- **Limited validation:** Fewer datasets/benchmarks than stable models
- **Unclear advantage:** When/why prompts help not well-characterized
- **Prompt design:** Optimal number and dimension unclear
- **Hyperparameter sensitivity:** More parameters to tune than baseline transformer
- **Computational overhead:** Prompts add sequence length
- **Theory less developed:** Prompt-based tabular learning understudied
```

## Best Practices for Experimental Models

### Version Management

```python
# ✅ GOOD: Pin exact version
# requirements.txt
deeptab==2.0.0

# ❌ BAD: Allow any compatible version
# deeptab>=2.0.0  # Could break on 2.0.1!
```

### Monitoring for Changes

```{tip}
**Stay informed:**
1. Monitor DeepTab release notes closely
2. Join community discussions (GitHub issues)
3. Test thoroughly after any update
4. Have migration plan to stable models
5. Set up alerts for new releases
```

### Evaluation Protocol

```python
# Systematic evaluation before production use
import deeptab
assert deeptab.__version__ == "2.0.0"

from deeptab.models.experimental import TromptClassifier
from deeptab.models import FTTransformerClassifier

# Compare Trompt with stable transformer baseline
trompt = TromptClassifier()
trompt.fit(X_train, y_train, max_epochs=50)
trompt_score = trompt.score(X_test, y_test)

fttransformer = FTTransformerClassifier()
fttransformer.fit(X_train, y_train, max_epochs=50)
ft_score = fttransformer.score(X_test, y_test)

# Only use Trompt if clear improvement
if trompt_score > ft_score + 0.02:  # 2% threshold
    print("Trompt provides clear benefit on this dataset")
else:
    print("Stick with FTTransformer (stable)")
```

## Prompt Analysis

```{tip}
**Interpreting learned prompts:** After training, examine prompt attention patterns to understand how prompts condition feature processing. High attention from prompt to feature suggests that prompt learns to focus on relevant features.
```

**Analyzing prompts (conceptual):**

```python
# After training
model = TromptClassifier()
model.fit(X_train, y_train, max_epochs=50)

# Access learned prompts (requires model internals)
# prompts = model.model.prompts  # Shape: [n_prompts, d_model]

# Analyze attention: which features do prompts attend to?
# High attention → prompt conditions that feature strongly
```

## Experimental Workflow

```{tip}
**Recommended approach:**
1. Start with stable transformer ([FTTransformer](../stable/fttransformer))
2. If interested in prompt-based learning, try Trompt
3. Validate improvement on held-out test set
4. Analyze prompt behavior (attention patterns)
5. Pin version if deploying
6. Monitor for updates and evaluate migration path
```

**Decision tree:**

```
Need transformer architecture?
    ↓ No → Try other architectures
    ↓ Yes
FTTransformer sufficient?
    ↓ Yes → Stay with stable
    ↓ No (want to explore prompts)
Can handle API instability?
    ↓ No → Stay with FTTransformer
    ↓ Yes
→ Try Trompt (pin version!)
    ↓
Provides >2% improvement?
    ↓ No → Return to FTTransformer
    ↓ Yes
→ Deploy with version pinning and monitoring
```

## Architecture Details

### Prompt-Augmented Attention

**Standard self-attention (FTTransformer):**

```
Features: [f₁, f₂, ..., fₙ]
          ↓ self-attention
Features attend to each other
```

**Prompt-augmented attention (Trompt):**

```
Sequence: [p₁, p₂, ..., pₘ, f₁, f₂, ..., fₙ]
           ↓ self-attention
Prompts ↔ Features (bidirectional attention)
   ↓
Prompt-conditioned feature representations
```

### Mathematical Formulation

**Feature embeddings:**

$$
\mathbf{E}_f = [\mathbf{e}_1, \mathbf{e}_2, ..., \mathbf{e}_n] \in \mathbb{R}^{n \times d}
$$

**Learnable prompts:**

$$
\mathbf{P} = [\mathbf{p}_1, \mathbf{p}_2, ..., \mathbf{p}_m] \in \mathbb{R}^{m \times d}
$$

**Combined sequence:**

$$
\mathbf{S} = [\mathbf{P}; \mathbf{E}_f] \in \mathbb{R}^{(m+n) \times d}
$$

**Self-attention over combined sequence:**

$$
\mathbf{S}' = \text{TransformerLayers}(\mathbf{S})
$$

**Output from prompt tokens:**

$$
\hat{y} = \text{Head}(\text{Pool}(\mathbf{S}'_{1:m}))
$$

Where $\mathbf{S}'_{1:m}$ are the updated prompt representations.

### Full Architecture

```
Input features [f₁, f₂, ..., fₙ]
        ↓
Feature embedding
   [e₁, e₂, ..., eₙ]
        ↓
Prepend learnable prompts
   [p₁, p₂, ..., pₘ, e₁, e₂, ..., eₙ]
        ↓
╔═══════════════════════════════╗
║ Transformer Layer 1           ║
║ Self-attention (all tokens)   ║
║ - Prompts attend to features  ║
║ - Features attend to prompts  ║
║ - Features attend to features ║
║ Feed-forward                  ║
╚═══════════════════════════════╝
        ↓
╔═══════════════════════════════╗
║ Transformer Layer 2           ║
║ (similar structure)           ║
╚═══════════════════════════════╝
        ↓
    ... (L layers)
        ↓
Extract prompt representations
   [p₁', p₂', ..., pₘ']
        ↓
Pooling (e.g., mean or first prompt)
        ↓
Output head
        ↓
Predictions
```

## Migration to Stable Models

```{important}
**Exit strategy:** If Trompt doesn't work out or API changes are disruptive:

**Similar stable alternatives:**
- [FTTransformer](../stable/fttransformer) — Stable transformer without prompts
- [Mambular](../stable/mambular) — Stable general-purpose model
- [ResNet](../stable/resnet) — Fast stable baseline

**Migration path:**

    # Trompt (experimental)
    from deeptab.models.experimental import TromptClassifier
    model = TromptClassifier()

    # → FTTransformer (stable)
    from deeptab.models import FTTransformerClassifier
    model = FTTransformerClassifier()  # Same API, no prompts!
```

## API Change Examples

```{warning}
**Past API changes (hypothetical examples):**

**v2.0.0 → v2.1.0:**
- Parameter `n_prompts` → `num_prompt_tokens` (renamed)
- Added required parameter `prompt_init_strategy` (breaking)
- Changed default `n_prompts` from 4 → 8 (behavior change)
- Removed `prompt_dim` (now always equals d_model)

**Impact:** Code using v2.0.0 breaks on v2.1.0 without modification.

**Protection:** Pin to `deeptab==2.0.0` exactly.
```

## Prompt-based Learning in NLP vs Tabular

**NLP success:**

- Prompts guide language models effectively
- Pre-training + prompting works well
- Clear semantic meaning to prompts

**Tabular challenges:**

- No pre-training (typically)
- Less clear what prompts "mean"
- Feature semantics differ from language

**Open questions:**

- Do prompts help tabular data similarly?
- How many prompts optimal?
- What do learned prompts represent?

## Community Feedback

```{note}
**Help improve Trompt:** If you experiment with this model:

1. Share results (GitHub issues/discussions)
2. Report scenarios where prompts help/don't help
3. Analyze learned prompt patterns
4. Suggest improvements to prompt mechanism

Community validation essential for promotion to stable tier!
```

## References

**Prompt-based learning in NLP:**

- Lester, B., et al. (2021). _The Power of Scale for Parameter-Efficient Prompt Tuning_. EMNLP 2021
- Li, X., & Liang, P. (2021). _Prefix-Tuning: Optimizing Continuous Prompts_. ACL 2021

**Transformers for tabular data:**

- Gorishniy, Y., et al. (2021). _Revisiting Deep Learning Models for Tabular Data_. NeurIPS 2021. (FTTransformer)

**Prompt learning:**

- Various applications of learnable prompts in deep learning

## See Also

- [Model Tiers](../../core_concepts/model_tiers) — Understanding stable vs experimental
- [Experimental Models Tutorial](../../tutorials/experimental) — Best practices
- [FTTransformer](../stable/fttransformer) — Stable transformer alternative
- [Mambular](../stable/mambular) — Stable general-purpose model
- [Version Pinning Guide](../../developer_guide/version_pinning) — Managing experimental dependencies

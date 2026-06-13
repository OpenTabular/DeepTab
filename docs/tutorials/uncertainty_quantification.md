# Uncertainty Quantification

<div style="display: flex; gap: 10px; margin-bottom: 20px;">
  <a href="https://colab.research.google.com/github/OpenTabular/DeepTab/blob/main/docs/tutorials/notebooks/uncertainty_quantification.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://github.com/OpenTabular/DeepTab/blob/main/docs/tutorials/notebooks/uncertainty_quantification.ipynb" target="_blank">
    <img src="https://img.shields.io/badge/View%20on-GitHub-181717?logo=github&logoColor=white" alt="View on GitHub"/>
  </a>
</div>

A point regressor answers "what value?" but never "how sure are you?". For pricing,
demand, latency, or risk, the second question is often the one that matters. This
tutorial builds a model that answers it. Distributional regression, marked by the
`*LSS` suffix in DeepTab, predicts the parameters of a full conditional
distribution for every row, so you get calibrated prediction intervals and an
uncertainty estimate that changes with the input (heteroscedasticity).

We construct a deliberately heteroscedastic problem, show why a point regressor
cannot represent it, train a `NODELSS` model, verify that its intervals are
calibrated, confirm it recovers the true input-dependent noise, score it with
proper scoring rules, and select a distribution family for a heavy-tailed target.

```{note}
The notebook linked above is generated from this same tutorial content. The markdown page is the readable lesson; the notebook is the executable copy.
```

## What You Will Learn

- How to train a `*LSS` model and read its predicted distribution parameters.
- Why a point regressor cannot express input-dependent uncertainty, and how LSS recovers it.
- How to build prediction intervals and verify their calibration across nominal levels.
- How to choose a distribution family by matching the target's support and tails, scored with CRPS.
- How `evaluate()` reports proper scoring rules and how `score()` returns the negative log-likelihood.
- How to serve an uncertainty-aware model with `InferenceModel.predict_params()`.

## Setup

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from deeptab.configs import NODEConfig, PreprocessingConfig, TrainerConfig
from deeptab.core.observability import ObservabilityConfig
from deeptab.core.reproducibility import set_seed
from deeptab.models import NODELSS, NODERegressor
```

```{note}
For a quick demonstration these tutorials train with very low `max_epochs` and `patience` (5 and 2). Treat these as placeholders and choose values that match your own compute budget and problem. As a starting point, at least `max_epochs=100` and `patience=10` are recommended for meaningful results.
```

```python
import logging
import warnings

# These tutorials use small synthetic datasets and short training runs, which
# surfaces a few non-actionable framework messages. Quieten them so the output
# stays focused on the tutorial; none of them affect correctness.
warnings.filterwarnings("ignore", message=".*n_quantiles.*")
warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*have no logger configured.*")
warnings.filterwarnings("ignore", message=".*lr_patience.*")
warnings.filterwarnings("ignore", message=".*Checkpoint directory.*")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
```

## A Heteroscedastic Dataset

The defining feature of an uncertainty problem is that the spread of the target,
not just its mean, depends on the inputs. We build exactly that: the conditional
mean is a smooth function of several drivers, but the noise standard deviation
grows with one of them. Because we generate the noise ourselves, we know the true
`sigma(x)` and can later check whether the model recovered it.

```python
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
N = 6000

X = pd.DataFrame({
    "load":      rng.uniform(0.0, 1.0, N),     # drives both the mean and the noise
    "distance":  rng.uniform(0.0, 1.0, N),
    "priority":  rng.normal(0.0, 1.0, N),
    "size":      rng.gamma(2.0, 1.0, N),
})

# Conditional mean: smooth, nonlinear function of the drivers
mean = 20.0 + 30.0 * X["load"] + 12.0 * np.sin(3.0 * X["distance"]) + 4.0 * X["priority"]

# Heteroscedastic noise: standard deviation grows sharply with load
true_sigma = 1.5 + 9.0 * X["load"] ** 2
y = (mean + rng.normal(0.0, true_sigma)).to_numpy()

print(f"target range: [{y.min():.1f}, {y.max():.1f}]")
print(f"true sigma range: [{true_sigma.min():.2f}, {true_sigma.max():.2f}]")
```

```
target range: [...]
true sigma range: [1.50, 10.50]
```

The noise at high `load` is roughly seven times wider than at low `load`. A single
error bar for the whole dataset would be wrong almost everywhere; that is the gap
distributional regression closes.

```python
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_STATE)
sigma_test = (1.5 + 9.0 * X_test["load"] ** 2).to_numpy()   # ground-truth noise on the test split

print(f"Train: {len(y_train)}  |  Val: {len(y_val)}  |  Test: {len(y_test)}")
```

## Reproducibility and Shared Configuration

`set_seed` fixes initialisation, dropout, and shuffling across CPU, CUDA, and MPS.
We reuse one preprocessing and trainer configuration so the point baseline and the
LSS model differ only in what they predict.

```python
PREPROC = PreprocessingConfig(
    numerical_preprocessing="ple",   # piecewise-linear encoding of numericals
    n_bins=64,
)
TRAINER = TrainerConfig(
    max_epochs=5,
    batch_size=256,
    lr=1e-3,
    patience=2,
    weight_decay=1e-5,
)
FIT_KWARGS = dict(X_val=X_val, y_val=y_val)
```

## Why Point Regression Is Not Enough

Train an ordinary regressor first. It fits the conditional mean well, but its
output is a single number per row with no notion of spread. Splitting the test set
into a low-noise and a high-noise half makes the missing information obvious: the
residuals are far wider in the high-load half, yet the point model reports nothing
to warn you.

```python
set_seed(RANDOM_STATE)
point = NODERegressor(
    model_config=NODEConfig(num_layers=3, depth=5, layer_dim=128),
    preprocessing_config=PREPROC,
    trainer_config=TRAINER,
    random_state=RANDOM_STATE,
)
point.fit(X_train, y_train, **FIT_KWARGS)

resid = y_test - point.predict(X_test)
low, high = X_test["load"] < 0.5, X_test["load"] >= 0.5
print(f"residual std (low load):  {resid[low].std():.2f}")
print(f"residual std (high load): {resid[high].std():.2f}")
```

```{important}
A point regressor minimises average error and converges to the conditional mean.
It is silent about variance, so every prediction carries the same implicit
confidence even when the real uncertainty differs by an order of magnitude.
```

## Train an LSS Model

The `*LSS` variant predicts distribution parameters instead of a point. For the
normal family it emits two numbers per row, a location and a scale, and trains by
maximising the Gaussian log-likelihood, so the scale head learns the local noise
directly. The family is chosen at `fit()` time.

```python
set_seed(RANDOM_STATE)
lss = NODELSS(
    model_config=NODEConfig(num_layers=3, depth=5, layer_dim=128),
    preprocessing_config=PREPROC,
    trainer_config=TRAINER,
    random_state=RANDOM_STATE,
)
lss.fit(X_train, y_train, family="normal", **FIT_KWARGS)
```

```{tip}
Every DeepTab architecture has an LSS variant (`MLPLSS`, `FTTransformerLSS`,
`NODELSS`, and so on). Swapping the backbone is a one-line change; the
distribution machinery is shared.
```

## Predicting Distribution Parameters

`predict()` returns one row of parameters per sample. With `raw=False` (the
default) the inverse-link transforms are applied, so the values are ready to use.

```python
params = lss.predict(X_test)         # shape (n_samples, 2) for the normal family
print(params.shape)

loc = params[:, 0]
scale = params[:, 1]
```

```{important}
Distribution parameters are model outputs, not universal statistics. For DeepTab's
normal family the two columns are the location and a strictly positive scale (the
softplus-transformed second output is used directly as the Gaussian's standard
deviation in the likelihood). Other families return different quantities: a shape
and a rate for `"gamma"`, degrees of freedom plus location and scale for
`"studentt"`. Always confirm the convention for the family you train. Pass
`raw=True` to see the untransformed network outputs.
```

## Building Prediction Intervals

With a location and a scale per row, a central interval at any confidence level is
a direct quantile lookup. Because the scale varies by row, the intervals are
naturally wider where the model is less certain.

```python
def normal_interval(loc, scale, level=0.90):
    alpha = (1.0 - level) / 2.0
    lower = stats.norm.ppf(alpha, loc=loc, scale=scale)
    upper = stats.norm.ppf(1.0 - alpha, loc=loc, scale=scale)
    return lower, upper


lower, upper = normal_interval(loc, scale, level=0.90)
print(f"mean interval width (low load):  {(upper - lower)[low].mean():.2f}")
print(f"mean interval width (high load): {(upper - lower)[high].mean():.2f}")
```

The high-load intervals come out much wider than the low-load ones, exactly the
behaviour the point model could not produce.

## Calibration: Do the Intervals Mean What They Say?

A 90% interval is only useful if it actually contains the truth about 90% of the
time. Empirical coverage at several nominal levels is the standard check: each
realised coverage should land close to its nominal target.

```python
print(f"{'nominal':>8}  {'empirical':>9}")
for level in [0.50, 0.80, 0.90, 0.95]:
    lo, hi = normal_interval(loc, scale, level=level)
    covered = np.mean((y_test >= lo) & (y_test <= hi))
    print(f"{level:8.2f}  {covered:9.3f}")
```

```{tip}
If empirical coverage is consistently below nominal, the model is overconfident
(scales too small); above nominal means it is underconfident (scales too large).
Persistent miscalibration is a cue to train longer, adjust capacity, or try a
family whose tails match the data.
```

## Recovering Heteroscedasticity

The real payoff is that the predicted scale tracks the true `sigma(x)` we built
into the data. A point regressor has no parameter that could do this.

```python
corr = np.corrcoef(scale, sigma_test)[0, 1]
print(f"corr(predicted scale, true sigma): {corr:.3f}")

order = np.argsort(X_test["load"].to_numpy())
binned = pd.DataFrame({"load": X_test["load"].to_numpy()[order],
                       "pred_scale": scale[order],
                       "true_sigma": sigma_test[order]})
print(binned.groupby(pd.cut(binned["load"], 5), observed=True)[["pred_scale", "true_sigma"]].mean())
```

A high correlation and matching per-bin averages confirm the model learned where
it should be uncertain, not just an average error bar.

## Evaluate With Proper Scoring Rules

Calling `evaluate()` without a `metrics` argument returns the default scoring rules
for the fitted family. For `"normal"` these are CRPS (a proper scoring rule that
rewards both accuracy and well-calibrated sharpness) plus RMSE and MAE on the mean.

```python
print(lss.evaluate(X_test, y_test))
# {"crps": ..., "rmse": ..., "mae": ...}

print("NLL:", lss.score(X_test, y_test))   # negative log-likelihood, lower is better
```

```{note}
RMSE and accuracy alone cannot tell a confident-but-wrong model from a
well-calibrated one. CRPS and NLL evaluate the whole predicted distribution, which
is what you actually deploy in an uncertainty-aware system.
```

## Choosing a Distribution Family

The family encodes your assumptions about the target's support and tails. Match it
to the data, then let a proper scoring rule settle close calls. Here we add a few
heavy-tailed outliers and compare the thin-tailed normal against the heavy-tailed
Student's t, selecting by CRPS.

```python
contam = rng.random(len(y_train)) < 0.05
y_train_heavy = y_train.copy()
y_train_heavy[contam] += rng.standard_t(df=2, size=contam.sum()) * 25.0

scores = {}
for family in ["normal", "studentt"]:
    set_seed(RANDOM_STATE)
    m = NODELSS(
        model_config=NODEConfig(num_layers=3, depth=5, layer_dim=128),
        preprocessing_config=PREPROC, trainer_config=TRAINER, random_state=RANDOM_STATE,
    )
    m.fit(X_train, y_train_heavy, family=family, **FIT_KWARGS)
    scores[family] = m.evaluate(X_test, y_test)["crps"]

print(scores)   # lower CRPS wins
```

Match the family to the target support before tuning anything else:

```{tip}
Wrong support is a modeling error, not a tuning issue. Do not use a positive-only
family for targets that can go negative, or a count family for continuous targets.
```

| Target                   | Candidate family               |
| ------------------------ | ------------------------------ |
| Continuous unbounded     | `"normal"`, `"studentt"`       |
| Right-skewed positive    | `"lognormal"`, `"gamma"`       |
| Count data               | `"poisson"`, `"negativebinom"` |
| Zero-inflated counts     | `"zip"`                        |
| Proportions in `(0, 1)`  | `"beta"`                       |
| Insurance / pure premium | `"tweedie"`                    |

## Observability

Attach an `ObservabilityConfig` to record each run's hyperparameters, lifecycle
events, and final metrics in one self-contained directory. This is especially
useful here, where you compare families and calibration across several fits.

```python
obs = ObservabilityConfig(
    experiment_name="uncertainty_node_lss",
    structured_logging=True,
    log_to_file=True,
    verbosity=2,
    experiment_trackers=["tensorboard"],
)

set_seed(RANDOM_STATE)
tracked = NODELSS(
    model_config=NODEConfig(num_layers=3, depth=5, layer_dim=128),
    preprocessing_config=PREPROC,
    trainer_config=TRAINER,
    observability_config=obs,
    random_state=RANDOM_STATE,
)
tracked.fit(X_train, y_train, family="normal", **FIT_KWARGS)
```

```{note}
Structured logging needs `structlog` (`pip install 'deeptab[logs]'`) and the
TensorBoard tracker needs `tensorboard`. Drop `observability_config` to train
silently, or see the [Observability guide](../core_concepts/observability) for
MLflow, verbosity levels, and bringing your own logger.
```

## Save and Load

Persist the fitted estimator as a single artifact. The recommended extension is
`.deeptab`; the bundle stores the weights, fitted preprocessor, feature schema, and
the distribution family, so a reloaded model predicts identical parameters.

```python
lss.save("uncertainty_model.deeptab")

loaded = NODELSS.load("uncertainty_model.deeptab")
print(loaded.task_info_["family"])   # 'normal'
np.testing.assert_allclose(lss.predict(X_test), loaded.predict(X_test), atol=1e-5)
print("Reload parameters match")
```

## Production Inference with `InferenceModel`

For a service or batch job, load the artifact through `InferenceModel`. It exposes
a narrow, prediction-only surface and validates the incoming schema. For an LSS
model, `predict()` returns the distribution mean while `predict_params()` returns
the full parameter array you need for intervals.

```python
from deeptab import InferenceModel

infer = InferenceModel.from_path("uncertainty_model.deeptab")
print(infer.task)         # "distributional_regression"
print(infer.n_features)   # 4

X_clean = infer.validate_input(X_test)
params = infer.predict_params(X_clean)
loc, scale = params[:, 0], params[:, 1]
lower, upper = normal_interval(loc, scale, level=0.90)
```

`predict_proba()` is a classification-only method and raises on an LSS model, so
deployment code cannot misuse the estimator:

```python
infer.predict_proba(X_clean)
# TypeError: predict_proba() is only available for classification models,
# but this model's task is 'distributional_regression'.
```

See [Inference Model](../core_concepts/inference) for the full production API.

## Next Steps

- [Hyperparameter optimization](hpo): tune distributional models and pick a family with Bayesian search
- [Skewed-target regression](skewed_regression): point regression on a right-skewed target
- [Advanced training](advanced_training): schedulers, callbacks, and fine-grained control
- [Observability](../core_concepts/observability): lifecycle events, structured logging, and experiment tracking
- [Inference model](../core_concepts/inference): the deployment-safe prediction surface
- [Distribution API](../api/distributions/index): every supported family and its parameters

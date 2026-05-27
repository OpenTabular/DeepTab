# Training and Evaluation

This page explains how DeepTab trains models, what happens during `fit()`, and how to evaluate and monitor performance.

```{tip}
DeepTab uses **PyTorch Lightning** under the hood, providing automatic GPU support, early stopping, checkpointing, and progress bars—all without manual configuration.
```

## The training loop

When you call `fit()`, DeepTab executes a multi-epoch training loop powered by PyTorch Lightning:

```python
model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=100)
```

### What happens during fit()

```{important}
**The fit() pipeline:**
1. **Preprocessing** — Detect types, fit transformers, apply transforms
2. **Dataset creation** — Wrap in `TabularDataset`, create `DataLoader`
3. **Model initialization** — Build architecture, initialize weights
4. **Training epochs** — Forward pass → loss → backward → optimize
5. **Checkpointing** — Save best model, restore at end
```

## Fit parameters

The `fit()` method accepts several parameters:

```python
model.fit(
    X_train, y_train,           # Required: training data
    X_val=None, y_val=None,     # Optional: validation set
    X_embedding=None,           # Optional: pre-computed embeddings
    max_epochs=100,             # Training epochs
    family="normal",            # LSS only: distribution family
)
```

### X_train, y_train

Training features and labels.

- `X_train`: DataFrame or NumPy array, shape `(n_samples, n_features)`
- `y_train`: Array-like, shape `(n_samples,)` or `(n_samples, 1)`

### X_val, y_val

Optional validation set. If not provided, DeepTab creates one via train/val split:

```{note}
**Automatic validation split:**
- Uses 20% of training data by default (configurable via `TrainerConfig.val_split`)
- **Stratified** for classification (preserves class distribution)
- **Random** for regression
```

```python
# Explicit validation set
model.fit(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    max_epochs=100,
)
```

**Benefits of explicit validation:**

- More control over the split
- Can use time-based splits for time series
- Ensures consistent evaluation across experiments

### X_embedding

Pre-computed embeddings to concatenate with tabular features:

```python
# Text embeddings
text_embeds = sentence_model.encode(df["description"])

model.fit(
    X_train, y_train,
    X_embedding=text_embeds,
    max_epochs=50,
)
```

Must have shape `(n_samples, embedding_dim)`.

### max_epochs

Maximum number of training epochs:

```python
model.fit(X_train, y_train, max_epochs=100)
```

Training may stop earlier due to early stopping (see below).

### family (LSS only)

Distribution family for LSS models:

```python
lss_model = MambularLSS()
lss_model.fit(X_train, y_train, family="normal", max_epochs=50)
```

See [Distributional Regression](distributional_regression) for available families.

## Early stopping

```{important}
**Early stopping prevents overfitting** by monitoring validation loss and stopping training when it plateaus. The best model (lowest validation loss) is automatically restored.
```

Early stopping prevents overfitting by monitoring validation loss and stopping when it stops improving.

### Configuration

```python
from deeptab.configs import TrainerConfig

cfg = TrainerConfig(
    patience=15,  # Stop if no improvement for 15 epochs
    min_delta=1e-4,  # Minimum change to count as improvement
)

model = MambularClassifier(trainer_config=cfg)
model.fit(X_train, y_train, max_epochs=100)
```

### How it works

1. Track validation loss after each epoch
2. If loss improves by at least `min_delta`, reset patience counter
3. If loss doesn't improve, increment counter
4. If counter reaches `patience`, stop training
5. Restore weights from best epoch

### Example

```
Epoch 1: val_loss = 0.50  ← Best so far
Epoch 2: val_loss = 0.45  ← Best so far
Epoch 3: val_loss = 0.46  (No improvement, patience = 1)
Epoch 4: val_loss = 0.43  ← Best so far (patience reset)
...
Epoch 20: val_loss = 0.44 (No improvement, patience = 15) → Stop!
Restore weights from Epoch 4
```

## Learning rate scheduling

Adjust learning rate during training for better convergence.

### Reduce on plateau

Reduce LR when validation loss plateaus:

```python
cfg = TrainerConfig(
    lr=1e-3,
    lr_scheduler="reduce_on_plateau",
    lr_scheduler_patience=5,      # Reduce after 5 epochs without improvement
    lr_scheduler_factor=0.5,      # Multiply LR by 0.5
)

model = MambularClassifier(trainer_config=cfg)
```

### Cosine annealing

Smoothly decrease LR following a cosine curve:

```python
cfg = TrainerConfig(
    lr=1e-3,
    lr_scheduler="cosine",
    lr_scheduler_t_max=50,  # Period of cosine annealing
)
```

### Step decay

Decrease LR at fixed intervals:

```python
cfg = TrainerConfig(
    lr=1e-3,
    lr_scheduler="step",
    lr_scheduler_step_size=20,  # Reduce every 20 epochs
    lr_scheduler_gamma=0.1,     # Multiply by 0.1
)
```

### No scheduling

Default (no scheduler):

```python
cfg = TrainerConfig(
    lr=1e-3,
    lr_scheduler=None,  # Constant learning rate
)
```

## Gradient clipping

Prevents exploding gradients by clipping gradient norms.

```python
cfg = TrainerConfig(
    gradient_clip_val=1.0,  # Clip to max norm of 1.0
)
```

**Enabled by default with value 1.0**. Disable with `None`:

```python
cfg = TrainerConfig(gradient_clip_val=None)  # No clipping
```

## Optimization

### Optimizer selection

```python
cfg = TrainerConfig(
    optimizer="adam",     # Options: "adam", "adamw", "sgd"
    lr=1e-3,
    weight_decay=1e-4,    # L2 regularization (for adamw/sgd)
)
```

| Optimizer | Description                      | When to use               |
| --------- | -------------------------------- | ------------------------- |
| `"adam"`  | Adaptive moment estimation       | General purpose (default) |
| `"adamw"` | Adam with decoupled weight decay | When using weight decay   |
| `"sgd"`   | Stochastic gradient descent      | Simple baseline           |

### Learning rate

```python
cfg = TrainerConfig(lr=1e-3)  # Default: 1e-4
```

**Guidelines:**

- Start with 1e-4 (default)
- Increase to 1e-3 or 5e-4 for faster convergence (but risk instability)
- Decrease to 1e-5 or 1e-6 if training is unstable

### Weight decay

L2 regularization to prevent overfitting:

```python
cfg = TrainerConfig(
    optimizer="adamw",
    weight_decay=1e-4,  # Default: 0.0
)
```

Higher weight decay → more regularization.

## Batch size

```python
cfg = TrainerConfig(batch_size=256)  # Default: 128
```

**Effects:**

- **Larger batches** → faster training (GPU utilization), less noisy gradients, more memory
- **Smaller batches** → slower training, noisier gradients (can help escape local minima), less memory

**Guidelines:**

- Use largest batch that fits in memory
- Try 128, 256, 512 for most datasets
- Reduce if you get OOM errors

## Monitoring progress

### Progress bar

Enabled by default:

```
Epoch 10/100: 100%|██████████| 50/50 [00:02<00:00, 20.5batch/s, loss=0.42, val_loss=0.38]
```

Disable:

```python
cfg = TrainerConfig(progress_bar=False)
```

### Verbose logging

```python
cfg = TrainerConfig(verbose=True)
```

Prints detailed metrics each epoch:

```
Epoch 1: train_loss=0.50, val_loss=0.45, train_acc=0.75, val_acc=0.78
Epoch 2: train_loss=0.45, val_loss=0.42, train_acc=0.78, val_acc=0.80
...
```

### Custom logging

Use Lightning callbacks for advanced logging (TensorBoard, Weights & Biases, etc.):

```python
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# This requires using TabularDataModule directly (advanced usage)
# See Lightning docs for details
```

## Evaluation

After training, evaluate on test data:

```python
model.fit(X_train, y_train, max_epochs=50)
metrics = model.evaluate(X_test, y_test)
```

### Output format

Returns a dictionary of metrics:

```python
# Classification
metrics = model.evaluate(X_test, y_test)
# {'accuracy': 0.85, 'loss': 0.42, ...}

# Regression
metrics = model.evaluate(X_test, y_test)
# {'rmse': 12.34, 'mae': 8.56, 'loss': 152.3}

# LSS
metrics = lss_model.evaluate(X_test, y_test)
# {'loss': -234.5}  # Negative log-likelihood
```

### Access metrics

```python
print(f"Test accuracy: {metrics['accuracy']:.3f}")
print(f"Test loss: {metrics['loss']:.3f}")
```

### Custom metrics

Specify in `TrainerConfig`:

```python
from torchmetrics import F1Score, Precision, Recall

cfg = TrainerConfig(
    metrics=[
        F1Score(task="multiclass", num_classes=3),
        Precision(task="multiclass", num_classes=3, average="macro"),
        Recall(task="multiclass", num_classes=3, average="macro"),
    ]
)

model = MambularClassifier(trainer_config=cfg)
model.fit(X_train, y_train, max_epochs=50)

metrics = model.evaluate(X_test, y_test)
# Includes all specified metrics
```

### score() method

For scikit-learn compatibility:

```python
score = model.score(X_test, y_test)
# Classification → accuracy
# Regression → R² score
```

## Training on GPU

DeepTab automatically uses GPU if available:

```python
import torch
print(torch.cuda.is_available())  # True → will use GPU

model = MambularClassifier()
model.fit(X_train, y_train, max_epochs=50)  # Runs on GPU automatically
```

### Force CPU

```python
cfg = TrainerConfig(device="cpu")
model = MambularClassifier(trainer_config=cfg)
```

### Specific GPU

```python
cfg = TrainerConfig(device="cuda:1")  # Use GPU 1
```

Or set environment variable:

```bash
export CUDA_VISIBLE_DEVICES=1
python train_script.py
```

### Multi-GPU

For multi-GPU training, use Lightning's distributed strategies directly with `TabularDataModule` (advanced usage).

## Mixed precision training

Train with float16 for faster training and less memory:

```python
cfg = TrainerConfig(precision="16")  # Default: "32"
model = MambularClassifier(trainer_config=cfg)
```

**Caution:** May cause numerical instability for some models. If you see NaN losses, switch back to float32.

## Deterministic training

For reproducible results:

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

cfg = TrainerConfig(deterministic=True)
model = MambularClassifier(trainer_config=cfg)
model.fit(X_train, y_train, max_epochs=50)
```

**Warning:** Deterministic training is slower due to disabling performance optimizations.

## Saving and loading models

### Save after training

```python
model.fit(X_train, y_train, max_epochs=50)
model.save("my_model.pkl")
```

### Load later

```python
from deeptab.models import MambularClassifier

loaded_model = MambularClassifier.load("my_model.pkl")
predictions = loaded_model.predict(X_test)
```

The saved file includes:

- Model architecture and weights
- Preprocessing state (fitted transformers)
- Config objects
- Training history

## Inspecting training history

Access training history after fitting:

```python
model.fit(X_train, y_train, max_epochs=50)

# Access internal Lightning trainer
trainer = model.model.trainer

# Training history
print(trainer.logged_metrics)
```

This is advanced usage. For most cases, `evaluate()` is sufficient.

## Troubleshooting

### Training is slow

- Use GPU if available
- Increase batch size
- Reduce model complexity (smaller d_model, fewer layers)
- Use multiple data loading workers: `TrainerConfig(num_workers=4)`

### Loss is NaN

- Reduce learning rate
- Enable gradient clipping (default)
- Check for NaN/Inf in data
- Try different initialization

### Overfitting (train good, val poor)

- Increase dropout: `ModelConfig(dropout=0.3)`
- Add weight decay: `TrainerConfig(weight_decay=1e-4)`
- Use early stopping (default)
- Get more data or augment
- Reduce model complexity

### Underfitting (both train and val poor)

- Increase model capacity: `ModelConfig(d_model=256, n_layers=8)`
- Train longer: `max_epochs=200`
- Reduce regularization: lower dropout, no weight decay
- Check feature engineering (preprocessing)

### Training is unstable (loss jumps)

- Reduce learning rate
- Increase gradient clipping value
- Use smaller batch size
- Check for data quality issues

### GPU out of memory

- Reduce batch size: `TrainerConfig(batch_size=64)`
- Reduce model size
- Use mixed precision: `TrainerConfig(precision="16")`
- Clear GPU cache between experiments: `torch.cuda.empty_cache()`

## Best practices

1. **Start with defaults** — Only tune if necessary
2. **Use validation set** — Explicit is better than automatic split
3. **Monitor early stopping** — Prevents overfitting
4. **Save best models** — Automatic with early stopping
5. **Log experiments** — Track metrics across runs
6. **Use GPU** — Significant speedup for larger datasets
7. **Set random seed** — For reproducibility
8. **Evaluate on holdout** — Never use test set for model selection

## Common training recipes

### Quick experimentation

```python
cfg = TrainerConfig(
    max_epochs=20,
    patience=5,
    batch_size=512,
)
```

### Production training

```python
cfg = TrainerConfig(
    max_epochs=200,
    patience=20,
    lr=5e-4,
    batch_size=256,
    gradient_clip_val=1.0,
    lr_scheduler="reduce_on_plateau",
)
```

### Overfit check

Intentionally overfit to verify model can learn:

```python
# Train on small subset
X_small = X_train[:100]
y_small = y_train[:100]

cfg = TrainerConfig(
    max_epochs=500,
    patience=50,  # High patience
    dropout=0.0,   # No regularization
)

model = MambularClassifier(
    model_config=MambularConfig(dropout=0.0),
    trainer_config=cfg,
)
model.fit(X_small, y_small)

# Should achieve very low training loss
```

## Next steps

- **[sklearn API](sklearn_api)** — Understand the fit/predict interface
- **[Config System](config_system)** — Full TrainerConfig reference
- **[Classification](classification)** — Classification-specific training
- **[Regression](regression)** — Regression-specific training

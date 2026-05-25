from dataclasses import dataclass, field

from sklearn.base import BaseEstimator


@dataclass
class TrainerConfig(BaseEstimator):
    """Configuration for training loop, optimizer, and runtime execution.

    These settings are entirely separate from model architecture.  They control
    *how* a model is trained and executed, not *what* the model is.

    Parameters
    ----------
    max_epochs : int, default=100
        Maximum number of training epochs.
    batch_size : int, default=128
        Number of samples per gradient update.
    val_size : float, default=0.2
        Fraction of the training data held out for validation when no explicit
        validation set is provided.
    shuffle : bool, default=True
        Whether to shuffle training data before each epoch.
    patience : int, default=15
        Number of epochs with no improvement on ``monitor`` before early stopping
        is triggered.
    monitor : str, default="val_loss"
        Metric name to monitor for early stopping and checkpoint selection.
    mode : str, default="min"
        Whether the monitored metric should be minimised (``"min"``) or
        maximised (``"max"``).
    lr : float, default=1e-4
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement before the learning rate is reduced
        by ``lr_factor``.
    lr_factor : float, default=0.1
        Multiplicative factor applied to the learning rate when patience is
        exceeded.
    weight_decay : float, default=1e-6
        L2 regularisation coefficient (weight decay) for the optimizer.
    optimizer_type : str, default="Adam"
        Optimizer class name.  Must be a valid ``torch.optim`` class name or a
        name registered in the project's optimizer registry.
    checkpoint_path : str, default="model_checkpoints"
        Directory where PyTorch Lightning model checkpoints are saved.
    """

    max_epochs: int = 100
    batch_size: int = 128
    val_size: float = 0.2
    shuffle: bool = True
    patience: int = 15
    monitor: str = "val_loss"
    mode: str = "min"
    lr: float = 1e-4
    lr_patience: int = 10
    lr_factor: float = 0.1
    weight_decay: float = 1e-6
    optimizer_type: str = "Adam"
    checkpoint_path: str = "model_checkpoints"

import numpy as np
import torch


def check_numpy(x):
    """Makes sure x is a numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if not isinstance(x, np.ndarray):
        raise TypeError("Expected input to be a numpy array")
    return x

import numpy as np


def poisson_deviance(y_true, y_pred):
    # Ensure no zero to avoid log(0)
    y_pred = np.clip(y_pred, 1e-9, None)
    return 2 * np.sum(y_true * np.log(y_true / y_pred) - (y_true - y_pred))


def gamma_deviance(y_true, y_pred):
    # Avoid division by zero and log(0)
    y_pred = np.clip(y_pred, 1e-9, None)
    y_true = np.clip(y_true, 1e-9, None)
    return 2 * np.sum(np.log(y_true / y_pred) + (y_true - y_pred) / y_pred)


def beta_brier_score(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)


def dirichlet_error(y_true, y_pred):
    # Simple sum of squared differences as an example
    return np.mean(np.sum((y_pred - y_true) ** 2, axis=1))


def student_t_loss(y_true, y_pred, df=2):
    # Assuming y_pred includes both location and scale
    mu = y_pred[:, 0]
    scale = np.clip(y_pred[:, 1], 1e-9, None)  # Avoid zero scale
    return np.mean((df + 1) * np.log(1 + (y_true - mu) ** 2 / (df * scale)) / scale)


def negative_binomial_deviance(y_true, y_pred, alpha):
    # Here alpha is the overdispersion parameter
    mu = y_pred
    return 2 * np.sum(y_true * np.log(y_true / mu + 1e-9) + (y_true + alpha) * np.log((mu + alpha) / (y_true + alpha)))


def inverse_gamma_loss(y_true, y_pred):
    # Assuming y_pred includes both shape and scale
    shape = y_pred[:, 0]
    scale = np.clip(y_pred[:, 1], 1e-9, None)  # Avoid zero scale
    return np.mean((shape + 1) * np.log(y_true / scale) + np.log(scale**shape / y_true))

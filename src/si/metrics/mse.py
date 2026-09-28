import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the mean squared error (MSE) between the real and predicted values.

    Parameters
    ----------
    y_true: np.ndarray
        The real values of the label
    y_pred: np.ndarray
        The predicted values of the label

    Returns
    -------
    mse: float
        The mean squared error between y_true and y_pred
    """
    return np.sum((y_true - y_pred) ** 2) / len(y_true)

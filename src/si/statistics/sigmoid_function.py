import numpy as np


def sigmoid_function(x: np.ndarray) -> np.ndarray:
    """
    Computes the sigmoid function of x, mapping any real value to the ]0, 1[ interval.

        sigmoid(x) = 1 / (1 + e^(-x))

    Parameters
    ----------
    x: np.ndarray
        The input values

    Returns
    -------
    sigmoid: np.ndarray
        The probability of the values being equal to 1
    """
    return 1 / (1 + np.exp(-x))

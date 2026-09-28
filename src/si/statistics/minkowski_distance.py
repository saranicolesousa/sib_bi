import numpy as np


def minkowski_distance(x: np.ndarray, y: np.ndarray, p: float = 2) -> np.ndarray:
    """
    Computes the Minkowski distance of order p between a single sample x and
    every sample in y.

        distance_y1 = (|x1 - y11|^p + |x2 - y12|^p + ... + |xn - y1n|^p) ^ (1/p)

    p = 1 is equivalent to the Manhattan distance and p = 2 to the Euclidean distance.

    Parameters
    ----------
    x: np.ndarray (n_features,)
        A single sample
    y: np.ndarray (n_samples, n_features)
        Multiple samples
    p: float
        The order of the distance. Must be >= 1

    Returns
    -------
    distances: np.ndarray (n_samples,)
        The Minkowski distance between x and every sample in y
    """
    if p < 1:
        raise ValueError("p must be greater than or equal to 1")
    return (np.abs(x - y) ** p).sum(axis=1) ** (1 / p)

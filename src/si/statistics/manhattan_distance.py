import numpy as np


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the Manhattan (L1) distance between a single sample x and
    every sample in y.

        distance_y1 = |x1 - y11| + |x2 - y12| + ... + |xn - y1n|

    Parameters
    ----------
    x: np.ndarray (n_features,)
        A single sample
    y: np.ndarray (n_samples, n_features)
        Multiple samples

    Returns
    -------
    distances: np.ndarray (n_samples,)
        The Manhattan distance between x and every sample in y
    """
    return np.abs(x - y).sum(axis=1)

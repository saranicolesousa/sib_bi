import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    return np.sum(y_true == y_pred) / len(y_true)
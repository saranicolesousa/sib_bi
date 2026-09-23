import numpy as np
from si.data.dataset import Dataset

def read_data_file(filename: str, sep: str = ',', label: bool = False):
    raw_data = np.genfromtxt(filename, delimiter=sep)

    if label:
        X = raw_data[:, :-1]
        y = raw_data[:, -1]
    else:
        X = raw_data[:, :]
        y = None
    
    return Dataset(X=X, y=y)

def write_data_file(filename: str, dataset: Dataset, sep: str = ',', label: bool = False) -> None:

    if label:
        data = np.hstack((dataset.X, dataset.y.reshape(-1, 1)))
    else:
        data = dataset.X

    np.savetxt(filename, data, delimiter=sep)
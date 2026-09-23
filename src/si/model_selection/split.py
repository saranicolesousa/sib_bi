from typing import Tuple
import numpy as np
from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    np.random.seed(random_state)

    n_samples = dataset.shape()[0]

    n_test = int(n_samples * test_size)

    permutations = np.random.permutation(n_samples)

    test_idxs = permutations[:n_test]

    train_idxs = permutations[n_test:]

    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)

    return train, test

def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:

    np.random.seed(random_state)
    labels, counts = np.unique(dataset.y, return_counts=True)

    train_idxs = []
    test_idxs = []

    for label, count in zip(labels, counts):

        n_test_label = int(count * test_size)
        label_idxs = np.where(dataset.y == label)[0]
        permuted_idxs = np.random.permutation(label_idxs)
        test_idxs.extend(permuted_idxs[:n_test_label])
        train_idxs.extend(permuted_idxs[n_test_label:])

    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)

    return train, test
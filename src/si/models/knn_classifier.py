from typing import Callable, Union

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance


class KNNClassifier(Model):

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance

        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNClassifier':
        
        self.dataset = dataset
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        predictions = []

        for sample in dataset.X:
            # distance between the sample and every sample in the training dataset
            distances = self.distance(sample, self.dataset.X)

            # indexes of the k most similar samples
            k_nearest_neighbors = np.argsort(distances)[:self.k]

            # classes of those k samples
            k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]

            # most common class among them
            labels, counts = np.unique(k_nearest_neighbors_labels, return_counts=True)
            predictions.append(labels[np.argmax(counts)])

        return np.array(predictions)

    
    def _score(self, dataset: Dataset) -> float:
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)

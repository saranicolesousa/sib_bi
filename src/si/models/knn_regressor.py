from typing import Callable

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance

        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        self.dataset = dataset
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        predictions = []

        for sample in dataset.X:
            distances = self.distance(sample, self.dataset.X)
            k_nearest_neighbors = np.argsort(distances)[:self.k]
            k_nearest_neighbors_values = self.dataset.y[k_nearest_neighbors]
            predictions.append(np.mean(k_nearest_neighbors_values))

        return np.array(predictions)
    
    def _score(self, dataset: Dataset) -> float:
        predictions = self.predict(dataset)
        return rmse(dataset.y, predictions)
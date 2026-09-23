from typing import Callable

import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectKBest(Transformer):

    def __init__(self, score_func: Callable = f_classification, k: int = 10, **kwargs):
    
        super().__init__(**kwargs)

        if k < 1:
            raise ValueError("k must be greater than 0")

        self.score_func = score_func
        self.k = k

        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectKBest':
        
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        
        idxs = np.argsort(self.F)[-self.k:]
        new_X = dataset.X[:, idxs]
        new_features = [str(f) for f in np.array(dataset.features)[idxs]]
        return Dataset(X=new_X, y=dataset.y, features=new_features, label=dataset.label)

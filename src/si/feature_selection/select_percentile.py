from typing import Callable

import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile(Transformer):

    def __init__(self, score_func: Callable = f_classification, percentile: float = 50, **kwargs):
       
        super().__init__(**kwargs)

        if not 0 <= percentile <= 100:
            raise ValueError("Percentile must be between 0 and 100")

        self.score_func = score_func
        self.percentile = percentile

        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        
        n_features = len(self.F)
        n_select = int(n_features * self.percentile / 100)
        
        threshold = np.percentile(self.F, 100 - self.percentile)
        
        mask = self.F > threshold

        n_missing = n_select - int(np.sum(mask))
        if n_missing > 0:
            tied_idxs = np.where(self.F == threshold)[0]
            mask[tied_idxs[:n_missing]] = True

        new_X = dataset.X[:, mask]
        new_features = [str(f) for f in np.array(dataset.features)[mask]]
        return Dataset(X=new_X, y=dataset.y, features=new_features, label=dataset.label)

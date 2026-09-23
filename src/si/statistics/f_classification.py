from typing import Tuple, Union

import numpy as np
from scipy import stats

from si.data.dataset import Dataset


def f_classification(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    classes = dataset.get_classes()
    groups = [dataset.X[dataset.y == c] for c in classes]
    F, p = stats.f_oneway(*groups)
    return F, p

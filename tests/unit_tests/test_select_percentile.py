import os
import unittest

import numpy as np

from datasets import DATASETS_PATH

from si.data.dataset import Dataset
from si.feature_selection.select_percentile import SelectPercentile
from si.io.csv_file import read_csv
from si.statistics.f_classification import f_classification


class TestSelectPercentile(unittest.TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, sep=",", features=True, label=True)

    def test_fit(self):
        selector = SelectPercentile(score_func=f_classification, percentile=50)
        selector.fit(self.dataset)

        self.assertEqual(len(selector.F), self.dataset.shape()[1])
        self.assertEqual(len(selector.p), self.dataset.shape()[1])

    def test_transform(self):
        selector = SelectPercentile(score_func=f_classification, percentile=50)
        selector.fit(self.dataset)
        new_dataset = selector.transform(self.dataset)

        # 50% of 4 features
        self.assertEqual((150, 2), new_dataset.shape())
        self.assertEqual(['petal_length', 'petal_width'], list(new_dataset.features))
        self.assertEqual(self.dataset.y.tolist(), new_dataset.y.tolist())

    def test_transform_all_features(self):
        selector = SelectPercentile(score_func=f_classification, percentile=100)
        new_dataset = selector.fit_transform(self.dataset)

        self.assertEqual(self.dataset.shape(), new_dataset.shape())

    def test_transform_handles_ties(self):
        # example from the slides: 10 features, percentile=40 -> 4 features
        F = np.array([1.2, 3.4, 2.1, 5.6, 4.3, 5.6, 7.8, 6.5, 5.6, 3.2])
        dataset = Dataset(X=np.tile(F, (3, 1)), y=np.array([0, 1, 0]))

        selector = SelectPercentile(percentile=40)
        selector.F = F
        selector.p = np.zeros(10)
        selector._is_fitted = True

        new_dataset = selector.transform(dataset)

        self.assertEqual(4, new_dataset.shape()[1])
        self.assertEqual([5.6, 5.6, 7.8, 6.5], new_dataset.X[0].tolist())

    def test_invalid_percentile(self):
        with self.assertRaises(ValueError):
            SelectPercentile(percentile=150)

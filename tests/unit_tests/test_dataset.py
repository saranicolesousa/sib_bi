import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())

    def test_dropna(self):
        X = np.array([[1, 2, 3], [np.nan, 5, 6], [7, 8, 9]])
        y = np.array([1, 2, 3])
        dataset = Dataset(X, y, features=['a', 'b', 'c'], label='y')

        dataset.dropna()

        self.assertEqual((2, 3), dataset.shape())
        self.assertEqual(2, len(dataset.y))
        self.assertEqual([1, 3], dataset.y.tolist())
        self.assertFalse(np.isnan(dataset.X).any())

    def test_fillna_value(self):
        X = np.array([[1., 2., 3.], [np.nan, 5., 6.], [7., 8., 9.]])
        y = np.array([1, 2, 3])
        dataset = Dataset(X, y, features=['a', 'b', 'c'], label='y')

        dataset.fillna(0.0)

        self.assertEqual((3, 3), dataset.shape())
        self.assertFalse(np.isnan(dataset.X).any())
        self.assertEqual(0.0, dataset.X[1, 0])

    def test_fillna_mean(self):
        X = np.array([[1., 2., 3.], [np.nan, 5., 6.], [7., 8., 9.]])
        y = np.array([1, 2, 3])
        dataset = Dataset(X, y, features=['a', 'b', 'c'], label='y')

        dataset.fillna('mean')

        self.assertFalse(np.isnan(dataset.X).any())
        self.assertEqual(4.0, dataset.X[1, 0])  # mean of 1 and 7

    def test_fillna_median(self):
        X = np.array([[1., 2., 3.], [np.nan, 5., 6.], [7., 8., 9.]])
        y = np.array([1, 2, 3])
        dataset = Dataset(X, y, features=['a', 'b', 'c'], label='y')

        dataset.fillna('median')

        self.assertFalse(np.isnan(dataset.X).any())
        self.assertEqual(4.0, dataset.X[1, 0])  # median of 1 and 7

    def test_remove_by_index(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([1, 2, 3])
        dataset = Dataset(X, y, features=['a', 'b', 'c'], label='y')

        dataset.remove_by_index(1)

        self.assertEqual((2, 3), dataset.shape())
        self.assertEqual([1, 3], dataset.y.tolist())
        self.assertEqual([7, 8, 9], dataset.X[1].tolist())

    def test_remove_by_index_out_of_bounds(self):
        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])
        dataset = Dataset(X, y, features=['a', 'b', 'c'], label='y')

        with self.assertRaises(IndexError):
            dataset.remove_by_index(5)

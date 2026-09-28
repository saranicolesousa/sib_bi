import unittest

import numpy as np

from si.metrics.mse import mse


class TestMSE(unittest.TestCase):

    def test_mse_perfect_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0])
        self.assertEqual(0.0, mse(y_true, y_true))

    def test_mse_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 6.0])
        # (0 + 0 + 0 + 4) / 4
        self.assertEqual(1.0, mse(y_true, y_pred))

    def test_mse_is_symmetric(self):
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        self.assertAlmostEqual(mse(y_true, y_pred), mse(y_pred, y_true))

    def test_mse_matches_manual_formula(self):
        rng = np.random.default_rng(42)
        y_true = rng.normal(size=100)
        y_pred = rng.normal(size=100)
        expected = np.mean((y_true - y_pred) ** 2)
        self.assertAlmostEqual(expected, mse(y_true, y_pred))

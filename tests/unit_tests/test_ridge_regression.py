import os
import unittest

import numpy as np

from datasets import DATASETS_PATH

from si.data.dataset import Dataset
from si.io.csv_file import read_csv
from si.metrics.mse import mse
from si.model_selection.split import train_test_split
from si.models.linear_regression import RidgeRegression
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares


class TestRidgeRegression(unittest.TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, sep=",", features=True, label=True)
        self.train, self.test = train_test_split(self.dataset, test_size=0.2, random_state=42)

    def test_fit_estimates_the_parameters(self):
        model = RidgeRegression(l2_penalty=1.0, alpha=0.001, max_iter=1000, scale=True)
        model.fit(self.train)

        n_features = self.train.shape()[1]
        self.assertEqual(n_features, len(model.theta))
        self.assertEqual(n_features, len(model.mean))
        self.assertEqual(n_features, len(model.std))
        self.assertNotEqual(0.0, model.theta_zero)
        self.assertTrue(np.allclose(self.train.X.mean(axis=0), model.mean))

    def test_fit_without_scaling_does_not_estimate_mean_and_std(self):
        model = RidgeRegression(alpha=0.0001, max_iter=100, scale=False)
        model.fit(self.train)

        self.assertIsNone(model.mean)
        self.assertIsNone(model.std)

    def test_cost_decreases_along_the_iterations(self):
        model = RidgeRegression(l2_penalty=1.0, alpha=0.001, max_iter=500, patience=500, scale=True)
        model.fit(self.train)

        costs = list(model.cost_history.values())
        self.assertEqual(500, len(costs))
        self.assertLess(costs[-1], costs[0])
        # monotonically decreasing for a learning rate this small
        self.assertTrue(all(later <= earlier for earlier, later in zip(costs, costs[1:])))

    def test_patience_stops_before_max_iter(self):
        model = RidgeRegression(l2_penalty=1.0, alpha=0.01, max_iter=100000, patience=10, scale=True)
        model.fit(self.train)

        self.assertLess(len(model.cost_history), 100000)

    def test_predict_output_shape(self):
        model = RidgeRegression(alpha=0.001, max_iter=500).fit(self.train)
        predictions = model.predict(self.test)

        self.assertEqual(self.test.shape()[0], len(predictions))

    def test_score_equals_mse_of_the_predictions(self):
        model = RidgeRegression(alpha=0.001, max_iter=500).fit(self.train)
        predictions = model.predict(self.test)

        self.assertAlmostEqual(mse(self.test.y, predictions), model.score(self.test))

    def test_gradient_descent_converges_to_the_least_squares_solution(self):
        gd = RidgeRegression(l2_penalty=1.0, alpha=0.01, max_iter=100000, patience=1000, scale=True)
        gd.fit(self.train)
        ls = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True).fit(self.train)

        self.assertTrue(np.allclose(ls.theta, gd.theta, atol=0.5))
        self.assertAlmostEqual(ls.theta_zero, gd.theta_zero, places=3)


class TestRidgeRegressionLeastSquares(unittest.TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, sep=",", features=True, label=True)
        self.train, self.test = train_test_split(self.dataset, test_size=0.2, random_state=42)

    def test_fit_estimates_the_parameters(self):
        model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True).fit(self.train)

        n_features = self.train.shape()[1]
        self.assertEqual(n_features, len(model.theta))
        self.assertEqual(n_features, len(model.mean))
        self.assertEqual(n_features, len(model.std))

    def test_fit_matches_the_closed_form_solution(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 9.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        dataset = Dataset(X=X, y=y)

        model = RidgeRegressionLeastSquares(l2_penalty=2.0, scale=False).fit(dataset)

        X_intercept = np.c_[np.ones(4), X]
        penalty = 2.0 * np.eye(3)
        penalty[0, 0] = 0
        expected = np.linalg.inv(X_intercept.T.dot(X_intercept) + penalty).dot(X_intercept.T).dot(y)

        self.assertTrue(np.allclose(expected[1:], model.theta))
        self.assertAlmostEqual(expected[0], model.theta_zero)

    def test_intercept_is_not_penalized(self):
        # with a centred X, theta_zero must equal the mean of y for any l2_penalty
        X = np.array([[-1.0], [0.0], [1.0]])
        y = np.array([2.0, 4.0, 6.0])
        dataset = Dataset(X=X, y=y)

        for l2_penalty in (0.0, 1.0, 100.0):
            model = RidgeRegressionLeastSquares(l2_penalty=l2_penalty, scale=False).fit(dataset)
            self.assertAlmostEqual(np.mean(y), model.theta_zero)

    def test_higher_penalty_shrinks_the_coefficients(self):
        weak = RidgeRegressionLeastSquares(l2_penalty=0.1, scale=True).fit(self.train)
        strong = RidgeRegressionLeastSquares(l2_penalty=1000.0, scale=True).fit(self.train)

        self.assertLess(np.sum(np.abs(strong.theta)), np.sum(np.abs(weak.theta)))

    def test_score_equals_mse_of_the_predictions(self):
        model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True).fit(self.train)
        predictions = model.predict(self.test)

        self.assertAlmostEqual(mse(self.test.y, predictions), model.score(self.test))

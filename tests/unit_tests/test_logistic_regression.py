import os
import unittest

import numpy as np

from datasets import DATASETS_PATH

from si.io.csv_file import read_csv
from si.model_selection.split import stratified_train_test_split
from si.models.logistic_regression import LogisticRegression
from si.statistics.sigmoid_function import sigmoid_function


class TestSigmoidFunction(unittest.TestCase):

    def test_sigmoid_known_values(self):
        self.assertAlmostEqual(0.5, sigmoid_function(np.array([0.0]))[0])
        self.assertAlmostEqual(1.0, sigmoid_function(np.array([100.0]))[0])
        self.assertAlmostEqual(0.0, sigmoid_function(np.array([-100.0]))[0])

    def test_sigmoid_output_is_a_probability(self):
        values = sigmoid_function(np.linspace(-10, 10, 50))
        self.assertTrue(np.all((values > 0) & (values < 1)))


class TestLogisticRegression(unittest.TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_csv(filename=self.csv_file, sep=",", features=False, label=True)
        self.train, self.test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=42)

    def test_fit_estimates_the_parameters(self):
        model = LogisticRegression(alpha=0.1, max_iter=500).fit(self.train)

        self.assertEqual(self.train.shape()[1], len(model.theta))
        self.assertIsNotNone(model.theta_zero)

    def test_predict_returns_only_zeros_and_ones(self):
        model = LogisticRegression(alpha=0.1, max_iter=500).fit(self.train)
        predictions = model.predict(self.test)

        self.assertTrue(set(np.unique(predictions)).issubset({0, 1}))

    def test_predict_proba_is_bounded(self):
        model = LogisticRegression(alpha=0.1, max_iter=500).fit(self.train)
        probabilities = model.predict_proba(self.test)

        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

    def test_model_learns(self):
        model = LogisticRegression(alpha=0.1, max_iter=2000, patience=20).fit(self.train)

        self.assertGreater(model.score(self.test), 0.9)

    def test_cost_decreases(self):
        model = LogisticRegression(alpha=0.1, max_iter=200, patience=200).fit(self.train)
        costs = list(model.cost_history.values())

        self.assertLess(costs[-1], costs[0])

    def test_momentum_speeds_up_convergence(self):
        without = LogisticRegression(alpha=0.1, max_iter=5000, patience=20, momentum=0.0).fit(self.train)
        with_momentum = LogisticRegression(alpha=0.1, max_iter=5000, patience=20, momentum=0.9).fit(self.train)

        self.assertLess(len(with_momentum.cost_history), len(without.cost_history))

    def test_invalid_momentum(self):
        with self.assertRaises(ValueError):
            LogisticRegression(momentum=1.0)
        with self.assertRaises(ValueError):
            LogisticRegression(momentum=-0.1)

    def test_l1_penalty_shrinks_the_coefficients(self):
        without = LogisticRegression(l2_penalty=0.0, l1_penalty=0.0, alpha=0.1,
                                     max_iter=1000, patience=1000).fit(self.train)
        with_l1 = LogisticRegression(l2_penalty=0.0, l1_penalty=200.0, alpha=0.1,
                                     max_iter=1000, patience=1000).fit(self.train)

        self.assertLess(np.sum(np.abs(with_l1.theta)), np.sum(np.abs(without.theta)))

    def test_l1_penalty_sets_coefficients_to_exactly_zero(self):
        model = LogisticRegression(l2_penalty=0.0, l1_penalty=200.0, alpha=0.1,
                                   max_iter=3000, patience=50).fit(self.train)

        self.assertGreater(np.sum(model.theta == 0), 0)

    def test_l1_shrinkage_is_monotonic(self):
        norms = []
        for l1_penalty in (0.0, 50.0, 100.0, 200.0, 500.0):
            model = LogisticRegression(l2_penalty=0.0, l1_penalty=l1_penalty, alpha=0.1,
                                       max_iter=3000, patience=50).fit(self.train)
            norms.append(np.sum(np.abs(model.theta)))

        self.assertTrue(all(later <= earlier for earlier, later in zip(norms, norms[1:])), norms)

    def test_l2_penalty_does_not_zero_the_coefficients(self):
        model = LogisticRegression(l2_penalty=500.0, l1_penalty=0.0, alpha=0.1,
                                   max_iter=3000, patience=50).fit(self.train)

        self.assertEqual(0, np.sum(model.theta == 0))

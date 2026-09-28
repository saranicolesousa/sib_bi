import unittest

import numpy as np

from si.statistics.euclidean_distance import euclidean_distance
from si.statistics.manhattan_distance import manhattan_distance
from si.statistics.minkowski_distance import minkowski_distance


class TestDistances(unittest.TestCase):

    def setUp(self):
        self.x = np.array([1.0, 2.0, 3.0])
        self.y = np.array([[1.0, 2.0, 3.0],
                           [4.0, 6.0, 8.0],
                           [0.0, 0.0, 0.0]])

    def test_manhattan_known_values(self):
        # |0|+|0|+|0| ; |3|+|4|+|5| ; |1|+|2|+|3|
        expected = np.array([0.0, 12.0, 6.0])
        self.assertTrue(np.allclose(expected, manhattan_distance(self.x, self.y)))

    def test_manhattan_output_shape(self):
        self.assertEqual((3,), manhattan_distance(self.x, self.y).shape)

    def test_minkowski_p1_equals_manhattan(self):
        self.assertTrue(np.allclose(manhattan_distance(self.x, self.y),
                                    minkowski_distance(self.x, self.y, p=1)))

    def test_minkowski_p2_equals_euclidean(self):
        self.assertTrue(np.allclose(euclidean_distance(self.x, self.y),
                                    minkowski_distance(self.x, self.y, p=2)))

    def test_minkowski_known_value_p3(self):
        # (3^3 + 4^3 + 5^3) ^ (1/3) = (27 + 64 + 125) ^ (1/3) = 6.0
        self.assertAlmostEqual(6.0, minkowski_distance(self.x, self.y, p=3)[1])

    def test_minkowski_invalid_p(self):
        with self.assertRaises(ValueError):
            minkowski_distance(self.x, self.y, p=0.5)

    def test_distance_to_itself_is_zero(self):
        for distance in (manhattan_distance, minkowski_distance, euclidean_distance):
            self.assertAlmostEqual(0.0, distance(self.x, self.y)[0])

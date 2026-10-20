import unittest

import numpy as np

from si.neural_networks.layers import Layer, DenseLayer


class TestLayer(unittest.TestCase):

    def test_layer_is_abstract(self):
        with self.assertRaises(TypeError):
            Layer()


class TestDenseLayer(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.layer = DenseLayer(4, (3,)).initialize()

    def test_slide_tests(self):
        out = self.layer.forward_propagation(np.random.rand(10, 3))
        self.assertEqual(out.shape, (10, 4))
        self.assertEqual(self.layer.parameters(), 3 * 4 + 4)  # 16
        self.assertEqual(self.layer.output_shape(), (4,))

    def test_initialization(self):
        self.assertEqual(self.layer.weights.shape, (3, 4))
        self.assertEqual(self.layer.biases.shape, (1, 4))
        self.assertTrue(np.all(self.layer.weights >= -0.5))
        self.assertTrue(np.all(self.layer.weights < 0.5))
        self.assertTrue(np.all(self.layer.biases == 0))

    def test_forward_matches_class_example(self):
        # slide 5 example: x = [1, 2], W (2x3), b (1x3) -> [[0.8, -0.2, 0.6]]
        layer = DenseLayer(3, (2,)).initialize()
        layer.weights = np.array([[0.1, 0.2, -0.1],
                                  [0.3, -0.2, 0.4]])
        layer.biases = np.array([[0.1, 0.0, -0.1]])
        out = layer.forward_propagation(np.array([[1, 2]]))
        np.testing.assert_allclose(out, [[0.8, -0.2, 0.6]])
        self.assertIs(layer.output, out)

    def test_layer_helpers(self):
        self.assertEqual(self.layer.input_shape(), (3,))
        self.assertEqual(self.layer.layer_name(), 'DenseLayer')
        self.layer.set_input_shape((5,))
        self.assertEqual(self.layer.input_shape(), (5,))

    def test_initialize_without_input_shape_fails(self):
        with self.assertRaises(ValueError):
            DenseLayer(4).initialize()

    def test_stacked_layers_2_3_1(self):
        # exercise from slide 5: 2 -> 3 -> 1 with 5 samples
        X = np.random.rand(5, 2)
        l1 = DenseLayer(3, (2,)).initialize()
        l2 = DenseLayer(1, l1.output_shape()).initialize()
        out = l2.forward_propagation(l1.forward_propagation(X))
        self.assertEqual(out.shape, (5, 1))


if __name__ == '__main__':
    unittest.main()

import copy
from abc import ABCMeta, abstractmethod

import numpy as np


class Layer(metaclass=ABCMeta):
    """
    Base class for every neural network layer.
    Defines the shared interface (forward pass, output shape, number of parameters)
    and the helpers to get/set the input shape. It has no __init__ on purpose:
    each concrete layer defines its own.
    """

    @abstractmethod
    def forward_propagation(self, input, training):
        """
        Computes the output of the layer for a given input.

        Parameters
        ----------
        input: np.ndarray
            The layer input
        training: bool
            Whether the layer is being used in training mode (relevant for e.g. dropout)

        Returns
        -------
        output: np.ndarray
            The layer output
        """
        raise NotImplementedError

    @abstractmethod
    def output_shape(self):
        """
        Returns the shape of the layer output (without the batch dimension).
        """
        raise NotImplementedError

    @abstractmethod
    def parameters(self):
        """
        Returns the number of trainable parameters of the layer.
        """
        raise NotImplementedError

    def set_input_shape(self, input_shape):
        """
        Sets the input shape of the layer (without the batch dimension).
        """
        self._input_shape = input_shape

    def input_shape(self):
        """
        Returns the input shape of the layer (without the batch dimension).
        """
        return self._input_shape

    def layer_name(self):
        """
        Returns the name of the layer (the class name).
        """
        return self.__class__.__name__


class DenseLayer(Layer):
    """
    Fully connected layer: every input feature is connected to every neuron.
    Output is computed as Y = XW + B.

    Parameters
    ----------
    n_units: int
        The number of neurons of the layer
    input_shape: tuple
        The input shape, without the batch dimension. For tabular data it is (n_features,)

    Attributes
    ----------
    input: np.ndarray
        The last input received in forward_propagation, shape (n_samples, n_features)
    output: np.ndarray
        The last output computed in forward_propagation, shape (n_samples, n_units)
    weights: np.ndarray
        The layer weights, shape (n_features, n_units)
    biases: np.ndarray
        The layer biases, shape (1, n_units)
    """

    def __init__(self, n_units: int, input_shape: tuple = None):
        super().__init__()
        if n_units < 1:
            raise ValueError("n_units must be a positive integer")
        self.n_units = n_units
        self._input_shape = input_shape

        # estimated parameters
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

        # optimizers (one per parameter set), set in initialize
        self.w_opt = None
        self.b_opt = None

    def initialize(self, optimizer=None) -> 'DenseLayer':
        """
        Initializes the weights with uniform random values in [-0.5, 0.5[ (n_features x n_units),
        the biases with zeros (1 x n_units) and, if given, one optimizer for the weights
        and another for the biases.

        Parameters
        ----------
        optimizer: Optimizer, optional
            The optimizer used to update weights and biases. It is deep-copied so that
            weights and biases keep independent optimizer states (e.g. momentum).

        Returns
        -------
        self: DenseLayer
            The initialized layer
        """
        if self.input_shape() is None:
            raise ValueError("input_shape must be set before initializing the layer")

        n_features = self.input_shape()[0]
        # np.random.rand is in [0, 1[ -> subtract 0.5 to centre the distribution at 0
        self.weights = np.random.rand(n_features, self.n_units) - 0.5
        self.biases = np.zeros((1, self.n_units))

        if optimizer is not None:
            self.w_opt = copy.deepcopy(optimizer)
            self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer (weights + biases).
        """
        if self.weights is None:
            raise ValueError("The layer must be initialized first")
        return int(np.prod(self.weights.shape) + np.prod(self.biases.shape))

    def forward_propagation(self, input: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Computes the layer output: Y = XW + B.

        Parameters
        ----------
        input: np.ndarray
            The layer input, shape (n_samples, n_features)
        training: bool
            Not used by this layer (kept for interface consistency)

        Returns
        -------
        output: np.ndarray
            The layer output, shape (n_samples, n_units)
        """
        if self.weights is None:
            raise ValueError("The layer must be initialized first")
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def output_shape(self) -> tuple:
        """
        Returns the shape of the layer output (without the batch dimension): (n_units,)
        """
        return (self.n_units,)

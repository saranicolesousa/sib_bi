import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function


class LogisticRegression(Model):
    """
    The LogisticRegression is a linear model for binary classification using regularization.
    This model solves the logistic regression problem using an adapted Gradient Descent technique.

    Extras (relative to the base exercise):
    - momentum: accelerates the gradient descent by keeping a fraction of the previous update
    - l1_penalty: L1 (Lasso) regularization, which drives coefficients towards exactly zero

    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    l1_penalty: float
        The L1 regularization parameter (0 disables L1)
    alpha: float
        The learning rate
    max_iter: int
        The maximum number of iterations
    patience: int
        The maximum number of iterations without improvement allowed
    scale: bool
        Whether to scale the dataset or not
    momentum: float
        The momentum coefficient, in [0, 1[ (0 disables momentum)

    Attributes
    ----------
    theta: np.ndarray
        The coefficients of the model for every feature
    theta_zero: float
        The zero coefficient (y intercept)
    mean: np.ndarray
        Mean of the dataset (for every feature)
    std: np.ndarray
        Standard deviation of the dataset (for every feature)
    cost_history: dict
        The value of the cost function at each iteration ({iteration: cost})
    """

    def __init__(self, l2_penalty: float = 1.0, l1_penalty: float = 0.0, alpha: float = 0.001,
                 max_iter: int = 1000, patience: int = 5, scale: bool = True,
                 momentum: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in the [0, 1[ interval")
        self.l2_penalty = l2_penalty
        self.l1_penalty = l1_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale
        self.momentum = momentum

        # estimated parameters
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Estimates theta, theta_zero, mean, std and cost_history using Gradient Descent.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: LogisticRegression
            The fitted model
        """
        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            # constant features have std = 0: keep them unscaled instead of dividing by zero
            self.std[self.std == 0] = 1
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        m, n = dataset.shape()

        self.theta = np.zeros(n)
        self.theta_zero = 0.0
        self.cost_history = {}

        # velocity of the momentum (the accumulated update)
        velocity = np.zeros(n)
        velocity_zero = 0.0

        i = 0
        early_stopping = 0
        while i < self.max_iter and early_stopping < self.patience:

            # predict the probability of y being 1
            y_pred = sigmoid_function(np.dot(X, self.theta) + self.theta_zero)

            # gradient of the cost function
            gradient = (1 / m) * np.dot(y_pred - dataset.y, X)
            gradient_zero = (1 / m) * np.sum(y_pred - dataset.y)

            # momentum: exponential moving average of the gradients
            #   v_t = beta * v_(t-1) + (1 - beta) * gradient
            # (with momentum = 0 this reduces to plain gradient descent)
            velocity = self.momentum * velocity + (1 - self.momentum) * gradient
            velocity_zero = self.momentum * velocity_zero + (1 - self.momentum) * gradient_zero

            # L2 regularization term
            penalization_term = self.theta * (1 - self.alpha * (self.l2_penalty / m))

            # update theta and theta_zero (the intercept is never penalized)
            theta = penalization_term - self.alpha * velocity
            self.theta_zero = self.theta_zero - self.alpha * velocity_zero

            # L1 regularization, applied as a soft-thresholding (proximal) step:
            #   theta_j = sign(theta_j) * max(|theta_j| - alpha * l1 / m, 0)
            # A plain sign() sub-gradient makes the coefficients oscillate around zero
            # instead of settling on it; soft-thresholding sets them to exactly zero.
            if self.l1_penalty:
                threshold = self.alpha * self.l1_penalty / m
                theta = np.sign(theta) * np.maximum(np.abs(theta) - threshold, 0)
            self.theta = theta

            self.cost_history[i] = self.cost(dataset)

            if i > 0 and self.cost_history[i] > self.cost_history[i - 1] - 0.0001:
                early_stopping += 1
            else:
                early_stopping = 0

            i += 1

        return self

    def predict_proba(self, dataset: Dataset) -> np.ndarray:
        """
        Estimates the probability of each sample belonging to class 1.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the probabilities of

        Returns
        -------
        probabilities: np.ndarray
            The estimated probabilities
        """
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X
        return sigmoid_function(np.dot(X, self.theta) + self.theta_zero)

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the class of each sample (probability >= 0.5 becomes 1).

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of

        Returns
        -------
        predictions: np.ndarray
            The predicted classes
        """
        return np.where(self.predict_proba(dataset) >= 0.5, 1, 0)

    def _score(self, dataset: Dataset) -> float:
        """
        Computes the accuracy between the real and predicted classes.

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on

        Returns
        -------
        accuracy: float
            The accuracy of the model
        """
        return accuracy(dataset.y, self._predict(dataset))

    def cost(self, dataset: Dataset) -> float:
        """
        Computes the regularized cross-entropy cost function between the real
        and predicted classes.

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost function on

        Returns
        -------
        cost: float
            The cost function of the model
        """
        predictions = self.predict_proba(dataset)
        # clip avoids log(0)
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        m = dataset.shape()[0]
        cost = -(dataset.y * np.log(predictions) + (1 - dataset.y) * np.log(1 - predictions)).sum() / m
        cost += (self.l2_penalty / (2 * m)) * np.sum(self.theta ** 2)
        cost += (self.l1_penalty / m) * np.sum(np.abs(self.theta))
        return cost

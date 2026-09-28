import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegression(Model):
    """
    The RidgeRegression is a linear model using the L2 regularization.
    This model solves the linear regression problem using an adapted Gradient Descent technique.

    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    alpha: float
        The learning rate
    max_iter: int
        The maximum number of iterations
    patience: int
        The maximum number of iterations without improvement allowed
    scale: bool
        Whether to scale the dataset or not

    Attributes
    ----------
    theta: np.ndarray
        The model parameters, namely the coefficients of the linear model.
        For example, x0 * theta[0] + x1 * theta[1] + ...
    theta_zero: float
        The model parameter, namely the intercept of the linear model.
        For example, theta_zero * 1
    mean: np.ndarray
        Mean of the dataset (for every feature)
    std: np.ndarray
        Standard deviation of the dataset (for every feature)
    cost_history: dict
        The value of the cost function at each iteration ({iteration: cost})
    """

    def __init__(self, l2_penalty: float = 1.0, alpha: float = 0.001, max_iter: int = 1000,
                 patience: int = 5, scale: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        # estimated parameters
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset: Dataset) -> 'RidgeRegression':
        """
        Estimates theta, theta_zero, mean, std and cost_history using Gradient Descent.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: RidgeRegression
            The fitted model
        """
        # 1. scale the data if required
        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0.0
        self.cost_history = {}

        i = 0
        early_stopping = 0
        # 8. run until max_iter or patience is reached
        while i < self.max_iter and early_stopping < self.patience:

            # 2. predict the values of Y
            y_pred = np.dot(X, self.theta) + self.theta_zero

            # 3. compute the gradient with the learning rate
            gradient = (self.alpha / m) * np.dot(y_pred - dataset.y, X)

            # 4. compute the L2 regularization term with the learning rate
            penalization_term = self.theta * (1 - self.alpha * (self.l2_penalty / m))

            # 5. update theta
            self.theta = penalization_term - gradient

            # 6. update theta_zero (not penalized)
            self.theta_zero = self.theta_zero - (self.alpha / m) * np.sum(y_pred - dataset.y)

            # 7. compute the cost function
            self.cost_history[i] = self.cost(dataset)

            # early stopping: no relevant improvement of the cost
            if i > 0 and self.cost_history[i] > self.cost_history[i - 1] - 0.0001:
                early_stopping += 1
            else:
                early_stopping = 0

            i += 1

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the dependent variable (y) using the estimated theta coefficients.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the values of

        Returns
        -------
        predictions: np.ndarray
            The predicted values of y
        """
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X
        return np.dot(X, self.theta) + self.theta_zero

    def _score(self, dataset: Dataset) -> float:
        """
        Computes the mean squared error between the real and predicted y values.

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on

        Returns
        -------
        mse: float
            The mean squared error of the model
        """
        y_pred = self._predict(dataset)
        return mse(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Computes the cost function (J) between the real and predicted y values.

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost function on

        Returns
        -------
        cost: float
            The cost function of the model
        """
        y_pred = self._predict(dataset)
        m = dataset.shape()[0]
        return (np.sum((y_pred - dataset.y) ** 2) + (self.l2_penalty * np.sum(self.theta ** 2))) / (2 * m)

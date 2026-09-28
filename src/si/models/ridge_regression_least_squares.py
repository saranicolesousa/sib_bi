import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares(Model):
    """
    The RidgeRegressionLeastSquares is a linear model using the L2 regularization.
    Unlike RidgeRegression, this model solves the linear regression problem analytically,
    using the least squares closed-form solution:

        theta = (X.T * X + l2_penalty * I) ^ -1 * X.T * y

    where the first position of the identity matrix is set to 0 so that the
    intercept (theta_zero) is not penalized.

    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    scale: bool
        Whether to scale the dataset or not

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
    """

    def __init__(self, l2_penalty: float = 1.0, scale: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.scale = scale

        # estimated parameters
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Estimates theta, theta_zero, mean and std using the least squares solution.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: RidgeRegressionLeastSquares
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

        # 2. add the intercept term to X (column of ones in the first position)
        X = np.c_[np.ones(m), X]

        # 3. compute the penalty matrix (l2_penalty * identity matrix)
        penalty_matrix = self.l2_penalty * np.eye(n + 1)

        # 4. set the first position to 0 so the intercept is not penalized
        penalty_matrix[0, 0] = 0

        # 5. compute the model parameters
        thetas = np.linalg.inv(X.T.dot(X) + penalty_matrix).dot(X.T).dot(dataset.y)
        self.theta_zero = thetas[0]
        self.theta = thetas[1:]

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
        # 1. scale the data using the mean and std estimated in the fit method
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X

        # 2. add the intercept term to X
        X = np.c_[np.ones(X.shape[0]), X]

        # 3. compute the predicted Y
        return X.dot(np.r_[self.theta_zero, self.theta])

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

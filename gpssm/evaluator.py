"""Python Script Template."""
from gpytorch.distributions import MultivariateNormal
from torch import Tensor


class Evaluator(object):
    """Object that evaluates the predictive performance of a model."""

    def __init__(self):
        self.criteria = ['loglik', 'rmse']

    def evaluate(self, predictions: MultivariateNormal, true_values: Tensor):
        """Return the RMS error between the true values and the mean predictions.

        Parameters
        ----------
        predictions: MultivariateNormal.
            A multivariate normal with loc [time x dim] and covariance (or scale)
            [time x dim x dim] or [time x dim].
        true_values: Tensor.
            A tensor with shape [time x dim].

        Returns
        -------
        log_likelihood: float.
        """
        return {criterion: getattr(self, criterion)(predictions, true_values)
                for criterion in self.criteria}

    def loglik(self, predictions: MultivariateNormal, true_values: Tensor) -> float:
        """Return the log likelihood of the true values under the predictions.

        Parameters
        ----------
        predictions: MultivariateNormal.
            A multivariate normal with loc [time x dim] and covariance (or scale)
            [time x dim x dim] or [time x dim].
        true_values: Tensor.
            A tensor with shape [time x dim].

        Returns
        -------
        log_likelihood: float.
        """
        return predictions.log_prob(true_values).sum().items()

    def rmse(self, predictions: MultivariateNormal, true_values: Tensor) -> float:
        """Return the RMS error between the true values and the mean predictions.

        Parameters
        ----------
        predictions: MultivariateNormal.
            A multivariate normal with loc [time x dim] and covariance (or scale)
            [time x dim x dim] or [time x dim].
        true_values: Tensor.
            A tensor with shape [time x dim].

        Returns
        -------
        log_likelihood: float.
        """
        return (predictions.loc - true_values).pow(2).mean().sqrt()

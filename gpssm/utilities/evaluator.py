"""Python Script Template."""
from torch.distributions import Normal
from torch import Tensor


class Evaluator(object):
    """Object that evaluates the predictive performance of a model."""

    def __init__(self):
        self.criteria = ['loglik', 'rmse']

    def evaluate(self, predictions: Normal, true_values: Tensor) -> dict:
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

    @staticmethod
    def loglik(predictions: Normal, true_values: Tensor) -> float:
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
        return predictions.log_prob(true_values).mean().item()

    @staticmethod
    def rmse(predictions: Normal, true_values: Tensor) -> float:
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
        l2 = (predictions.loc - true_values).pow(2).mean(dim=(1, 2))
        return l2.sqrt().mean().item()

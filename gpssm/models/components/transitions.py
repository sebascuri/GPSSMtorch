"""Emission model for GPSSM's."""
from gpytorch.likelihoods import Likelihood
from torch.nn import ModuleList
from torch import Tensor
from gpytorch.distributions import MultivariateNormal
from typing import List, Union


class Transitions(Likelihood):
    """Implementation of Transitions of a GPSSMs.

    Parameters
    ----------
    likelihoods: list of Likelihood.
        list of likelihoods for each component.

    Examples
    --------
    >>> from gpytorch.distributions import MultivariateNormal
    >>> import torch
    >>> from torch import Size
    >>> from gpytorch.likelihoods import GaussianLikelihood
    >>> from torch.testing import assert_allclose
    >>> dim_states, dim_outputs = 3, 2
    >>> batch_size, num_particles = 32, 8
    >>> transitions = Transitions([GaussianLikelihood() for _ in range(dim_states)])
    >>> loc = torch.randn(batch_size, num_particles)
    >>> cov = torch.eye(num_particles).expand(batch_size, num_particles, num_particles)
    >>> next_f = [MultivariateNormal(loc, cov) for _ in range(dim_states)]
    >>> next_state = transitions(next_f)
    >>> assert_allclose(next_state[0].loc, loc)
    >>> q = transitions.likelihoods[0].noise_covar.noise * torch.eye(num_particles)
    >>> assert_allclose(next_state[0].covariance_matrix, cov + q)
    """

    def __init__(self, likelihoods: List[Likelihood]) -> None:
        super().__init__()
        self.likelihoods = ModuleList(likelihoods)

    def __str__(self) -> str:
        """Return transition model parameters as a string."""
        string = ""
        for i in range(len(self.likelihoods)):
            noise_str = str(self.likelihoods[i].noise_covar.noise.detach())
            string += "component {} {}".format(i, noise_str)
        return string

    def __call__(self, next_f: List[Union[MultivariateNormal, Tensor]], *args, **kwargs
                 ) -> List[MultivariateNormal]:
        """Compute the conditional or marginal distribution for each likelihood."""
        return [l(next_f_, *args, **kwargs)
                for l, next_f_ in zip(self.likelihoods, next_f)]

    def forward(self, function_samples: List[Tensor], *args, **kwargs
                ) -> List[MultivariateNormal]:
        """Compute the conditional distribution p(y|f) that defines the likelihoods.

        Parameters
        ----------
        function_samples: list of Tensor.
            Samples from function `f`.

        Returns
        -------
        y: list of MultivariateNormal.
            Distribution object (with same shape as `function_samples`).
        """
        return [l(y) for l, y in zip(self.likelihoods, function_samples)]

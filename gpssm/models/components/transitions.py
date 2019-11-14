"""Emission model for GPSSM's."""
from gpytorch.likelihoods import Likelihood
import torch
import torch.nn as nn
from torch import Tensor
from gpytorch.distributions import MultivariateNormal
from typing import List, Union
State = Union[Tensor, MultivariateNormal]

__author__ = 'Sebastian Curi'
__all__ = ['Transitions']


class Transitions(nn.Module):
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
    >>> deb = str(transitions)
    >>> loc = torch.randn(dim_states, batch_size, num_particles)
    >>> cov = torch.eye(num_particles)
    >>> cov = cov.expand(dim_states, batch_size, num_particles, num_particles)
    >>> next_f = MultivariateNormal(loc, cov)
    >>> next_state = transitions(next_f)
    """

    def __init__(self, likelihoods: List[Likelihood]) -> None:
        super().__init__()
        self.likelihoods = likelihoods
        for idx, likelihood in enumerate(likelihoods):
            self.add_module('transition_{}'.format(idx), likelihood)

    def __str__(self) -> str:
        """Return transition model parameters as a string."""
        string = ""
        for i in range(len(self.likelihoods)):
            noise_str = str(
                self.likelihoods[i].noise_covar.noise.detach())  # type: ignore
            string += "component {} {} ".format(i, noise_str)
        return string

    def __call__(self, next_f: MultivariateNormal, *args, **kwargs
                 ) -> MultivariateNormal:
        """See `self.forward'."""
        return self.forward(next_f, *args, **kwargs)

    def forward(self, f_samples: MultivariateNormal, *args, **kwargs
                ) -> MultivariateNormal:
        """Compute the conditional or marginal distribution of the transmission.

         If f_samples is a Tensor (or a List of Tensors) then compute the conditional
         p(x|f).
         If f_samples is a MultivariateNormal (or a List of Multivariate Normals) then
         compute the marginal p(x).

        Parameters
        ----------
        f_samples: State.
            State of dimension dim_state x batch_size x num_particles.

        Returns
        -------
        next_state: MultivariateNormal.
            Next state of dimension dim_state x batch_size x num_particles.
        """
        out = [self.likelihoods[i](MultivariateNormal(
            f_samples.loc[i], f_samples.covariance_matrix[i]
        ), *args, **kwargs)
               for i in range(len(self.likelihoods))]

        loc = torch.stack([f.loc for f in out])
        cov = torch.stack([f.covariance_matrix for f in out])
        return MultivariateNormal(loc, cov)

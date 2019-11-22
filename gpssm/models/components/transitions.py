"""Emission model for GPSSM's."""
import numpy as np
import torch
import torch.nn as nn
from gpytorch.distributions import MultivariateNormal


__author__ = 'Sebastian Curi'
__all__ = ['Transitions']


class Transitions(nn.Module):
    """Implementation of Transitions of the first n states.

    Parameters
    ----------
    dim_states: int.
        State dimension.
    variance: float.
        Initial variance estimate.
    learnable: bool.
        Flag that indicates if parameters are learnable.

    """

    def __init__(self, dim_states: int, variance: float = 1.0, learnable: bool = True):
        super().__init__()
        self.dim_states = dim_states
        self.sd_noise = nn.Parameter(torch.ones(dim_states) * np.sqrt(variance),
                                     requires_grad=learnable)

    def __str__(self) -> str:
        """Return emission model parameters as a string."""
        return str(self.sd_noise.detach().numpy() ** 2)

    def forward(self, *args: MultivariateNormal, **kwargs) -> MultivariateNormal:
        """Compute the marginal distribution of the transmission.

        Parameters
        ----------
        args: MultivariateNormal.
            State of dimension batch_size x dim_state x num_particles.

        Returns
        -------
        next_state: MultivariateNormal.
            Next state of dimension batch_size x dim_state x num_particles.
        """
        f_samples = args[0]
        batch_size, dim_state, num_particles = f_samples.loc.shape
        cov = torch.diag_embed((self.sd_noise ** 2).expand(
            batch_size, num_particles, dim_state).permute(0, 2, 1))

        return MultivariateNormal(f_samples.loc, f_samples.lazy_covariance_matrix + cov)

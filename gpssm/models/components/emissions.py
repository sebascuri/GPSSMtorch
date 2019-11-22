"""Emission model for GPSSM's."""
from torch import Tensor
from torch.distributions import Normal
import numpy as np
import torch
import torch.nn as nn

__author__ = 'Sebastian Curi'
__all__ = ['Emissions']


class Emissions(nn.Module):
    """Implementation of Emissions of the first n states.

    Parameters
    ----------
    dim_outputs: int.
        Output dimension.
    variance: float.
        Initial variance estimate.
    learnable: bool.
        Flag that indicates if parameters are learnable.

    """

    def __init__(self, dim_outputs: int, variance: float = 1.0, learnable: bool = True,
                 ) -> None:
        super().__init__()
        self.dim_outputs = dim_outputs
        self.sd_noise = nn.Parameter(torch.ones(dim_outputs) * np.sqrt(variance),
                                     requires_grad=learnable)

    def __str__(self) -> str:
        """Return emission model parameters as a string."""
        return str(self.sd_noise.detach().numpy() ** 2)

    def forward(self, *args: Tensor, **kwargs) -> Normal:
        """Compute the conditional distribution of the emissions p(y|f).

        Parameters
        ----------
        args: Tensor.
            State of dimension batch_size x dim_state x num_particles.

        Returns
        -------
        y: Normal.
            Output of dimension batch_size x dim_output x num_particles.
        """
        state = args[0]
        batch_size, _, num_particles = state.shape
        loc = state[:, :self.dim_outputs]
        cov = (self.sd_noise ** 2).expand(batch_size, num_particles,
                                          self.dim_outputs).transpose(1, 2)
        return Normal(loc, cov)

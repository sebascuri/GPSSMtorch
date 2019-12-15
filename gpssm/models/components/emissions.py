"""Emission model for GPSSM's."""
from torch import Tensor
from gpytorch.distributions import MultivariateNormal
from .utilities import inverse_softplus, safe_softplus
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
        self.variance_t = nn.Parameter(
            torch.ones(dim_outputs) * inverse_softplus(torch.tensor(variance)),
            requires_grad=learnable)

    def __str__(self) -> str:
        """Return emission model parameters as a string."""
        return str(self.variance.detach().numpy())

    @property
    def variance(self) -> torch.Tensor:
        """Get Diagonal Covariance Matrix."""
        return safe_softplus(self.variance_t)

    def forward(self, *args: MultivariateNormal, **kwargs) -> MultivariateNormal:
        """Compute the marginal distribution p(y).

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
        if type(state) == Tensor:
            batch_size, _, num_particles = state.shape
            loc = state[:, :self.dim_outputs]

        else:
            batch_size, dim_states, num_particles = state.loc.shape
            loc = state.loc[:, :self.dim_outputs]

        cov = self.variance.expand(batch_size, num_particles, self.dim_outputs
                                   ).transpose(1, 2)
        cov = torch.diag_embed(cov)

        return MultivariateNormal(loc, cov)

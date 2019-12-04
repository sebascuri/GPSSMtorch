"""Emission model for GPSSM's."""
import torch
import torch.nn as nn
from gpytorch.distributions import MultivariateNormal
from .utilities import inverse_softplus, safe_softplus

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
        self.variance_t = nn.Parameter(
            torch.ones(dim_states) * inverse_softplus(torch.tensor(variance)),
            requires_grad=learnable)

    def __str__(self) -> str:
        """Return emission model parameters as a string."""
        return str(self.variance.detach().numpy())

    @property
    def variance(self) -> torch.Tensor:
        """Get Diagonal Covariance Matrix."""
        return safe_softplus(self.variance_t)

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
        cov = self.variance.expand(batch_size, num_particles, dim_state).transpose(1, 2)
        return MultivariateNormal(
            f_samples.loc, f_samples.lazy_covariance_matrix.add_diag(cov).add_jitter())

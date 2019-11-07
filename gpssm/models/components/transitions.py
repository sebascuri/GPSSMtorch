"""Emission model for GPSSM's."""
from gpytorch.likelihoods import Likelihood
from torch.nn import ModuleList
from torch import Tensor
from gpytorch.distributions import MultivariateNormal
from typing import List, Union
State = Union[Tensor, MultivariateNormal]

__author__ = 'Sebastian Curi'
__all__ = ['Transitions']


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
    >>> deb = str(transitions)
    >>> loc = torch.randn(batch_size, num_particles)
    >>> cov = torch.eye(num_particles).expand(batch_size, num_particles, num_particles)
    >>> next_f = [MultivariateNormal(loc, cov) for _ in range(dim_states)]
    >>> next_state = transitions(next_f)
    >>> for i in range(dim_states):
    ...     assert_allclose(next_state[i].loc, loc)
    ...     q = transitions.likelihoods[i].noise_covar.noise * torch.eye(num_particles)
    ...     assert_allclose(next_state[i].covariance_matrix, cov + q)
    >>> states = torch.randn(dim_states, batch_size, num_particles)
    >>> next_state = transitions.forward(states)
    >>> for i in range(dim_states):
    ...     assert next_state[i].loc.shape == torch.Size([batch_size, num_particles])
    ...     assert next_state[i].scale.shape == torch.Size([batch_size, num_particles])
    >>> states = [torch.randn(batch_size, num_particles) for _ in range(dim_states)]
    >>> next_state = transitions.forward(states)
    >>> for i in range(dim_states):
    ...     assert next_state[i].loc.shape == torch.Size([batch_size, num_particles])
    ...     assert next_state[i].scale.shape == torch.Size([batch_size, num_particles])
    """

    def __init__(self, likelihoods: List[Likelihood]) -> None:
        super().__init__()
        self.likelihoods = ModuleList(likelihoods)

    def __str__(self) -> str:
        """Return transition model parameters as a string."""
        string = ""
        for i in range(len(self.likelihoods)):
            noise_str = str(self.likelihoods[i].noise_covar.noise.detach())
            string += "component {} {} ".format(i, noise_str)
        return string

    def __call__(self, next_f: Union[State, List[State]], *args, **kwargs
                 ) -> List[MultivariateNormal]:
        """See `self.forward'."""
        return self.forward(next_f, *args, **kwargs)

    def forward(self, f_samples: Union[State, List[State]], *args, **kwargs
                ) -> List[MultivariateNormal]:
        """Compute the conditional or marginal distribution of the transmission.

         If f_samples is a Tensor (or a List of Tensors) then compute the conditional
         p(x|f).
         If f_samples is a MultivariateNormal (or a List of Multivariate Normals) then
         compute the marginal p(x).

        Parameters
        ----------
        f_samples: State or List of State.
            State of dimension dim_state x batch_size x num_particles or a list of
            length dim_state of State with dimension batch_size x num_particles.

        Returns
        -------
        y: list of MultivariateNormal.
            List of length dim_state of MultivariateNormal with dimension batch_size x
            num_particles.
        """
        return [l(f, *args, **kwargs) for l, f in zip(self.likelihoods, f_samples)]

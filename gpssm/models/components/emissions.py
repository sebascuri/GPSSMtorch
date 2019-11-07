"""Emission model for GPSSM's."""
from gpytorch.likelihoods import Likelihood
from gpytorch.distributions import MultivariateNormal
from torch import Tensor
from torch.nn import ModuleList
from typing import Union, List
State = Union[Tensor, MultivariateNormal]

__author__ = 'Sebastian Curi'
__all__ = ['Emissions']


class Emissions(Likelihood):
    """Implementation of Emissions of the first n states.

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
    >>> num_particles = 8
    >>> emissions = Emissions([GaussianLikelihood(), GaussianLikelihood()])
    >>> debug_str = str(emissions)
    >>> d = MultivariateNormal(torch.zeros(dim_states), torch.eye(dim_states))
    >>> x = d.rsample(sample_shape=torch.Size([num_particles]))
    >>> c = torch.zeros((dim_states, dim_outputs))
    >>> c[:dim_outputs, :dim_outputs] = torch.eye(dim_outputs)
    >>> for i in range(dim_outputs):
    ...     assert_allclose(emissions(x)[i].loc, (x @ c)[:, i])
    ...     assert_allclose(emissions(x)[i].loc.shape, Size([num_particles]))
    ...     assert_allclose(emissions(x)[i].scale.shape, Size([num_particles]))
    >>> loc = torch.randn(num_particles, dim_states)
    >>> cov = 0.1 * torch.eye(dim_states).expand(num_particles, dim_states, dim_states)
    >>> d = MultivariateNormal(loc, cov)
    >>> y = emissions(d)
    >>> for i in range(dim_outputs):
    ...     assert_allclose(emissions.forward(d)[i].loc[:, 0], loc[:, i])
    ...     assert_allclose(emissions.forward(d)[i].loc.shape, Size([num_particles, 1]))
    ...     assert_allclose(emissions.forward(d)[i].covariance_matrix.shape,
    ...         Size([num_particles, 1, 1]))
    >>> x = [torch.randn(32, num_particles) for _ in range(dim_states)]
    >>> for i in range(dim_outputs):
    ...     assert_allclose(emissions(x)[i].loc.shape, Size([32, num_particles]))
    ...     assert_allclose(emissions(x)[i].scale.shape, Size([32, num_particles]))
    """

    def __init__(self, likelihoods: List[Likelihood]) -> None:
        super().__init__()
        self.dim_outputs = len(likelihoods)
        self.likelihoods = ModuleList(likelihoods)

    def __str__(self) -> str:
        """Return recognition model parameters as a string."""
        string = ""
        for i in range(self.dim_outputs):
            string += "component {} {}".format(
                i, str(self.likelihoods[i].noise_covar.noise.detach()))
        return string

    def __call__(self, state: Union[State, List[State]], *args, **kwargs
                 ) -> List[MultivariateNormal]:
        """See `self.forward'."""
        return self.forward(state, *args, **kwargs)

    def forward(self, state: Union[State, List[State]], *args, **kwargs
                ) -> List[MultivariateNormal]:
        """Compute the conditional or marginal distribution of the emissions.

         If f_samples is a Tensor (or a List of Tensors) then compute the conditional
         p(y|f).
         If f_samples is a MultivariateNormal (or a List of Multivariate Normals) then
         compute the marginal p(y).

        Parameters
        ----------
        state: State or List of State.
            State of dimension dim_state x batch_size x num_particles or a list of
            length dim_state of State with dimension batch_size x num_particles.

        Returns
        -------
        y: list of MultivariateNormal.
            List of length dim_output of MultivariateNormal with dimension batch_size x
            num_particles.
        """
        if type(state) is Tensor:
            return [self.likelihoods[i](state[..., i]) for i in range(self.dim_outputs)]

        elif type(state) is MultivariateNormal:
            return [self.likelihoods[i](MultivariateNormal(
                state.loc[:, i:(i + 1)],
                state.covariance_matrix[:, i:(i + 1), i:(i + 1)]))
                for i in range(self.dim_outputs)]
        elif type(state) is list:
            return [self.likelihoods[i](state[i]) for i in range(self.dim_outputs)]
        else:
            raise NotImplementedError('Type {} of state not implemented'.format(
                type(state)))

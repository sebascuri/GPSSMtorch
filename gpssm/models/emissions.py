"""Emission model for GPSSM's."""
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.distributions import MultivariateNormal
from abc import ABC
import torch
from torch import Tensor
from typing import Union, Any

__author__ = 'Sebastian Curi'
__all__ = ['Emission', 'GaussianEmission']


class Emission(ABC):
    """Base class of emission model."""

    def __init__(self, dim_states: int, dim_outputs: int) -> None:
        self.dim_states = dim_states
        self.dim_outputs = dim_outputs

    def __call__(self, state: Tensor) -> MultivariateNormal:
        """Call the emission model for a given state.

        Parameters
        ----------
        state: Tensor.
            State tensor of size [state_dim x batch_size].

        Returns
        -------
        output_distribution: MultivariateNormal.
            Distribution of size [output_dim x batch_size].
        """
        raise NotImplementedError


class GaussianEmission(Emission):
    """Implementation of Gaussian Emissions with a fixed Measurement Function.

    Parameters
    ----------
    dim_states: int.
        State dimension.

    dim_outputs: int.
        Output dimension.

    batch_size: int, optional.
        Batch size of Likelihood (default: 1).

    likelihood: Likelihood, optional.
        Emission likelihood (default: Gaussian Likelihood).

    Examples
    --------
    >>> from gpytorch.distributions import MultivariateNormal
    >>> from torch.testing import assert_allclose
    >>> dim_states, dim_outputs = 3, 2
    >>> num_particles = 8
    >>> emission = GaussianEmission(dim_states=dim_states, dim_outputs=dim_outputs)
    >>> d = MultivariateNormal(torch.zeros(dim_states), torch.eye(dim_states))
    >>> x = d.rsample(sample_shape=torch.Size([num_particles]))
    >>> x = x.transpose(-1, -2)
    >>> c = torch.zeros((dim_outputs, dim_states))
    >>> c[:dim_outputs, :dim_outputs] = torch.eye(dim_outputs)
    >>> assert_allclose(emission(x).loc, c @ x)
    >>> assert_allclose(emission(x).loc.shape, torch.Size([dim_outputs, num_particles]))
    """

    def __init__(self, dim_states: int, dim_outputs: int,
                 likelihood: Likelihood = None) -> None:
        super().__init__(dim_states, dim_outputs)
        self._c = torch.zeros((dim_outputs, dim_states))
        self._c[:dim_outputs, :dim_outputs] = torch.eye(dim_outputs)

        if likelihood is not None:
            self.likelihood = likelihood
        else:
            self.likelihood = GaussianLikelihood()

    def __call__(self, state: Union[Tensor, MultivariateNormal]) -> MultivariateNormal:
        """Call the emission model for a given state.

        Parameters
        ----------
        state: Tensor or MultivariateNormal.
            State tensor of size [state_dim x num_particles].

        Returns
        -------
        output_distribution: MultivariateNormal
            Distribution of size [output_dim x num_particles].
        """
        if type(state) is Tensor:
            return self.likelihood(state[:self.dim_outputs])
        elif type(state) is MultivariateNormal:
            state_distribution = MultivariateNormal(
                state.loc[:self.dim_outputs],
                state.covariance_matrix[:self.dim_outputs])
            return self.likelihood(state_distribution)
        else:
            raise TypeError('Type {} of state not understood'.format(type(state)))

    def expected_log_prob(self, measurement: Tensor, state: MultivariateNormal,
                          *params: Any, **kwargs: Any) -> Tensor:
        """Return the expected log prob of a `tensor' under the `state' distribution."""
        predicted_measurement = MultivariateNormal(
            state.loc[:self.dim_outputs],
            state.covariance_matrix[:self.dim_outputs])
        return self.likelihood.expected_log_prob(measurement, predicted_measurement,
                                                 *params, **kwargs)

"""Emission model for GPSSM's."""
from gpytorch.likelihoods import Likelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods.noise_models import Noise
from torch import Tensor
from torch.nn import ModuleList
from typing import Union, Any, List

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
    >>> emission = Emissions([GaussianLikelihood(), GaussianLikelihood()])
    >>> d = MultivariateNormal(torch.zeros(dim_states), torch.eye(dim_states))
    >>> x = d.rsample(sample_shape=torch.Size([num_particles]))
    >>> c = torch.zeros((dim_states, dim_outputs))
    >>> c[:dim_outputs, :dim_outputs] = torch.eye(dim_outputs)
    >>> assert_allclose(emission(x)[0].loc, (x @ c)[:, 0])
    >>> assert_allclose(emission(x)[0].loc.shape, Size([num_particles]))
    >>> assert_allclose(emission(x)[0].scale.shape, Size([num_particles]))
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

    def __call__(self, state: Union[Tensor, MultivariateNormal,
                                    List[MultivariateNormal]],
                 *args, **kwargs) -> List[MultivariateNormal]:
        """Call the emission model for a given state.

        Parameters
        ----------
        state: Tensor, MultivariateNormal or list of MultivariateNormal.
            State tensor of size [batch_size x num_particles x dim_state] or,
            MultivariateNormal of size [batch_size x dim_state] or,
            list w/ length dim_state of MultivariateNormal [batch_size x num_particles].

        Returns
        -------
        output_distribution: List of MultivariateNormal
            list w/ len dim_output of MultivariateNormal [batch_size x num_particles].

        """
        if type(state) is Tensor:
            return [self.likelihoods[i](state[..., i]) for i in range(self.dim_outputs)]

        elif type(state) is MultivariateNormal:
            return [self.likelihoods[i](MultivariateNormal(
                state.loc[:, i:(i+1)],
                state.covariance_matrix[:, i:(i + 1), i:(i + 1)]))
                for i in range(self.dim_outputs)]
        elif type(state) is list:
            return [self.likelihoods[i](state[i]) for i in range(self.dim_outputs)]
        else:
            raise TypeError('Type {} of state not understood'.format(type(state)))

    def forward(self, state: List[Tensor], *args, **kwargs
                ) -> List[MultivariateNormal]:
        """Compute the conditional distribution p(y|f) that defines the likelihoods.

        Parameters
        ----------
        state: list of Tensor.
            Samples from function `f`.

        Returns
        -------
        y: list of MultivariateNormal.
            Distribution object (with same shape as `function_samples`).
        """
        return [self.likelihoods[i](state[i]) for i in range(self.dim_outputs)]

    def noise_covar(self) -> Noise:
        """Return component noise covariance."""
        return self.likelihoods.noise_covar

    def expected_log_prob(self, measurement: Tensor,
                          state: List[MultivariateNormal],
                          *params: Any, **kwargs: Any) -> List[Tensor]:
        """Return the expected log prob of a `tensor' under the `state' distribution."""
        return [self.likelihoods[i].expected_log_prob(measurement[..., i], state[i],
                                                      *params, **kwargs)
                for i in range(self.dim_outputs)]
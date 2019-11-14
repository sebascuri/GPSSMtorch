"""Emission model for GPSSM's."""
from gpytorch.likelihoods import Likelihood
from gpytorch.distributions import MultivariateNormal
from torch import Tensor
from torch.distributions import Normal
import torch
import torch.nn as nn
from typing import Union, List
State = Union[Tensor, MultivariateNormal]

__author__ = 'Sebastian Curi'
__all__ = ['Emissions']


class Emissions(nn.Module):
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
    >>> batch_size = 32
    >>> emissions = Emissions([GaussianLikelihood(), GaussianLikelihood()])
    >>> debug_str = str(emissions)
    >>> d = MultivariateNormal(torch.zeros(dim_states), torch.eye(dim_states))
    >>> x = d.rsample(sample_shape=torch.Size([num_particles]))
    >>> c = torch.zeros((dim_states, dim_outputs))
    >>> c[:dim_outputs, :dim_outputs] = torch.eye(dim_outputs)
    >>> y = emissions(x)
    >>> assert_allclose(y.loc, x[:, :dim_outputs].t())
    >>> assert y.loc.shape == Size([dim_outputs, num_particles])
    >>> assert y.scale.shape == Size([dim_outputs, num_particles])
    >>> x = d.rsample(sample_shape=torch.Size([batch_size, num_particles]))
    >>> y = emissions(x)
    >>> assert_allclose(y.loc, x[:, :, :dim_outputs].permute(2, 0, 1))
    >>> assert y.loc.shape == Size([dim_outputs, batch_size, num_particles])
    >>> assert y.scale.shape == Size([dim_outputs, batch_size, num_particles])
    """

    def __init__(self, likelihoods: List[Likelihood]) -> None:
        super().__init__()
        self.dim_outputs = len(likelihoods)
        self.likelihoods = likelihoods
        for idx, likelihood in enumerate(likelihoods):
            self.add_module('emission_{}'.format(idx), likelihood)

    def __str__(self) -> str:
        """Return recognition model parameters as a string."""
        string = ""
        for i in range(self.dim_outputs):
            string += "component {} {}".format(
                i, str(self.likelihoods[i].noise_covar.noise.detach()))  # type: ignore
        return string

    def forward(self, *args: Tensor, **kwargs) -> Normal:
        """Compute the conditional distribution of the emissions p(y|f).

        Parameters
        ----------
        args: Tensor.
            State of dimension batch_size x num_particles x dim_state.

        Returns
        -------
        y: Normal.
            Output of dimension dim_output x batch_size x num_particles.
        """
        state = args[0]
        y = [self.likelihoods[i](state[..., i], *args[1:], **kwargs)
             for i in range(self.dim_outputs)]
        loc = torch.stack([yi.loc for yi in y])
        cov = torch.stack([yi.scale for yi in y])
        return Normal(loc, cov)

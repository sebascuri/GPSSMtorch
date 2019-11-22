"""Emission model for GPSSM's."""
from gpytorch.likelihoods import Likelihood
from torch import Tensor
from torch.distributions import Normal, MultivariateNormal
import numpy as np
import torch
import torch.nn as nn
from typing import Union, List

State = Union[Tensor, MultivariateNormal]

__author__ = 'Sebastian Curi'
__all__ = ['Emissions', 'EmissionsNN']


class Emissions(nn.Module):
    """Implementation of Emissions of the first n states.

    Parameters
    ----------
    likelihoods: list of Likelihood.
        list of likelihoods for each component.

    Examples
    --------
    >>> from gpytorch.distributions import MultivariateNormal as MN
    >>> import torch
    >>> from torch import Size
    >>> from gpytorch.likelihoods import GaussianLikelihood
    >>> from torch.testing import assert_allclose
    >>> dim_states, dim_outputs = 3, 2
    >>> num_particles = 8
    >>> batch_size = 32
    >>> emissions = Emissions([GaussianLikelihood(), GaussianLikelihood()])
    >>> debug_str = str(emissions)
    >>> d = MN(torch.zeros(dim_states), torch.eye(dim_states))
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
        """Return emission model parameters as a string."""
        string = ""
        for i in range(self.dim_outputs):
            string += " component {} {}\n".format(
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


class EmissionsNN(nn.Module):
    """Implementation of Emissions of the first n states. in pytorch."""

    def __init__(self, dim_outputs: int, variance: float = 1.0, learnable: bool = True,
                 kind: str = 'diagonal'):
        super().__init__()
        self.dim_outputs = dim_outputs
        if kind == 'diagonal':
            self.sd_noise = nn.Parameter(torch.ones(dim_outputs) * np.sqrt(variance),
                                         requires_grad=learnable)
        else:
            raise NotImplementedError
        # elif type == 'full':
        #     # This will be the cholesky matrix.
        #     self.sd_noise = nn.Parameter(torch.eye(dim_outputs) * np.sqrt(variance),
        #                                  requires_grad=learnable)

    def __str__(self) -> str:
        """Return emission model parameters as a string."""
        return str(self.sd_noise.detach().numpy() ** 2)

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
        loc = state[..., :self.dim_outputs].permute(
            -1, *torch.arange(state.ndimension() - 1))
        cov = (self.sd_noise ** 2).expand(loc.shape)
        return Normal(loc, cov)

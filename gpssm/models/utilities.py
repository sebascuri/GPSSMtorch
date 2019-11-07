"""Utilities for SSMVI models."""

import torch
import torch.nn as nn

import numpy as np
from gpssm.models.components.emissions import Emissions
from gpssm.models.components.transitions import Transitions
from gpssm.models.components.gp import VariationalGP, ModelList

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, ZeroMean, LinearMean, Mean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, LinearKernel, Kernel
from gpytorch.variational import CholeskyVariationalDistribution

__author__ = 'Sebastian Curi'
__all__ = ['get_inducing_points', 'init_emissions', 'init_transmissions', 'init_gps']


def get_inducing_points(num_inducing_points: int, dim_inputs: int,
                        strategy: str = 'normal', scale: float = 1) -> torch.Tensor:
    """Initialize inducing points for variational GP.

    Parameters
    ----------
    num_inducing_points: int.
        Number of inducing points.
    dim_inputs: int.
        Input dimensionality.
    strategy: str, optional.
        Strategy to generate inducing points (by default normal).
        Either either normal, uniform or linspace.
    scale: float, optional.
        Scale of inducing points (default 1)

    Returns
    -------
    inducing_point: torch.Tensor.
        Inducing points with shape [dim_outputs x num_inducing_points x dim_inputs]

    Examples
    --------
    >>> num_inducing_points, dim_inputs = 24, 8
    >>> for strategy in ['normal', 'uniform', 'linspace']:
    ...     ip = get_inducing_points(num_inducing_points, dim_inputs, strategy, 2.)
    ...     assert type(ip) == torch.Tensor
    ...     assert ip.shape == torch.Size([num_inducing_points, dim_inputs])
    """
    if strategy == 'normal':
        ip = scale * torch.randn((num_inducing_points, dim_inputs))
    elif strategy == 'uniform':
        ip = scale * torch.rand((num_inducing_points, dim_inputs)) - (scale / 2)
    elif strategy == 'linspace':
        lin_points = int(np.ceil(num_inducing_points ** (1 / dim_inputs)))
        ip = np.linspace(-scale, scale, lin_points)
        ip = np.array(np.meshgrid(*([ip] * dim_inputs))).reshape(dim_inputs, -1).T
        idx = np.random.choice(np.arange(ip.shape[0]), size=num_inducing_points,
                               replace=False)
        ip = torch.from_numpy(ip[idx]).float()
    else:
        raise ValueError("strategy {} not implemented.".format(strategy))
    assert ip.shape == torch.Size([num_inducing_points, dim_inputs])
    return ip


def init_emissions(dim_outputs: int, initial_variance: float = None,
                   learnable: bool = True, shared: bool = False) -> Emissions:
    """Initialize emission model.

    Parameters
    ----------
    dim_outputs: int.
        Dimension of output space.
    initial_variance: float.
        Initial emission covariance estimate.
    learnable: bool.
        Flag that indicates if module is learnable.
    shared: bool.
        Flag that indicates if module parameters are shared between outputs.

    Returns
    -------
    emission: Emissions.
        Initialized emission module.
    """
    return Emissions(likelihoods=_init_likelihood_list(dim_outputs, initial_variance,
                                                       learnable, shared))


def init_transmissions(dim_states: int, initial_variance: float = None,
                       learnable: bool = True, shared: bool = False) -> Transitions:
    """Initialize transmissions model.

    Parameters
    ----------
    dim_states: int.
        Dimension of state space.
    initial_variance: float.
        Initial emission covariance estimate.
    learnable: bool.
        Flag that indicates if module is learnable.
    shared: bool.
        Flag that indicates if module parameters are shared between outputs.

    Returns
    -------
    emission: Emissions.
        Initialized emission module.

    """
    return Transitions(likelihoods=_init_likelihood_list(dim_states, initial_variance,
                                                         learnable, shared))


def init_gps(dim_states: int, dim_inputs: int,
             ard_num_dims: int = None,
             inducing_points_number: int = 20,
             inducing_points_strategy: str = 'normal',
             inducing_points_scale: float = 1.0,
             inducing_points_mean: float = None,
             inducing_points_var: float = None,
             inducing_points_learn_loc: bool = True,
             inducing_points_learn_mean: bool = True,
             inducing_points_learn_var: bool = True,
             mean_str: str = 'zero',
             kernel_str: str = 'rbf',
             kernel_outputscale: float = None,
             kernel_lengthscale: float = None,
             kernel_learn_outputscale: bool = True,
             kernel_learn_lengthscale: bool = True,
             shared: bool = False
             ) -> ModelList:
    """Initialize GP Model.

    Parameters
    ----------
    dim_states: int.
        Dimension of hidden states.
    dim_inputs:
        Dimension of inputs.
    ard_num_dims: int, optional.
        Number of Automatic Relevance Detection components (default: all).
    mean_str: str, optional.
        Mean function identifier (default: zero).
    kernel_str: str, optional.
        Kernel function identifier (default: rbf).
    kernel_outputscale: float, optional.
        Initial kernel output scale.
    kernel_lengthscale: float, optional.
        Initial kernel lengthscale.
    shared: bool, optional.
        Flag that indicates if mean and kernel are shared (default: False).
    inducing_points_number: int, optional.
        Number of inducing points (default: 20).
    inducing_points_strategy: str, optional.
        Strategy for generation of inducing points (default: normal distribution).
    inducing_points_scale: float, optional.
        Scale of inducing points (default: 1.0).

    Other Parameters
    ----------------
    inducing_points_mean: float, optional.
        Initial mean of inducing values.
    inducing_points_var: float, optional.
        Initial variance of inducing values.
    inducing_points_learn_loc: bool, optional.
        Flag that indicates if inducing points location is learnable (default: True).
    inducing_points_learn_mean: bool, optional.
        Flag that indicates if inducing values mean is learnable (default: True).
    inducing_points_learn_var: bool, optional.
        Flag that indicates if inducing values variance is learnable (default: True).
    kernel_learn_outputscale: bool, optional.
        Flag that indicates if output scale is learnable (default: True).
    kernel_learn_lengthscale: bool, optional.
        Flag that indicates if lengthscale is learnable (default: True).


    Returns
    -------
    model: ModelList.
        List with GP Models.
    """
    ard_num_dims = ard_num_dims if ard_num_dims is not None else dim_states + dim_inputs

    mean = _parse_mean(mean_str, dim_states + dim_inputs)
    kernel = _parse_kernel(kernel_str, ard_num_dims, dim_states + dim_inputs,
                           kernel_outputscale, kernel_lengthscale,
                           kernel_learn_outputscale, kernel_learn_lengthscale)

    gps = []
    for _ in range(dim_states):
        if not shared:
            mean = _parse_mean(mean_str, dim_states + dim_inputs)
            kernel = _parse_kernel(kernel_str, ard_num_dims, dim_states + dim_inputs,
                                   kernel_outputscale, kernel_lengthscale,
                                   kernel_learn_outputscale, kernel_learn_lengthscale)

        inducing_points = get_inducing_points(inducing_points_number,
                                              dim_states + dim_inputs,
                                              inducing_points_strategy,
                                              inducing_points_scale)

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points_number,
        )
        ip_mean = inducing_points_mean if inducing_points_mean is not None else 1.
        variational_distribution.variational_mean = nn.Parameter(
            ip_mean * torch.ones(inducing_points_number), inducing_points_learn_mean)
        ip_var = inducing_points_var if inducing_points_var is not None else 1.

        variational_distribution.chol_variational_covar = nn.Parameter(
            ip_var * torch.eye(inducing_points_number), inducing_points_learn_var)

        gp = VariationalGP(inducing_points, mean, kernel, inducing_points_learn_loc,
                           variational_distribution)

        gps.append(gp)

    return ModelList(gps)


def _init_likelihood_list(num_models: int, initial_variance: float = None,
                          learnable: bool = True, shared: bool = False) -> list:
    """Initialize a list of likelihoods.

    Parameters
    ----------
    num_models: int.
        Number of likelihoods.
    initial_variance: float.
        Initial emission covariance estimate.
    learnable: bool.
        Flag that indicates if module is learnable.
    shared: bool.
        Flag that indicates if module parameters are shared between outputs.

    Returns
    -------
    emission: Emissions.
        Initialized emission module.
    """
    if shared:
        likelihood = GaussianLikelihood()
        if initial_variance is not None:
            likelihood.noise_covar.noise = initial_variance
        likelihood.raw_noise.requires_grad = learnable
        likelihoods = [likelihood for _ in range(num_models)]

    else:
        likelihoods = []
        for _ in range(num_models):
            likelihood = GaussianLikelihood()
            if initial_variance is not None:
                likelihood.noise_covar.noise = initial_variance
            likelihood.raw_noise.requires_grad = learnable
            likelihoods.append(likelihood)

    return likelihoods


def _parse_mean(mean: str, input_size: int = None) -> Mean:
    """Parse Mean string.

    Parameters
    ----------
    mean: str.
        String that identifies mean function.
    input_size: int.
        Size of input to GP (needed for linear mean functions).

    Returns
    -------
    mean: Mean.
        Mean function.
    """
    if mean.lower() == 'constant':
        mean_ = ConstantMean()
    elif mean.lower == 'zero':
        mean_ = ZeroMean()
    elif mean.lower == 'linear':
        mean_ = LinearMean(input_size=input_size)
    else:
        raise NotImplementedError('Mean function {} not implemented'.format(mean))
    return mean_


def _parse_kernel(kernel: str, ard_num_dims: int, input_size: int = None,
                  kernel_scale: float = None,
                  kernel_lengthscale: float = None,
                  kernel_learn_scale: bool = True,
                  kernel_learn_lengthscale: bool = True
                  ) -> Kernel:
    if kernel.lower() == 'rbf':
        kernel_ = ScaleKernel(RBFKernel(ard_num_dims=ard_num_dims))
    elif kernel.lower() == 'matern 1/2':
        kernel_ = ScaleKernel(MaternKernel(nu=0.5, ard_num_dims=ard_num_dims))
    elif kernel.lower() == 'matern 3/2':
        kernel_ = ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=ard_num_dims))
    elif kernel.lower() == 'matern 5/2':
        kernel_ = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=ard_num_dims))
    elif kernel.lower() == 'linear':
        kernel_ = ScaleKernel(LinearKernel(input_size=input_size,
                                           ard_num_dims=ard_num_dims))
    else:
        raise NotImplementedError('Mean function {} not implemented'.format(kernel))

    if kernel_scale is not None:
        kernel_.outputscale = kernel_scale
    kernel_.raw_outputscale.requires_grad = kernel_learn_scale

    if kernel_lengthscale is not None:
        lengthscale = [kernel_lengthscale] * ard_num_dims
        kernel_.base_kernel.lengthscale = torch.tensor(lengthscale)
    kernel_.base_kernel.raw_lengthscale.requires_grad = kernel_learn_lengthscale

    return kernel_

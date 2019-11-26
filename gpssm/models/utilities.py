"""Initializers for SSMVI models."""

import torch
import numpy as np
from gpssm.models.components.emissions import Emissions
from gpssm.models.components.transitions import Transitions
from gpssm.models.components.gp import VariationalGP
from gpssm.models.components.recognition_model import Recognition, OutputRecognition, \
    ZeroRecognition, NNRecognition, ConvRecognition, LSTMRecognition
from gpytorch.means import ConstantMean, ZeroMean, LinearMean, Mean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, LinearKernel, Kernel
from gpytorch.variational import CholeskyVariationalDistribution
from typing import Tuple

__author__ = 'Sebastian Curi'
__all__ = ['init_emissions', 'init_transitions', 'init_dynamics', 'init_recognition']


def init_recognition(dim_outputs: int, dim_inputs: int, dim_states: int,
                     kind: str = 'output', length: int = 1,
                     variance: float = 0.1, learnable: bool = True) -> Recognition:
    """Initialize Recognition module.

    Parameters
    ----------
    dim_outputs: int.
        Dimension of output space.
    dim_inputs: int.
        Dimension of input space.
    dim_states: int.
        Dimension of state space.
    kind: str.
        Type of recognition strategy.
    length: int.
        Length of recognition sequence.
    variance: float.
        Initial state covariance estimate.
    learnable: bool.
        Flag that indicates if module is learnable.

    Returns
    -------
    recognition: Recognition.

    """
    if kind.lower() == 'output':
        recognition = OutputRecognition(dim_outputs, dim_inputs, dim_states,
                                        length=length, variance=variance
                                        )  # type: Recognition
    elif kind.lower() == 'zero':
        recognition = ZeroRecognition(dim_outputs, dim_inputs, dim_states,
                                      length=length, variance=variance)
    elif kind.lower() == 'nn':
        recognition = NNRecognition(dim_outputs, dim_inputs, dim_states,
                                    length=length, variance=variance)
    elif kind.lower() == 'conv':
        recognition = ConvRecognition(dim_outputs, dim_inputs, dim_states,
                                      length=length, variance=variance)
    elif kind.lower() == 'lstm':
        recognition = LSTMRecognition(dim_outputs, dim_inputs, dim_states,
                                      length=length, variance=variance,
                                      bidirectional=False)
    elif kind.lower() == 'bi-lstm':
        recognition = LSTMRecognition(dim_outputs, dim_inputs, dim_states,
                                      length=length, variance=variance,
                                      bidirectional=True)
    else:
        raise NotImplementedError('Recognition module {} not implemented.'.format(kind))

    for param in recognition.parameters():
        param.requires_grad = learnable

    return recognition


def init_emissions(dim_outputs: int, variance: float = 1.0, learnable: bool = True,
                   ) -> Emissions:
    """Initialize emission model.

    Parameters
    ----------
    dim_outputs: int.
        Dimension of output space.
    variance: float.
        Initial emission covariance estimate.
    learnable: bool.
        Flag that indicates if module is learnable.

    Returns
    -------
    emission: Emissions.
        Initialized emission module.
    """
    return Emissions(dim_outputs, variance, learnable)


def init_transitions(dim_states: int, variance: float = 0.01, learnable: bool = True
                     ) -> Transitions:
    """Initialize transitions model.

    Parameters
    ----------
    dim_states: int.
        Dimension of state space.
    variance: float.
        Initial emission covariance estimate.
    learnable: bool.
        Flag that indicates if module is learnable.

    Returns
    -------
    transitions: Transitions.
        Initialized emission module.

    """
    return Transitions(dim_states, variance, learnable)


def init_dynamics(dim_inputs: int, dim_outputs: int, kernel: dict = None,
                  mean: dict = None, inducing_points: dict = None,
                  variational_distribution: dict = None) -> VariationalGP:
    """Initialize dynamical Model.

    # TODO: Implement other function approximations such as NN.

    Parameters
    ----------
    dim_inputs:
        Dimension of inputs to dynamical model.
        Forward model, usually dim_inputs + dim_states.
        Backward model, usually dim_inputs + dim_states.
    dim_outputs: int.
        Dimension of output to dynamical model.
        Forward model, usually dim_states.
        Backward model, usually dim_states - dim_outputs.
    kernel: dict.
        Dictionary with kernel parameters.
    mean: dict.
        Dictionary with mean parameters.
    inducing_points: dict.
        Dictionary with inducing points parameters.
    variational_distribution: dict.
        Dictionary with Variational Distribution parameters.

    Returns
    -------
    model: ModelList.
        List with GP Models.
    """
    mean = mean if mean is not None else dict()
    kernel = kernel if kernel is not None else dict()
    inducing_points = inducing_points if inducing_points is not None else dict()
    var_d = variational_distribution if variational_distribution is not None else dict()

    mean_ = _parse_mean(dim_inputs, dim_outputs, **mean)
    kernel_ = _parse_kernel(dim_inputs, dim_outputs, **kernel)
    ip, learn_ip = _parse_inducing_points(dim_inputs, dim_outputs, **inducing_points)
    var_dist = _parse_var_dist(ip.shape[1], dim_outputs, **var_d)

    return VariationalGP(ip, mean_, kernel_, learn_ip, var_dist)


def _parse_mean(input_size: int, dim_outputs: int = 1, kind: str = 'zero') -> Mean:
    """Parse Mean string.

    Parameters
    ----------
    input_size: int.
        Size of input to GP (needed for linear mean functions).
    kind: str.
        String that identifies mean function.

    Returns
    -------
    mean: Mean.
        Mean function.
    """
    if kind.lower() == 'constant':
        mean = ConstantMean()
    elif kind.lower() == 'zero':
        mean = ZeroMean()
    elif kind.lower() == 'linear':
        mean = LinearMean(input_size=input_size, batch_size=dim_outputs)
    else:
        raise NotImplementedError('Mean function {} not implemented.'.format(kind))
    return mean


def _parse_kernel(input_size: int, dim_outputs: int = 1, shared: bool = False,
                  kind: str = 'rbf', ard_num_dims: int = None,
                  outputscale: float = None, lengthscale: float = None,
                  learn_outputscale: bool = True, learn_lengthscale: bool = True
                  ) -> Kernel:
    batch_size = 1 if shared else dim_outputs
    ard_num_dims = ard_num_dims if ard_num_dims is not None else input_size
    if kind.lower() == 'rbf':
        kernel = ScaleKernel(RBFKernel(ard_num_dims=ard_num_dims,
                                       batch_size=batch_size),
                             batch_size=batch_size)
    elif kind.lower() == 'matern 1/2':
        kernel = ScaleKernel(MaternKernel(nu=0.5, ard_num_dims=ard_num_dims,
                                          batch_size=batch_size),
                             batch_size=batch_size)
    elif kind.lower() == 'matern 3/2':
        kernel = ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=ard_num_dims,
                                          batch_size=batch_size),
                             batch_size=batch_size)
    elif kind.lower() == 'matern 5/2':
        kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=ard_num_dims,
                                          batch_size=batch_size),
                             batch_size=batch_size)
    elif kind.lower() == 'linear':
        kernel = ScaleKernel(LinearKernel(input_size=input_size,
                                          ard_num_dims=ard_num_dims,
                                          batch_size=batch_size),
                             batch_size=batch_size)
    else:
        raise NotImplementedError('Kernel function {} not implemented.'.format(kind))

    if outputscale is not None:
        new_outputscale = outputscale * torch.ones(batch_size, 1)
        kernel.outputscale = new_outputscale
    kernel.raw_outputscale.requires_grad = learn_outputscale

    if lengthscale is not None:
        new_lengthscale = lengthscale * torch.ones(batch_size, ard_num_dims)
        kernel.base_kernel.lengthscale = new_lengthscale
    kernel.base_kernel.raw_lengthscale.requires_grad = learn_lengthscale

    return kernel


def _parse_inducing_points(dim_inputs: int, dim_outputs: int = 1,
                           number_points: int = 20,
                           strategy: str = 'normal', scale: float = 1,
                           learnable: bool = True) -> Tuple[torch.Tensor, bool]:
    """Initialize inducing points for variational GP.

    Parameters
    ----------
    dim_inputs: int.
        Input dimensionality.
    number_points: int.
        Number of inducing points.
    strategy: str, optional.
        Strategy to generate inducing points (by default normal).
        Either either normal, uniform or linspace.
    scale: float, optional.
        Scale of inducing points (default 1)

    Returns
    -------
    inducing_point: torch.Tensor.
        Inducing points with shape [dim_outputs x num_inducing_points x dim_inputs]
    learn_loc: bool.
        Flag that indicates if inducing points are learnable.

    Examples
    --------
    >>> num_points, dim_inputs, dim_outputs = 24, 8, 4
    >>> for strategy in ['normal', 'uniform', 'linspace']:
    ...     ip, l = _parse_inducing_points(dim_inputs, dim_outputs, num_points,
    ...     strategy, 2.)
    ...     assert type(ip) == torch.Tensor
    ...     assert ip.shape == torch.Size([dim_outputs, num_points, dim_inputs])
    ...     assert l
    """
    if strategy == 'normal':
        ip = scale * torch.randn((dim_outputs, number_points, dim_inputs))
    elif strategy == 'uniform':
        ip = scale * torch.rand((dim_outputs, number_points, dim_inputs)) - (scale / 2)
    elif strategy == 'linspace':
        lin_points = int(np.ceil(number_points ** (1 / dim_inputs)))
        ip = np.linspace(-scale, scale, lin_points)
        ip = np.array(np.meshgrid(*([ip] * dim_inputs))).reshape(dim_inputs, -1).T
        idx = np.random.choice(np.arange(ip.shape[0]), size=number_points,
                               replace=False)
        ip = torch.from_numpy(ip[idx]).float().repeat(dim_outputs, 1, 1)
    else:
        raise NotImplementedError("inducing point {} not implemented.".format(strategy))
    assert ip.shape == torch.Size([dim_outputs, number_points, dim_inputs])
    return ip, learnable


def _parse_var_dist(num_points: int, dim_outputs: int = 1,
                    mean: float = None, var: float = None,
                    learn_mean: bool = True, learn_var: bool = True
                    ) -> CholeskyVariationalDistribution:
    var_dist = CholeskyVariationalDistribution(num_inducing_points=num_points,
                                               batch_size=dim_outputs)

    if mean is not None:
        new_mean = mean * torch.ones(dim_outputs, num_points)
        var_dist.variational_mean.data = new_mean
    var_dist.variational_mean.requires_grad = learn_mean

    if var is not None:
        new_cov = np.sqrt(var) * torch.eye(num_points).repeat(dim_outputs, 1, 1)
        var_dist.chol_variational_covar.data = new_cov
    var_dist.chol_variational_covar.requires_grad = learn_var

    return var_dist

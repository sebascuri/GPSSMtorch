"""Utilities for SSMVI models."""

import torch
import numpy as np

__author__ = 'Sebastian Curi'
__all__ = ['get_inducing_points']


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

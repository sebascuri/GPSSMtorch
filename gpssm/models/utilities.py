"""Utilities for SSMVI models."""

import torch
import numpy as np


def get_inducing_points(num_inducing_points: int, dim_inputs: int, dim_outputs: int,
                        strategy: str = 'random') -> torch.Tensor:
    """Initialize inducing points for variational GP.

    Parameters
    ----------
    num_inducing_points: int.
        Number of inducing points.
    dim_inputs: int.
        Input dimensionality.
    dim_outputs: int.
        Output dimensionality.
    strategy: str, optional.
        Strategy to generate inducing points (either random, by default, or uniform).

    Returns
    -------
    inducing_point: torch.Tensor.
        Inducing points with shape [dim_outputs x num_inducing_points x dim_inputs]

    """

    if strategy == 'random':
        ip = torch.randn((dim_outputs, num_inducing_points, dim_inputs))
    elif strategy == 'uniform':
        ip = np.linspace(-1, 1, np.ceil(num_inducing_points ** (1 / dim_inputs)))
        ip = np.array(np.meshgrid(*([ip] * dim_inputs))).reshape(dim_inputs, -1).T
        ip = np.broadcast_to(ip, (dim_outputs, *ip.shape))

        aux = np.empty((0, num_inducing_points, dim_inputs))
        for i in range(dim_outputs):
            idx = np.random.choice(np.arange(ip.shape[1]), size=num_inducing_points, replace=False)
            aux = np.concatenate((aux, ip[i:(i+1), idx]))

        ip = torch.from_numpy(aux).float()
    else:
        raise ValueError("strategy {} not implemented.".format(strategy))
    assert ip.shape == torch.Size([dim_outputs, num_inducing_points, dim_inputs])
    return ip

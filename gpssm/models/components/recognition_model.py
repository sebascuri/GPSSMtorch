"""Recognition Model Template."""

from torch import Tensor
import torch
import torch.nn as nn
import numpy as np
from gpytorch.distributions import MultivariateNormal
import copy

__author__ = 'Sebastian Curi'
__all__ = ['Recognition', 'OutputRecognition']


class Recognition(nn.Module):
    """Base Class for recognition Module."""

    def __init__(self, dim_states: int, length: int) -> None:
        super().__init__()
        self.dim_states = dim_states
        self.length = length
        self.prior = MultivariateNormal(torch.zeros(dim_states),
                                        covariance_matrix=torch.eye(dim_states))

    def copy(self):
        """Copy recognition model."""
        return copy.deepcopy(self)

    def __str__(self) -> str:
        """Return recognition model parameters as a string."""
        raise NotImplementedError


class OutputRecognition(Recognition):
    """Recognition model based that uses the outputs of the first time step.

    Parameters
    ----------
    dim_states: int.
        Dimension of the state.

    variance: float, optional.
        Initial variance of the noise.

    Examples
    --------
    >>> from torch.testing import assert_allclose
    >>> recognition = OutputRecognition(4, variance=0.01)
    >>> assert_allclose(recognition.sd_noise, torch.ones(4) * 0.1)
    >>> output_seq = torch.randn(32, 8, 2)
    >>> input_seq = torch.randn(32, 8, 1)
    >>> x0 = recognition(output_seq, input_seq)
    >>> assert_allclose(x0.loc[:, :2], output_seq[:, 0])
    >>> cov = torch.diag(torch.ones(4) * 0.01).expand(32, 4, 4)
    >>> assert_allclose(x0.covariance_matrix, cov)
    >>> debug_str = str(recognition)
    >>> other = recognition.copy()
    >>> assert other is not recognition
    >>> assert type(other) == type(recognition)
    """

    def __init__(self, dim_states: int, length: int = 1, variance: float = 1.0) -> None:
        super().__init__(dim_states, length)
        self.sd_noise = nn.Parameter(torch.ones(self.dim_states) * np.sqrt(variance),
                                     requires_grad=True)

    def forward(self, output_sequence: Tensor,
                input_sequence: Tensor) -> MultivariateNormal:
        """Forward execution of the recognition model."""
        assert output_sequence.ndim == 3
        dim_outputs = output_sequence.shape[-1]
        batch_size = output_sequence.shape[0]

        loc = torch.zeros(batch_size, self.dim_states)
        loc[:, :dim_outputs] = output_sequence[:, 0]
        cov = torch.diag(self.sd_noise ** 2)
        cov = cov.expand(batch_size, *cov.shape)
        return MultivariateNormal(loc, covariance_matrix=cov)

    def __str__(self) -> str:
        """Return recognition model parameters as a string."""
        return 'covariance: {}'.format(self.sd_noise.detach() ** 2)

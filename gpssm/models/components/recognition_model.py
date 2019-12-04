"""Recognition Model Template."""
from abc import ABC

from torch import Tensor
import torch
import torch.nn as nn
import numpy as np
from gpytorch.distributions import MultivariateNormal
import copy
from typing import Any

__author__ = 'Sebastian Curi'
__all__ = ['Recognition', 'OutputRecognition', 'ZeroRecognition', 'NNRecognition',
           'ConvRecognition', 'LSTMRecognition']

EPS = 1e-6


class Recognition(nn.Module, ABC):
    """Base Class for recognition Module.

    Parameters
    ----------
    dim_outputs: int.
        Dimension of the outputs.
    dim_inputs: int.
        Dimension of the inputs.
    dim_states: int.
        Dimension of the state.
    length: int.
        Recognition length.
    """

    def __init__(self, dim_outputs: int, dim_inputs: int, dim_states: int, length: int,
                 ) -> None:
        super().__init__()
        self.dim_outputs = dim_outputs
        self.dim_inputs = dim_inputs
        self.dim_states = dim_states
        self.length = length

    def copy(self):
        """Copy recognition model."""
        return copy.deepcopy(self)

    def __str__(self) -> str:
        """Return recognition model parameters as a string."""
        return str([p for p in self.parameters()])


class OutputRecognition(Recognition):
    """Recognition model based that uses the outputs of the first time step.

    Parameters
    ----------
    dim_outputs: int.
        Dimension of the outputs.
    dim_inputs: int.
        Dimension of the inptus.
    dim_states: int.
        Dimension of the state.
    length: int.
        Recognition length.
    variance: float, optional.
        Initial variance of the noise.

    Examples
    --------
    >>> from torch.testing import assert_allclose
    >>> recognition = OutputRecognition(2, 1, 4, 1, variance=0.01)
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
    >>> assert type(other) is type(recognition)
    """

    def __init__(self, dim_outputs: int, dim_inputs: int, dim_states: int,
                 length: int = 1, variance: float = 1.0) -> None:
        super().__init__(dim_outputs, dim_inputs, dim_states, length)
        self.sd_noise = nn.Parameter(torch.ones(self.dim_states) * np.sqrt(variance),
                                     requires_grad=True)  # TODO: add learnable.

    def __str__(self) -> str:
        """Return recognition model parameters as a string."""
        return str(self.sd_noise.detach().numpy() ** 2)

    def forward(self, *inputs: Tensor, **kwargs) -> MultivariateNormal:
        """Forward execution of the recognition model."""
        output_sequence, input_sequence = inputs
        assert output_sequence.dim() == 3
        dim_outputs = output_sequence.shape[-1]
        batch_size = output_sequence.shape[0]

        loc = torch.zeros(batch_size, self.dim_states)
        loc[:, :dim_outputs] = output_sequence[:, 0]
        cov = torch.diag(self.sd_noise ** 2 + EPS)
        cov = cov.expand(batch_size, *cov.shape)
        return MultivariateNormal(loc, covariance_matrix=cov)


class ZeroRecognition(OutputRecognition):
    """Recognition model that predicts always a zero mean Multivariate Normal."""

    def forward(self, *inputs: Tensor, **kwargs) -> MultivariateNormal:
        """Forward execution of the recognition model."""
        output_sequence, _ = inputs

        batch_size = output_sequence.shape[0]
        loc = torch.zeros(batch_size, self.dim_states)
        cov = torch.diag(self.sd_noise ** 2 + EPS)
        cov = cov.expand(batch_size, *cov.shape)
        return MultivariateNormal(loc, covariance_matrix=cov)


class NNRecognition(Recognition):
    """Fully connected Recognition Module."""

    def __init__(self, dim_outputs: int, dim_inputs: int, dim_states: int, length: int,
                 variance: float = 1.0) -> None:
        super().__init__(dim_outputs, dim_inputs, dim_states, length)
        self.linear = nn.Linear(in_features=length * (dim_inputs + dim_outputs),
                                out_features=10)
        self.mean = nn.Linear(in_features=10, out_features=dim_states)
        self.var = nn.Linear(in_features=10, out_features=dim_states)
        self.var.bias = nn.Parameter(torch.ones(self.dim_states) * variance, True)

    def __str__(self) -> str:
        """Return recognition model parameters as a string."""
        return str(nn.functional.softplus(self.var.bias.detach()).numpy())

    def forward(self, *inputs: Tensor, **kwargs) -> MultivariateNormal:
        """Forward execution of the recognition model."""
        output_sequence, input_sequence = inputs

        batch_size = output_sequence.shape[0]

        # Reshape input/output sequence
        output_sequence = output_sequence[:, :self.length, :]
        input_sequence = input_sequence[:, :self.length, :]
        io_sequence = torch.cat((output_sequence, input_sequence), dim=-1)

        # Forward Propagate.
        x = io_sequence.view(batch_size, -1)
        x = torch.sigmoid(self.linear(x))

        return MultivariateNormal(self.mean(x), covariance_matrix=torch.diag_embed(
            nn.functional.softplus(self.var(x)) + EPS))


class ConvRecognition(Recognition):
    """Convolution Recognition Module."""

    def __init__(self, dim_outputs: int, dim_inputs: int, dim_states: int, length: int,
                 variance: float = 1.0) -> None:
        super().__init__(dim_outputs, dim_inputs, dim_states, length)
        self.conv1 = nn.Conv1d(in_channels=length, out_channels=16, kernel_size=2,
                               stride=1, padding=1)

        o = dim_inputs + dim_outputs  # type: Any
        o = int((o + 2 * 1 - 1 * (2 - 1) - 1) / 1 + 1)

        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        o = int((o + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32,
                               kernel_size=2, stride=2, padding=1)
        o = int((o + 2 * 1 - 1 * (2 - 1) - 1) / 2 + 1)
        self.max_pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        o = int((o + 2 * 1 - 1 * (2 - 1) - 1) / 2 + 1)
        self.mean = nn.Linear(in_features=32 * o, out_features=dim_states)
        self.var = nn.Linear(in_features=32 * o, out_features=dim_states)
        self.var.bias = nn.Parameter(torch.ones(self.dim_states) * variance, True)

    def __str__(self) -> str:
        """Return recognition model parameters as a string."""
        return str(nn.functional.softplus(self.var.bias.detach()).numpy())

    def forward(self, *inputs: Tensor, **kwargs) -> MultivariateNormal:
        """Forward execution of the recognition model."""
        output_sequence, input_sequence = inputs

        batch_size = output_sequence.shape[0]

        # Reshape input/output sequence
        output_sequence = output_sequence[:, :self.length, :]
        input_sequence = input_sequence[:, :self.length, :]
        io_sequence = torch.cat((output_sequence, input_sequence), dim=-1)

        # Forward Propagate.
        x = self.max_pool1(torch.relu(self.conv1(io_sequence)))
        x = self.max_pool2(torch.relu(self.conv2(x)))  # type: ignore
        x = x.view(batch_size, -1)  # type: ignore
        return MultivariateNormal(self.mean(x), covariance_matrix=torch.diag_embed(
            nn.functional.softplus(self.var(x)) + EPS))


class LSTMRecognition(Recognition):
    """LSTM Based Recognition."""

    def __init__(self, dim_outputs: int, dim_inputs: int, dim_states: int, length: int,
                 variance: float = 1.0, bidirectional: bool = True) -> None:
        super().__init__(dim_outputs, dim_inputs, dim_states, length)
        self.lstm = nn.LSTM(dim_inputs + dim_outputs, 10, batch_first=True,
                            bidirectional=bidirectional)
        in_features = 10 * (1+bidirectional)
        self.mean = nn.Linear(in_features=in_features, out_features=dim_states)
        self.var = nn.Linear(in_features=in_features, out_features=dim_states)
        self.var.bias = nn.Parameter(torch.ones(self.dim_states) * variance, True)

    def __str__(self) -> str:
        """Return recognition model parameters as a string."""
        return str(nn.functional.softplus(self.var.bias.detach()).numpy())

    def forward(self, *inputs: Tensor, **kwargs) -> MultivariateNormal:
        """Forward execution of the recognition model."""
        output_sequence, input_sequence = inputs

        batch_size = output_sequence.shape[0]

        # Reshape input/output sequence
        output_sequence = output_sequence[:, :self.length, :]
        input_sequence = input_sequence[:, :self.length, :]
        io_sequence = torch.cat((output_sequence, input_sequence), dim=-1)

        num_layers = self.lstm.num_layers * (1 + self.lstm.bidirectional)
        hidden = (torch.randn(num_layers, batch_size, self.lstm.hidden_size),
                  torch.randn(num_layers, batch_size, self.lstm.hidden_size))
        out, hidden = self.lstm(io_sequence, hidden)
        x = out[:, -1]
        return MultivariateNormal(self.mean(x), covariance_matrix=torch.diag_embed(
            nn.functional.softplus(self.var(x)) + EPS))

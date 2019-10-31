"""Base Class for System Id using Variational Inference with SSMs."""

from abc import ABC, abstractmethod
from torch import Tensor
from torch.nn import Module
from typing import Iterator


class SSMSVI(Module, ABC):
    """Abstract Base Class for Stochastic Variational Inference algorithms on SSMs."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def elbo(self, output_sequence: Tensor, input_sequence: Tensor = None,
             state_sequence: Tensor = None) -> Tensor:
        """Calculate the ELBO for the given output/input/state data.

        Parameters
        ----------
        output_sequence: Tensor.
            Tensor of output data [batch_size x sequence_length x dim_outputs].
        input_sequence: Tensor, optional.
            Tensor of input data, if any [batch_size x sequence_length x dim_inputs].
        state_sequence: Tensor, optional.
            Tensor of state data, if any [batch_size x sequence_length x dim_states].

        Returns
        -------
        elbo: Tensor.
            Differentiable tensor with ELBO of sequence.
        """
        raise NotImplementedError

    def forward(self, output_sequence: Tensor, input_sequence: Tensor = None,
                state_sequence: Tensor = None) -> Tensor:
        """See `self.elbo'."""
        return self.elbo(output_sequence, input_sequence, state_sequence)

    @abstractmethod
    def properties(self) -> Iterator:
        """Return list of learnable parameters."""
        raise NotImplementedError

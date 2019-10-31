"""Recognition Model Template."""

from torch import Tensor
import torch
import torch.nn as nn
from gpytorch.distributions import MultivariateNormal


class Recognition(nn.Module):
    """Base Class for recognition Module."""

    def __init__(self, dim_states) -> None:
        super().__init__()
        self.dim_states = dim_states
        self.prior = MultivariateNormal(torch.zeros(dim_states),
                                        covariance_matrix=torch.eye(dim_states))


class OutputRecognition(Recognition):
    """Recognition model based on the outputs of the first time-step."""

    def __init__(self, dim_states: int) -> None:
        super().__init__(dim_states)

    def forward(self, output_sequence: Tensor,
                input_sequence: Tensor) -> MultivariateNormal:
        """Forward execution of the recognition model."""
        dim_outputs = output_sequence.shape[-1]
        loc = torch.zeros(self.dim_states)
        loc[:dim_outputs] = output_sequence[0]

        return self.posterior

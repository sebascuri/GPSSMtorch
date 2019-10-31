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

    def copy(self):
        """Copy recognition model."""
        raise NotImplementedError


class OutputRecognition(Recognition):
    """Recognition model based on the outputs of the first time-step."""

    def __init__(self, dim_states: int, sd_noise: float = 1.0) -> None:
        super().__init__(dim_states)
        self.sd_noise = nn.Parameter(torch.ones(self.dim_states) * sd_noise,
                                     requires_grad=True)

    def forward(self, output_sequence: Tensor,
                input_sequence: Tensor) -> MultivariateNormal:
        """Forward execution of the recognition model."""
        dim_outputs = output_sequence.shape[-1]
        loc = torch.zeros(self.dim_states)
        loc[:dim_outputs] = output_sequence[0]

        return MultivariateNormal(loc, covariance_matrix=torch.diag(self.sd_noise ** 2))

    def copy(self) -> Recognition:
        """Copy recognition model."""
        return OutputRecognition(self.dim_states)

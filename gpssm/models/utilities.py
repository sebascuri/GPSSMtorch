"""Python Script Template."""

import torch
from torch.distributions import Normal
from gpytorch.distributions import MultivariateNormal
from typing import List

__author__ = 'Sebastian Curi'
__all__ = ['approximate_with_normal']


def approximate_with_normal(predicted_outputs: List[MultivariateNormal]) -> Normal:
    """Approximate a particle distribution with a Normal by moment matching."""
    sequence_length = len(predicted_outputs)
    dim_outputs, batch_size, _ = predicted_outputs[0].loc.shape

    output_loc = torch.zeros((batch_size, sequence_length, dim_outputs))
    output_cov = torch.zeros((batch_size, sequence_length, dim_outputs))
    for t, y_pred in enumerate(predicted_outputs):
        # Collapse particles!
        output_loc[:, t] = y_pred.loc.mean(dim=-1).t()
        output_cov[:, t] = y_pred.scale.mean(dim=-1).t()

    return Normal(output_loc, output_cov)

"""Python Script Template."""

import torch
from gpytorch.distributions import MultivariateNormal
from typing import List

__author__ = 'Sebastian Curi'
__all__ = ['approximate_with_normal']


def approximate_with_normal(predicted_outputs: List[List[MultivariateNormal]]
                            ) -> MultivariateNormal:
    """Approximate a particle distribution with a Normal by moment matching."""
    sequence_length = len(predicted_outputs)
    dim_outputs = len(predicted_outputs[0])
    batch_size = predicted_outputs[0][0].loc.shape[0]

    output_loc = torch.zeros((batch_size, sequence_length, dim_outputs))
    output_cov = torch.zeros((batch_size, sequence_length, dim_outputs, dim_outputs)
                             )
    for t in range(sequence_length):
        y_pred = predicted_outputs[t]
        # Collapse particles!
        for iy in range(dim_outputs):
            output_loc[:, t, iy] = y_pred[iy].loc.mean(dim=-1)
            output_cov[:, t, iy, iy] = y_pred[iy].scale.mean(dim=-1)

    return MultivariateNormal(output_loc, covariance_matrix=output_cov)

"""Base Class for System Id using Variational Inference with SSMs."""

from abc import ABC, abstractmethod
from torch import Tensor
import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Normal
from typing import List

__author__ = 'Sebastian Curi'
__all__ = ['SSMSVI']


class SSMSVI(nn.Module, ABC):
    """Abstract Base Class for Stochastic Variational Inference algorithms on SSMs."""

    @torch.jit.export
    def loss(self, predicted_outputs: List[Normal], output_sequence: Tensor,
             input_sequence: Tensor, key: str = None) -> Tensor:
        """Calculate the between the predicted and the true sequence.

        Parameters
        ----------
        predicted_outputs: List[Normal].
            List of predicted distributions [batch_size x num_particle x dim_outputs].
        output_sequence: Tensor.
            Tensor of output data of size [batch_size x sequence_length x dim_outputs].
        input_sequence: Tensor.
            Tensor of input data of size [batch_size x sequence_length x dim_inputs].
        key: str, optional.
            Key to identify the loss.

        Returns
        -------
        loss: Tensor.
            Differentiable loss tensor of sequence.
        """
        key = key if key is not None else 'elbo'
        batch_size, sequence_length, dim_outputs = output_sequence.shape
        log_lik = torch.tensor(0.)
        l2 = torch.tensor(0.)
        for t in range(sequence_length):
            # Output: Torch (dim_outputs)
            y = output_sequence[:, t]  # .expand(num_particles, batch_size, dim_outputs)
            # y = y.permute(1, 0, 2)
            assert y.shape == torch.Size([batch_size, dim_outputs])

            y_pred = predicted_outputs[t]
            ############################################################################
            # Calculate the Log-likelihood and L2-error #
            ############################################################################

            log_lik += y_pred.log_prob(y).mean()
            l2 += ((y_pred.loc - y) ** 2).mean()

        ################################################################################
        # Add KL Divergences #
        ################################################################################
        kl_u = self.forward_model.kl_divergence()  # type: ignore
        kl_x1 = kl_divergence(
            self.posterior_recognition(output_sequence, input_sequence),  # type: ignore
            self.prior_recognition(output_sequence, input_sequence)  # type: ignore
        ).mean()

        ################################################################################
        # Return different keys. #
        ################################################################################

        if key.lower() == 'log_likelihood':
            return -log_lik
        elif key.lower() == 'elbo':
            elbo = -(log_lik - kl_x1 - kl_u)
            return elbo
        elif key.lower() == 'l2':
            return l2
        elif key.lower() == 'rmse':
            return torch.sqrt(l2)
        elif key.lower() == 'elbo_separated':
            return log_lik, kl_x1, kl_u  # type: ignore
        else:
            raise NotImplementedError("Key {} not implemented".format(key))

    @abstractmethod
    def forward(self, *inputs: Tensor) -> List[Normal]:  # type: ignore
        """Forward propagate the model.

        Parameters
        ----------
        inputs: Tensor.
            output_sequence: Tensor.
            Tensor of output data [recognition_length x dim_outputs].

            input_sequence: Tensor.
            Tensor of input data [prediction_length x dim_inputs].

        Returns
        -------
        output_distribution: Normal.
            Normal of prediction_length x dim_outputs
        """
        raise NotImplementedError

    @abstractmethod
    def properties(self) -> list:
        """Return list of learnable parameters."""
        raise NotImplementedError

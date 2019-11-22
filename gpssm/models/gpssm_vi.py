"""Base Class for System Id using Variational Inference with SSMs."""
from abc import ABC, abstractmethod
from torch import Tensor
# from torch import Size
import torch
import torch.jit
import torch.nn as nn
from torch.distributions import kl_divergence, Normal
from typing import List
from gpytorch.distributions import MultivariateNormal

from .components.gp import ModelList
from .components.emissions import Emissions
from .components.transitions import Transitions
from .components.recognition_model import Recognition

__author__ = 'Sebastian Curi'
__all__ = ['GPSSM', 'PRSSM', 'CBFSSM']


class GPSSM(nn.Module, ABC):
    """Base class for GPSSMs inference.

    Parameters
    ----------
    forward_model: ModelList.
        Forwards Model.
    transitions: Transitions.
        Transition Model.
    emissions: Emissions.
        Emission Model.
    recognition_model: Recognition.
        Recognition Model.
    num_particles: int.
        Number of Particles for sampling.
    backward_model: ModelList.
        Backwards Model.
    cubic_sampling: bool.
        Cubic Sampling strategy
    loss_key: str.
        Key to select loss to return, default ELBO.
    loss_factors: dict.
        Factors to multiply each term of the ELBO with.

    # TODO: Implement cubic sampling.
    """

    def __init__(self,
                 forward_model: ModelList,
                 transitions: Transitions,
                 emissions: Emissions,
                 recognition_model: Recognition,
                 num_particles: int = 20,
                 backward_model: ModelList = None,
                 cubic_sampling: bool = False,
                 loss_key: str = 'elbo',
                 loss_factors: dict = None) -> None:
        super().__init__()
        self.dim_states = forward_model.num_outputs
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.transitions = transitions
        self.emissions = emissions

        self.prior_recognition = recognition_model.copy()
        self.posterior_recognition = recognition_model.copy()

        self.num_particles = num_particles
        self.cubic_sampling = cubic_sampling
        self.loss_key = loss_key
        if loss_factors is None:
            loss_factors = dict(loglik=1., kl_uf=1., kl_ub=1., kl_x=1., entropy=1.)
        self.loss_factors = loss_factors

    def __str__(self) -> str:
        """Return string of object with parameters."""
        string = "Model Parameters: \n\n"
        string += "Forward Model\n{}\n".format(self.forward_model)
        if self.backward_model is not None:
            string = "Backward Model\n{}\n".format(self.backward_model)
        string += "Emission {}\n\n".format(self.emissions)
        string += "Transition {}\n\n".format(self.transitions)
        string += "Prior x1 {}\n\n".format(self.prior_recognition)
        string += "Posterior x1 {}\n".format(self.posterior_recognition)

        return string

    def properties(self) -> list:
        """Return list of learnable parameters."""
        return [
            {'params': self.forward_model.parameters()},
            {'params': self.emissions.parameters()},
            {'params': self.transitions.parameters()},
            {'params': self.prior_recognition.parameters()},
            {'params': self.posterior_recognition.parameters()}
        ]

    def dump(self, file_name):
        """Dump current model parameters to a file."""
        with open(file_name, 'w') as file:
            file.write(str(self))

    @torch.jit.export
    def loss(self, predicted_outputs: List[Normal], output_sequence: Tensor,
             input_sequence: Tensor) -> Tensor:
        """Calculate the between the predicted and the true sequence.

        Parameters
        ----------
        predicted_outputs: List[Normal].
            List of predicted distributions [batch_size x num_particle x dim_outputs].
        output_sequence: Tensor.
            Tensor of output data of size [batch_size x sequence_length x dim_outputs].
        input_sequence: Tensor.
            Tensor of input data of size [batch_size x sequence_length x dim_inputs].

        Returns
        -------
        loss: Tensor.
            Differentiable loss tensor of sequence.
        """
        batch_size, sequence_length, dim_outputs = output_sequence.shape
        log_lik = torch.tensor(0.)
        l2 = torch.tensor(0.)
        for t in range(sequence_length):
            # Output: Torch (dim_outputs)
            y = output_sequence[:, t].expand(1, batch_size, dim_outputs)
            y = y.permute(1, 2, 0)
            # assert y.shape == torch.Size([batch_size, dim_outputs, 1])

            y_pred = predicted_outputs[t]
            ############################################################################
            # Calculate the Log-likelihood and L2-error #
            ############################################################################

            log_lik += y_pred.log_prob(y).mean()
            l2 += ((y_pred.loc - y) ** 2).mean()

        ################################################################################
        # Add KL Divergences #
        ################################################################################
        kl_uf = self.forward_model.kl_divergence() / sequence_length  # type: ignore
        kl_x1 = kl_divergence(
            self.posterior_recognition(output_sequence, input_sequence),
            self.prior_recognition(output_sequence, input_sequence)
        ).mean()

        ################################################################################
        # Return different keys. #
        ################################################################################

        if self.loss_key.lower() == 'loglik':
            return -log_lik
        elif self.loss_key.lower() == 'elbo':
            # TODO: add backwards pass losses.
            elbo = -(self.loss_factors['loglik'] * log_lik
                     - self.loss_factors['kl_x'] * kl_x1
                     - self.loss_factors['kl_uf'] * kl_uf)
            return elbo
        elif self.loss_key.lower() == 'l2':
            return l2
        elif self.loss_key.lower() == 'rmse':
            return torch.sqrt(l2)
        else:
            raise NotImplementedError("Key {} not implemented".format(self.loss_key))

    @torch.jit.export
    def forward(self, *inputs: Tensor, **kwargs) -> List[Normal]:
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
        output_distribution: List[List[MultivariateNormal].
            List of list of distributions [prediction_length x dim_outputs x
            batch_size x num_particles].
        """
        output_sequence, input_sequence = inputs
        num_particles = self.num_particles
        # dim_states = self.dim_states
        batch_size, sequence_length, dim_inputs = input_sequence.shape

        # Initial State: Tensor (batch_size x dim_states x num_particles)
        if self.training:
            state_d = self.posterior_recognition(output_sequence, input_sequence)
        else:
            state_d = self.prior_recognition(output_sequence, input_sequence)

        state = state_d.rsample(sample_shape=torch.Size([num_particles]))
        state = state.permute(1, 2, 0)  # Move last component to end.
        # assert state.shape == Size([batch_size, dim_states, num_particles])

        ############################################################################
        # SAMPLE GP for cubic sampling #
        ############################################################################
        # if self.cubic_sampling:
        #     # TODO: Change inducing points only (and inducing variables) :).
        #     forward_model = self.forward_model.sample_gp(
        #         self.transitions.likelihoods)  # type: ignore
        #     forward_model.eval()
        # else:
        #     forward_model = self.forward_model

        ############################################################################
        # PERFORM Backward Pass #
        ############################################################################
        if self.training:
            output_distribution = self._backward(output_sequence)

        ############################################################################
        # PREDICT Outputs #
        ############################################################################
        outputs = []
        y_pred = self.emissions(state)
        outputs.append(y_pred)

        for t in range(sequence_length - 1):
            ############################################################################
            # PREDICT Next State #
            ############################################################################

            # Input: Torch (batch_size x dim_inputs x num_particles)
            u = input_sequence[:, t].expand(num_particles, batch_size, dim_inputs)
            u = u.permute(1, 2, 0)  # Move last component to end.
            # assert u.shape == Size([batch_size, dim_inputs, num_particles])

            # \hat{X}: Torch (batch_size x dim_states + dim_inputs x num_particles)
            state_input = torch.cat((state, u), dim=1)
            # assert state_input.shape == Size(
            #     [batch_size, dim_inputs + dim_states, num_particles])

            # next_f: Multivariate Normal (batch_size x state_dim x num_particles)
            next_f = self.forward_model(state_input)
            # assert next_f.loc.shape == Size([batch_size, dim_states, num_particles])
            # assert next_f.covariance_matrix.shape == Size(
            #     [batch_size, dim_states, num_particles, num_particles])

            # next_state: Multivariate Normal (batch_size x dim_states x num_particles)
            next_state = self.transitions(next_f)
            # assert next_f.loc.shape == Size([batch_size, dim_states, num_particles])
            # assert next_f.covariance_matrix.shape == Size(
            #     [batch_size, dim_states, num_particles, num_particles])
            # assert (next_state.loc == next_f.loc).all()
            # assert not (next_state.covariance_matrix ==next_f.covariance_matrix).all()

            ############################################################################
            # UPDATE GP for cubic sampling #
            ############################################################################
            # if self.cubic_sampling:
            #     forward_model = forward_model.get_fantasy_model(state_input, next_f)
            # assert forward_model.num_outputs == dim_states

            ############################################################################
            # CONDITION Next State #
            ############################################################################
            if self.training:
                next_state = self._condition(next_state, output_distribution[t + 1])

            ############################################################################
            # RESAMPLE State #
            ############################################################################

            # state: Tensor (batch_size x dim_states x num_particles)
            state_d = next_state
            state = state_d.rsample()
            # assert state.shape == Size([batch_size, dim_states, num_particles])

            ############################################################################
            # PREDICT Outputs #
            ############################################################################

            y_pred = self.emissions(state)
            outputs.append(y_pred)

        assert len(outputs) == sequence_length
        return outputs

    @torch.jit.export
    def _backward(self, output_sequence: Tensor) -> List[Normal]:
        """Implement backwards pass."""
        batch_size, sequence_length, dim_outputs = output_sequence.shape
        dim_states = self.dim_states
        num_particles = self.num_particles
        dim_delta = dim_states - dim_outputs

        outputs = []
        for t in reversed(range(sequence_length)):
            y = output_sequence[:, t]
            y_ = self.emissions(y)
            # assert y_.loc.shape == Size([batch_size, dim_outputs])
            # assert y_.scale.shape == Size([batch_size, dim_outputs])

            loc = torch.cat((y_.loc, torch.zeros(batch_size, dim_delta)), dim=1)
            loc = loc.expand(num_particles, batch_size, dim_states).permute(1, 2, 0)
            # assert loc.shape == Size([batch_size, dim_states, num_particles])

            cov = torch.cat((y_.scale, torch.ones(batch_size, dim_delta)), dim=1)
            cov = cov.expand(num_particles, batch_size, dim_states).permute(1, 2, 0)
            # assert cov.shape == Size([batch_size, dim_states, num_particles])

            # TODO: IMPLEMENT BACKWARDS-PASS
            outputs.append(Normal(loc, cov))

        return outputs[::-1]

    @abstractmethod
    def _condition(self, next_x: MultivariateNormal, next_y: Normal
                   ) -> MultivariateNormal:
        """Implement conditioning."""
        raise NotImplementedError


class PRSSM(GPSSM):
    """Implementation of PR-SSM Algorithm."""

    @torch.jit.export
    def _condition(self, next_x: MultivariateNormal, next_y: Normal
                   ) -> MultivariateNormal:
        """Implement conditioning."""
        return next_x


class CBFSSM(GPSSM):
    """Conditional Backwards Forwards Algorithm."""

    @torch.jit.export
    def _condition(self, next_x: MultivariateNormal, next_y: Normal
                   ) -> MultivariateNormal:
        """Condition the next_x distribution with the measurements of next_y.

        Next_x is a Multivariate Normal and the covariance matrix is between particles.
        However, between the x-coordinates is independent.

        Next_y is a Normal distribution (diagonal Multivariate Normal) with the same
        dimensionality as next_x.

        In this case, the Kalman Filtering reduces to 1-D, where the gain is just:
        k = sigma_x / (sigma_x + sigma_y).
        The location of each particle is updated by k times the error.
        The covariance is the prior covariance scaled by 1 - k.

        Parameters
        ----------
        next_x: MultivariateNormal.
            Covariance state_dim x batch_size x num_particles x num_particles.
        next_y: Normal.
            Scale state_dim x batch_size x num_particles.

        Returns
        -------
        next_x: MultivariateNormal.
            Covariance state_dim x batch_size x num_particles x num_particles.
        """
        error = next_x.loc - next_y.loc

        sigma_f = torch.diagonal(next_x.covariance_matrix, dim1=-1, dim2=-2)
        sigma_y = next_y.scale
        gain = sigma_f / (sigma_f + sigma_y)
        neg_gain = torch.diag_embed(1 - gain)

        loc = next_x.loc + gain * error
        cov = torch.diag_embed(gain.pow(2) * sigma_y)
        cov += neg_gain @ next_x.covariance_matrix @ neg_gain
        return MultivariateNormal(loc, cov)

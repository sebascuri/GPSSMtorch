"""Base Class for System Id using Variational Inference with SSMs."""
from abc import ABC, abstractmethod
from torch import Tensor
from torch import Size
import torch
import torch.jit
import torch.nn as nn
from torch.distributions import kl_divergence, Normal
from typing import List, Tuple
from gpytorch.distributions import MultivariateNormal

from .components.dynamics import Dynamics, IdentityDynamics
from .components.emissions import Emissions
from .components.transitions import Transitions
from .components.recognition_model import Recognition

__author__ = 'Sebastian Curi'
__all__ = ['SSM', 'PRSSM', 'CBFSSM']


class SSM(nn.Module, ABC):
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
    k_factor: float.

    loss_factors: dict.
        Factors to multiply each term of the ELBO with.

    # TODO: Implement cubic sampling.
    """

    def __init__(self,
                 forward_model: Dynamics,
                 transitions: Transitions,
                 emissions: Emissions,
                 recognition_model: Recognition,
                 num_particles: int = 20,
                 backward_model: Dynamics = None,
                 cubic_sampling: bool = False,
                 loss_key: str = 'elbo',
                 k_factor: float = 1.,
                 loss_factors: dict = None) -> None:
        super().__init__()
        self.dim_states = forward_model.num_outputs
        self.forward_model = forward_model
        if backward_model is None:
            backward_model = IdentityDynamics(self.dim_states - emissions.dim_outputs)
        self.backward_model = backward_model

        self.transitions = transitions
        self.emissions = emissions

        self.prior_recognition = recognition_model.copy()
        self.posterior_recognition = recognition_model.copy()

        self.num_particles = num_particles
        self.cubic_sampling = cubic_sampling
        self.loss_key = loss_key
        self.k_factor = k_factor
        if loss_factors is None:
            loss_factors = dict(loglik=1., kl_uf=1., kl_ub=1., kl_x=1.,
                                kl_conditioning=1.0, entropy=0.)
        self.loss_factors = loss_factors

    def __str__(self) -> str:
        """Return string of object with parameters."""
        string = "Model Parameters: \n\n"
        string += "Forward Model\n{}\n".format(self.forward_model)
        string += "Backward Model\n{}\n".format(self.backward_model)
        string += "Emission {}\n\n".format(self.emissions)
        string += "Transition {}\n\n".format(self.transitions)
        string += "Prior x1 {}\n\n".format(self.prior_recognition)
        string += "Posterior x1 {}\n".format(self.posterior_recognition)

        return string

    def properties(self) -> list:
        """Return list of learnable parameters."""
        return [
            {'params': self.forward_model.parameters()},
            {'params': self.backward_model.parameters()},
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
    def forward(self, *inputs: Tensor, **kwargs) -> Tuple[List[Normal], Tensor]:
        """Forward propagate the model.

        Parameters
        ----------
        inputs: Tensor.
            output_sequence: Tensor.
            Tensor of output data [batch_size x sequence_length x dim_outputs].

            input_sequence: Tensor.
            Tensor of input data [batch_size x sequence_length x dim_inputs].

        Returns
        -------
        output_distribution: List[Normal].
            List of length sequence_length of distributions of size
            [batch_size x dim_outputs x num_particles]
        """
        output_sequence, input_sequence = inputs
        num_particles = self.num_particles
        # dim_states = self.dim_states
        batch_size, sequence_length, dim_inputs = input_sequence.shape
        _, _, dim_outputs = output_sequence.shape

        ################################################################################
        # SAMPLE GP for cubic sampling #
        ################################################################################

        ################################################################################
        # PERFORM Backward Pass #
        ################################################################################
        if self.training:
            output_distribution = self.backward(output_sequence, input_sequence)

        ################################################################################
        # Initial State #
        ################################################################################
        q_x1 = self.posterior_recognition(output_sequence, input_sequence)
        p_x1 = self.prior_recognition(output_sequence, input_sequence)
        kl_x1 = kl_divergence(q_x1, p_x1).mean()

        # Initial State: Tensor (batch_size x dim_states x num_particles)

        if self.training:
            state_d = q_x1
        else:
            state_d = p_x1

        state = state_d.rsample(sample_shape=Size([num_particles]))
        state = state.permute(1, 2, 0)  # Move last component to end.
        # assert state.shape == Size([batch_size, dim_states, num_particles])

        ################################################################################
        # PREDICT Outputs #
        ################################################################################
        outputs = []
        y_pred = self.emissions(state)
        outputs.append(y_pred)

        ################################################################################
        # INITIALIZE losses #
        ################################################################################
        log_lik = torch.tensor(0.)
        l2 = torch.tensor(0.)
        kl_conditioning = torch.tensor(0.)
        entropy = torch.tensor(0.)
        if self.training:
            y_pred = output_distribution[0]
            y = output_sequence[:, 0].expand(1, batch_size, dim_outputs
                                             ).permute(1, 2, 0)
            log_lik += y_pred.log_prob(y).mean()
            l2 += ((y_pred.loc - y) ** 2).mean()
            entropy += y_pred.entropy().mean()

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
            # next_f = self.forward_model(state_input)
            next_f = self.forward_model(state_input)

            # assert next_f.loc.shape == Size([batch_size, dim_states, num_particles])
            # assert next_f.covariance_matrix.shape == Size(
            #     [batch_size, dim_states, num_particles, num_particles])

            # next_state: Multivariate Normal (batch_size x dim_states x num_particles)
            next_state = self.transitions(next_f)
            # assert next_state.loc.shape == Size(
            #     [batch_size, dim_states, num_particles])
            # assert next_state.covariance_matrix.shape == Size(
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
                p_next_state = next_state
                next_state = self._condition(next_state, output_distribution[t + 1])
                kl_conditioning += kl_divergence(next_state, p_next_state).mean()

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

            ############################################################################
            # COMPUTE Losses #
            ############################################################################
            if self.training:
                y_tilde = output_distribution[t + 1]
                y = output_sequence[:, t + 1].expand(
                    num_particles, batch_size, dim_outputs).permute(1, 2, 0)

                log_lik += y_pred.log_prob(y).mean()
                l2 += ((y_pred.loc - y) ** 2).mean()
                entropy += y_tilde.entropy().mean() / sequence_length

        assert len(outputs) == sequence_length

        ################################################################################
        # Compute model KL divergences Divergences #
        ################################################################################
        kl_uf = self.forward_model.kl_divergence() / sequence_length
        kl_ub = self.backward_model.kl_divergence() / sequence_length

        if self.loss_key.lower() == 'loglik':
            loss = -log_lik
        elif self.loss_key.lower() == 'elbo':
            loss = -(self.loss_factors['loglik'] * log_lik
                     - self.loss_factors['kl_x'] * kl_x1
                     - self.loss_factors['kl_uf'] * kl_uf
                     - self.loss_factors['kl_ub'] * kl_ub
                     - self.loss_factors['kl_conditioning'] * kl_conditioning
                     + self.loss_factors['entropy'] * entropy
                     )
            print("""elbo: {}, log_lik: {}, kl_cond: {}""".format(
                loss.item(), log_lik.item(), kl_conditioning.item()))
        elif self.loss_key.lower() == 'l2':
            loss = l2
        elif self.loss_key.lower() == 'rmse':
            loss = torch.sqrt(l2)
        else:
            raise NotImplementedError("Key {} not implemented".format(self.loss_key))

        return outputs, loss

    @torch.jit.export
    def backward(self, *inputs: Tensor) -> List[Normal]:
        """Implement backwards pass."""
        output_sequence, input_sequence = inputs
        _, _, dim_inputs = input_sequence.shape
        batch_size, sequence_length, dim_outputs = output_sequence.shape
        dim_states = self.dim_states
        num_particles = self.num_particles
        dim_delta = dim_states - dim_outputs

        ################################################################################
        # Final Pseudo Measurement #
        ################################################################################
        y = output_sequence[:, -1].expand(num_particles, -1, -1).permute(1, 2, 0)
        y_ = self.emissions(y)
        # assert y_.loc.shape == Size([batch_size, dim_outputs, num_particles])
        # assert y_.scale.shape == Size([batch_size, dim_outputs, num_particles])

        loc = torch.cat((y_.loc,
                         torch.zeros(batch_size, dim_delta, num_particles)), dim=1)
        # assert loc.shape == Size([batch_size, dim_states, num_particles])

        scale = torch.cat((y_.scale,
                           torch.ones(batch_size, dim_delta, num_particles)), dim=1)
        # assert scale.shape == Size([batch_size, dim_states, num_particles])
        y_tilde_d = Normal(loc, scale)
        outputs = [y_tilde_d]

        y_tilde = y_tilde_d.rsample()
        # assert y_tilde.shape == Size([batch_size, dim_states, num_particles])

        for t in reversed(range(sequence_length - 1)):
            ############################################################################
            # PREDICT Previous pseudo-measurement #
            ############################################################################
            y = output_sequence[:, t].expand(num_particles, -1, -1).permute(1, 2, 0)
            y_ = self.emissions(y)
            # assert y_.loc.shape == Size([batch_size, dim_outputs, num_particles])
            # assert y_.scale.shape == Size([batch_size, dim_outputs, num_particles])

            # Input: Torch (batch_size x dim_inputs x num_particles)
            u = input_sequence[:, t].expand(num_particles, batch_size, dim_inputs)
            u = u.permute(1, 2, 0)  # Move last component to end.
            # assert u.shape == Size([batch_size, dim_inputs, num_particles])

            # \hat{Y}: Torch (batch_size x dim_states + dim_inputs x num_particles)

            # Here change the order of y_tilde for identity dynamics (those that return
            # the first dim_output states). The reason for this is that we already
            # append the y_ from emissions in the first components.
            # We can check this by comparing before computing next_y_tilde_d
            # loc[0, :, 0], y_.loc[0, :, 0], y_tilde[0, :, 0]

            idx = torch.cat((torch.arange(dim_outputs, dim_states),
                             torch.arange(dim_outputs)))
            y_tilde_input = torch.cat((y_tilde[:, idx], u), dim=1)
            # assert y_tilde_input.shape == Size(
            #     [batch_size, dim_inputs + dim_states, num_particles])

            next_y_tilde = self.backward_model(y_tilde_input)
            # assert next_y_tilde.loc.shape == Size(
            #     [batch_size, dim_delta, num_particles])
            # assert next_y_tilde.covariance_matrix.shape == Size(
            #     [batch_size, dim_delta, num_particles, num_particles])

            loc = torch.cat((y_.loc, next_y_tilde.loc), dim=1)
            # assert loc.shape == Size([batch_size, dim_states, num_particles])

            scale = torch.cat((y_.scale, torch.diagonal(next_y_tilde.covariance_matrix,
                                                        dim1=-1, dim2=-2)), dim=1)
            # assert scale.shape == Size([batch_size, dim_states, num_particles])

            q = self.transitions.diag_covariance.expand(batch_size, num_particles, -1)
            next_y_tilde_d = Normal(loc, scale + q.transpose(-1, -2))
            ############################################################################
            # RESAMPLE y_tilde #
            ############################################################################

            # state: Tensor (batch_size x dim_states x num_particles)
            y_tilde_d = next_y_tilde_d
            y_tilde = y_tilde_d.rsample()
            # assert y_tilde.shape == Size([batch_size, dim_states, num_particles])

            ############################################################################
            # PREDICT Outputs #
            ############################################################################
            outputs.append(next_y_tilde_d)

        assert len(outputs) == sequence_length
        return outputs[::-1]

    @abstractmethod
    def _condition(self, next_x: MultivariateNormal, next_y: Normal
                   ) -> MultivariateNormal:
        """Implement conditioning."""
        raise NotImplementedError


class PRSSM(SSM):
    """Implementation of PR-SSM Algorithm."""

    @torch.jit.export
    def _condition(self, next_x: MultivariateNormal, next_y: Normal
                   ) -> MultivariateNormal:
        """Implement conditioning."""
        return next_x


class CBFSSM(SSM):
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
            Covariance batch_size x state_dim x num_particles x num_particles.
        next_y: Normal.
            Scale batch_size x state_dim x num_particles.

        Returns
        -------
        next_x: MultivariateNormal.
            Covariance batch_size x state_dim x num_particles x num_particles.
        """
        error = next_x.loc - next_y.loc

        sigma_f = torch.diagonal(next_x.covariance_matrix, dim1=-1, dim2=-2)
        sigma_y = next_y.scale + (self.k_factor - 1) * sigma_f

        gain = sigma_f / (sigma_f + sigma_y)
        neg_gain = torch.diag_embed(1 - gain)

        loc = next_x.loc + gain * error

        cov = torch.diag_embed(gain.pow(2) * sigma_y)
        cov += neg_gain.transpose(-2, -1) @ next_x.covariance_matrix @ neg_gain
        return MultivariateNormal(loc, cov)

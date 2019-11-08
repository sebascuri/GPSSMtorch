"""Implementation of PR-SSM algorithm."""

from gpytorch.distributions import MultivariateNormal
from .components.gp import ModelList
from .components.emissions import Emissions
from .components.transitions import Transitions
from .components.recognition_model import Recognition
from .ssm_vi import SSMSVI
import torch
import torch.jit
from torch import Tensor
from torch.distributions import kl_divergence
from typing import List

__author__ = 'Sebastian Curi'
__all__ = ['PRSSM']


class PRSSM(SSMSVI):
    """Implementation of PR-SSM algorithm."""

    def __init__(self,
                 gp_model: ModelList,
                 transitions: Transitions,
                 emissions: Emissions,
                 recognition_model: Recognition,
                 num_particles: int = 32
                 ) -> None:
        super().__init__()
        self.dim_states = gp_model.num_outputs
        self.gp = gp_model
        self.transitions = transitions
        self.emissions = emissions

        self.prior_recognition = recognition_model.copy()
        self.posterior_recognition = recognition_model.copy()

        self.num_particles = num_particles

    def properties(self) -> list:
        """Return list of learnable parameters."""
        return [
            {'params': self.gp.parameters()},
            {'params': self.emissions.parameters()},
            {'params': self.transitions.parameters()},
            {'params': self.prior_recognition.parameters()},
            {'params': self.posterior_recognition.parameters()}
        ]

    def __str__(self) -> str:
        """Return string of object with parameters."""
        string = "PRSSM Parameters: \n\n"
        string += "GP {}\n".format(self.gp)
        string += "Emission {}\n".format(self.emissions)
        string += "Transition {}\n".format(self.transitions)
        string += "Prior x1 {}\n".format(self.prior_recognition)
        string += "Posterior x1 {}\n".format(self.posterior_recognition)

        return string

    @torch.jit.export
    def loss(self, output_sequence: Tensor, input_sequence: Tensor = None,
             state_sequence: Tensor = None, key: str = None) -> Tensor:
        """Calculate the loss for the given output/input/state data.

        Parameters
        ----------
        output_sequence: Tensor.
            Tensor of output data [batch_size x sequence_length x dim_outputs].
        input_sequence: Tensor, optional.
            Tensor of input data, if any [batch_size x sequence_length x dim_inputs].
        state_sequence: Tensor, optional.
            Tensor of state data, if any [batch_size x sequence_length x dim_states].
        key: str, optional.
            Key to identify the loss (default = ELBO).

        Returns
        -------
        loss: Tensor.
            Differentiable loss tensor of sequence.
        """
        key = key if key is not None else 'elbo'
        batch_size, sequence_length, dim_outputs = output_sequence.shape

        # Calculate x1 from prior and posterior
        qx1 = self.posterior_recognition(output_sequence, input_sequence)
        px1 = self.prior_recognition(output_sequence, input_sequence)

        predicted_outputs = self._loop(qx1, input_sequence)

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

            for iy in range(dim_outputs):
                # call mean() to average the losses from different batches.
                log_lik += y_pred[iy].log_prob(y[..., iy:(iy + 1)]).mean() / dim_outputs
                l2 += ((y_pred[iy].loc - y[..., iy:(iy + 1)]) ** 2).mean() / dim_outputs

        ################################################################################
        # Add KL Divergences #
        ################################################################################
        kl_u = self.gp.kl_divergence()
        kl_x1 = kl_divergence(qx1, px1).mean()

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
            return log_lik, kl_x1, kl_u
        else:
            raise NotImplementedError("Key {} not implemented".format(key))

    @torch.jit.export
    def forward(self, output_sequence: Tensor, input_sequence: Tensor
                ) -> MultivariateNormal:
        """Forward propagate the model.

        Parameters
        ----------
        output_sequence: Tensor.
            Tensor of output data [batch_size x recognition_length x dim_outputs].

        input_sequence: Tensor.
            Tensor of input data [batch_size x prediction_length x dim_inputs].

        Returns
        -------
        output_distribution: MultivariateNormal.
            MultivariateNormal of prediction_length x dim_outputs
        """
        batch_size, recognition_length, dim_outputs = output_sequence.shape
        _, sequence_length, dim_inputs = input_sequence.shape

        # state_d: Multivariate Normal (batch_size x dim_states)
        px1 = self.prior_recognition(output_sequence, input_sequence)

        predicted_outputs = self._loop(px1, input_sequence)

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

    def _loop(self, initial_state_distribution: MultivariateNormal,
              input_sequence: Tensor) -> List[List[MultivariateNormal]]:
        """Generate the output predictions."""
        num_particles = self.num_particles
        batch_size, sequence_length, dim_inputs = input_sequence.shape

        # Initial State: Tensor (batch_size x num_particles x dim_states)
        state_d = initial_state_distribution
        state = state_d.rsample(sample_shape=torch.Size([num_particles]))
        state = state.permute(1, 0, 2)
        assert state.shape == torch.Size([batch_size, num_particles, self.dim_states])

        outputs = []
        for t in range(sequence_length):
            ############################################################################
            # Calculate the Outputs #
            ############################################################################

            y_pred = self.emissions(state)
            outputs.append(y_pred)

            ############################################################################
            # Calculate the Next State #
            ############################################################################

            # Input: Torch (batch_size x num_particles x dim_inputs)
            u = input_sequence[:, t].expand(num_particles, batch_size, dim_inputs)
            u = u.permute(1, 0, 2)
            assert u.shape == torch.Size([batch_size, num_particles, dim_inputs])

            # \hat{X}: Torch (batch_size x num_particles x dim_states + dim_inputs)
            state_input = torch.cat((state, u), dim=-1)
            assert state_input.shape == torch.Size(
                [batch_size, num_particles, dim_inputs + self.dim_states])

            # next_f: Multivariate Normal (batch_size x state_dim x num_particles)
            next_f = self.gp(state_input)

            for ix in range(self.dim_states):
                assert next_f[ix].loc.shape == torch.Size([batch_size, num_particles])
                assert next_f[ix].covariance_matrix.shape == torch.Size(
                    [batch_size, num_particles, num_particles])

            # next_state: Multivariate Normal(state_dim x num_particles)
            next_state = self.transitions(next_f)
            for ix in range(self.dim_states):
                assert next_state[ix].loc.shape == torch.Size(
                    [batch_size, num_particles])
                assert next_state[ix].covariance_matrix.shape == torch.Size(
                    [batch_size, num_particles, num_particles])

                assert (next_state[ix].loc == next_f[ix].loc).all()
                assert not (next_state[ix].covariance_matrix == next_f[
                    ix].covariance_matrix).all()

            ############################################################################
            # Resample State #
            ############################################################################

            # next_state: Tensor (num_particles x state_dim)
            # state_d: Multivariate Normal (dim_states)
            # state: Tensor (dim_states x num_particles)
            state_d = next_state
            state = torch.zeros((batch_size, num_particles, self.dim_states))
            for ix in range(self.dim_states):
                state[:, :, ix] = state_d[ix].rsample()

        return outputs


class PRSSMCubic(PRSSM):
    """Implementation of PR-SSM algorithm with cubic resampling."""

    def _loop(self, initial_state_distribution: MultivariateNormal,
              input_sequence: Tensor) -> List[List[MultivariateNormal]]:
        """Generate the output predictions."""
        num_particles = self.num_particles
        batch_size, sequence_length, dim_inputs = input_sequence.shape

        # Initial State: Tensor (batch_size x num_particles x dim_states)
        state_d = initial_state_distribution
        state = state_d.rsample(sample_shape=torch.Size([num_particles]))
        state = state.permute(1, 0, 2)
        assert state.shape == torch.Size([batch_size, num_particles, self.dim_states])

        outputs = []
        sampled_gp = self.gp.sample_gp(self.transitions.likelihoods)
        sampled_gp.eval()
        for t in range(sequence_length):
            ############################################################################
            # Calculate the Outputs #
            ############################################################################

            y_pred = self.emissions(state)
            outputs.append(y_pred)

            ############################################################################
            # Calculate the Next State #
            ############################################################################

            # Input: Torch (batch_size x num_particles x dim_inputs)
            u = input_sequence[:, t].expand(num_particles, batch_size, dim_inputs)
            u = u.permute(1, 0, 2)
            assert u.shape == torch.Size([batch_size, num_particles, dim_inputs])

            # \hat{X}: Torch (batch_size x num_particles x dim_states + dim_inputs)
            state_input = torch.cat((state, u), dim=-1)
            assert state_input.shape == torch.Size(
                [batch_size, num_particles, dim_inputs + self.dim_states])

            # next_f: Multivariate Normal (batch_size x state_dim x num_particles)
            next_f = sampled_gp(state_input)

            for ix in range(self.dim_states):
                assert next_f[ix].loc.shape == torch.Size([batch_size, num_particles])
                assert next_f[ix].covariance_matrix.shape == torch.Size(
                    [batch_size, num_particles, num_particles])

            # next_state: Multivariate Normal(state_dim x num_particles)
            next_state = self.transitions(next_f)
            for ix in range(self.dim_states):
                assert next_state[ix].loc.shape == torch.Size(
                    [batch_size, num_particles])
                assert next_state[ix].covariance_matrix.shape == torch.Size(
                    [batch_size, num_particles, num_particles])

                assert (next_state[ix].loc == next_f[ix].loc).all()
                assert not (next_state[ix].covariance_matrix == next_f[
                    ix].covariance_matrix).all()

            ############################################################################
            # Update GP #
            ############################################################################
            sampled_gp = sampled_gp.get_fantasy_model(state_input, next_f)
            assert sampled_gp.num_outputs == self.dim_states
            ############################################################################
            # Resample State #
            ############################################################################

            # next_state: Tensor (num_particles x state_dim)
            # state_d: Multivariate Normal (dim_states)
            # state: Tensor (dim_states x num_particles)
            state_d = next_state
            state = torch.zeros((batch_size, num_particles, self.dim_states))
            for ix in range(self.dim_states):
                state[:, :, ix] = state_d[ix].rsample()

        return outputs

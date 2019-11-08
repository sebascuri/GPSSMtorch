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
                 num_particles: int = 32,
                 cubic_sampling: bool = False
                 ) -> None:
        super().__init__()
        self.dim_states = gp_model.num_outputs
        self.gp = gp_model
        self.transitions = transitions
        self.emissions = emissions

        self.prior_recognition = recognition_model.copy()
        self.posterior_recognition = recognition_model.copy()

        self.num_particles = num_particles
        self.cubic_sampling = cubic_sampling

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
    def forward(self, output_sequence: Tensor, input_sequence: Tensor
                ) -> List[List[MultivariateNormal]]:
        """Forward propagate the model.

        Parameters
        ----------
        output_sequence: Tensor.
            Tensor of output data [batch_size x recognition_length x dim_outputs].

        input_sequence: Tensor.
            Tensor of input data [batch_size x prediction_length x dim_inputs].

        Returns
        -------
        output_distribution: List[List[MultivariateNormal].
            List of list of distributions [prediction_length x dim_outputs x
            batch_size x num_particles].
        """
        num_particles = self.num_particles
        batch_size, sequence_length, dim_inputs = input_sequence.shape

        # Initial State: Tensor (batch_size x num_particles x dim_states)
        if self.training:
            state_d = self.posterior_recognition(output_sequence, input_sequence)
        else:
            print('evaluating')
            state_d = self.prior_recognition(output_sequence, input_sequence)

        state = state_d.rsample(sample_shape=torch.Size([num_particles]))
        state = state.permute(1, 0, 2)
        assert state.shape == torch.Size([batch_size, num_particles, self.dim_states])

        outputs = []
        if self.cubic_sampling:
            # TODO: Change inducing points only (and inducing variables) :).
            gp_model = self.gp.sample_gp(self.transitions.likelihoods)
            gp_model.eval()
        else:
            gp_model = self.gp

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
            next_f = gp_model(state_input)

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
            # Update GP (ONLY IN CUBIC SAMPLING SCHEME)#
            ############################################################################
            if self.cubic_sampling:
                gp_model = gp_model.get_fantasy_model(state_input, next_f)
            assert gp_model.num_outputs == self.dim_states

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

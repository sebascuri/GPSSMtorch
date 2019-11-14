"""Implementation of PR-SSM algorithm."""

from .components.gp import ModelList
from .components.emissions import Emissions
from .components.transitions import Transitions
from .components.recognition_model import Recognition
from .ssm_vi import SSMSVI
import torch
import torch.jit
from torch import Tensor, Size
from torch.distributions import Normal
from typing import List

__author__ = 'Sebastian Curi'
__all__ = ['PRSSM']


class PRSSM(SSMSVI):
    """Implementation of PR-SSM algorithm."""

    def __init__(self,
                 forward_model: ModelList,
                 transitions: Transitions,
                 emissions: Emissions,
                 recognition_model: Recognition,
                 num_particles: int = 32,
                 cubic_sampling: bool = False
                 ) -> None:
        super().__init__(forward_model, transitions, emissions, recognition_model)
        self.num_particles = num_particles
        self.cubic_sampling = cubic_sampling

    def __str__(self) -> str:
        """Return string of object with parameters."""
        string = "PRSSM Parameters: \n\n"
        string += "GP {}\n".format(self.forward_model)
        string += "Emission {}\n".format(self.emissions)
        string += "Transition {}\n".format(self.transitions)
        string += "Prior x1 {}\n".format(self.prior_recognition)
        string += "Posterior x1 {}\n".format(self.posterior_recognition)

        return string

    @torch.jit.export
    def forward(self, *inputs: Tensor) -> List[Normal]:
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
        dim_states = self.dim_states
        batch_size, sequence_length, dim_inputs = input_sequence.shape

        # Initial State: Tensor (batch_size x num_particles x dim_states)
        if self.training:
            state_d = self.posterior_recognition(output_sequence, input_sequence)
        else:
            print('evaluating')
            state_d = self.prior_recognition(output_sequence, input_sequence)

        state = state_d.rsample(sample_shape=torch.Size([num_particles]))
        state = state.permute(1, 0, 2)
        assert state.shape == Size([batch_size, num_particles, dim_states])

        outputs = []
        # if self.cubic_sampling:
        #     # TODO: Change inducing points only (and inducing variables) :).
        #     forward_model = self.forward_model.sample_gp(
        #         self.transitions.likelihoods)  # type: ignore
        #     forward_model.eval()
        # else:
        #     forward_model = self.forward_model

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
            assert u.shape == Size([batch_size, num_particles, dim_inputs])

            # \hat{X}: Torch (batch_size x num_particles x dim_states + dim_inputs)
            state_input = torch.cat((state, u), dim=-1)
            assert state_input.shape == Size(
                [batch_size, num_particles, dim_inputs + dim_states])

            # next_f: Multivariate Normal (state_dim x batch_size x num_particles)
            next_f = self.forward_model(state_input)
            assert next_f.loc.shape == Size([dim_states, batch_size, num_particles])
            assert next_f.covariance_matrix.shape == Size(
                [dim_states, batch_size, num_particles, num_particles])

            # next_state: Multivariate Normal (state_dim x batch_size x num_particles)
            next_state = self.transitions(next_f)
            assert next_f.loc.shape == Size([dim_states, batch_size, num_particles])
            assert next_f.covariance_matrix.shape == Size(
                [dim_states, batch_size, num_particles, num_particles])
            assert (next_state.loc == next_f.loc).all()
            assert not (next_state.covariance_matrix == next_f.covariance_matrix).all()

            # ############################################################################
            # # Update GP (ONLY IN CUBIC SAMPLING SCHEME)#
            # ############################################################################
            # if self.cubic_sampling:
            #     forward_model = forward_model.get_fantasy_model(state_input, next_f)
            # assert forward_model.num_outputs == dim_states

            ############################################################################
            # Resample State #
            ############################################################################

            # next_state: Tensor (num_particles x state_dim)
            # state_d: Multivariate Normal (dim_states)
            # state: Tensor (dim_states x num_particles)
            state_d = next_state
            state = state_d.rsample().permute(1, 2, 0)
            assert state.shape == Size([batch_size, num_particles, dim_states])

        return outputs

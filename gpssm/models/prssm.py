"""Implementation of PR-SSM algorithm."""

from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import Likelihood
from .components.gp import VariationalGP
from .components.emissions import Emission
from .components.recognition_model import Recognition
from .ssm_vi import SSMSVI
import torch
import torch.jit
from torch import Tensor
from torch.distributions import kl_divergence


class PRSSM(SSMSVI):
    """Implementation of PR-SSM algorithm."""

    def __init__(self,
                 gp_model: VariationalGP,
                 transitions: Likelihood,
                 emissions: Emission,
                 recognition_model: Recognition,
                 loss_factors: list = None,
                 num_particles: int = 32
                 ) -> None:
        super().__init__()
        self.dim_states = gp_model.num_outputs
        self.gp = gp_model
        self.transitions = transitions
        self.emissions = emissions

        self.prior_recognition = recognition_model
        self.posterior_recognition = recognition_model.copy()

        self.loss_factors = loss_factors if loss_factors is not None else [1., 1.]
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

    @torch.jit.export
    def elbo(self, output_sequence: Tensor, input_sequence: Tensor = None,
             state_sequence: Tensor = None) -> Tensor:
        """Calculate the ELBO for the given output/input/state data.

        Parameters
        ----------
        output_sequence: Tensor.
            Tensor of output data [sequence_length x dim_outputs].
        input_sequence: Tensor, optional.
            Tensor of input data, if any [sequence_length x dim_inputs].
        state_sequence: Tensor, optional.
            Tensor of state data, if any [sequence_length x dim_states].

        Returns
        -------
        elbo: Tensor.
            Differentiable tensor with ELBO of sequence.
        """
        num_particles = self.num_particles
        dim_outputs = output_sequence.shape[-1]
        dim_inputs = input_sequence.shape[-1]

        sequence_length = output_sequence.shape[-2]

        # Calculate x1 from prior and posterior
        qx1 = self.posterior_recognition(output_sequence, input_sequence)
        px1 = self.prior_recognition(output_sequence, input_sequence)

        # Initial State: Tensor (num_particles x dim_states)
        state = qx1.rsample(sample_shape=torch.Size([num_particles]))
        assert state.shape == torch.Size([num_particles, self.dim_states])

        log_lik = torch.tensor(0.)
        for t in range(sequence_length):
            ############################################################################
            # Generate the Samples #
            ############################################################################
            # Input: Torch (num_particles x dim_inputs)
            u = input_sequence[t].expand(num_particles, dim_inputs)
            assert u.shape == torch.Size([num_particles, dim_inputs])

            # \hat{X}: Torch (num_particles x dim_states + dim_inputs)
            state_input = torch.cat((state, u), dim=-1)
            assert state_input.shape == torch.Size([num_particles,
                                                    dim_inputs + self.dim_states])

            # next_f: Multivariate Normal (state_dim x num_particles)
            next_f = self.gp(state_input)
            assert next_f.loc.shape == torch.Size([self.dim_states, num_particles])
            assert next_f.covariance_matrix.shape == torch.Size([self.dim_states,
                                                                 num_particles,
                                                                 num_particles])

            # next_state: Multivariate Normal(state_dim x num_particles)
            next_state = self.transitions(next_f)
            assert next_state.loc.shape == torch.Size([self.dim_states, num_particles])
            assert next_state.covariance_matrix.shape == torch.Size([self.dim_states,
                                                                     num_particles,
                                                                     num_particles])

            assert (next_state.loc == next_f.loc).all()
            assert not (next_state.covariance_matrix == next_f.covariance_matrix).all()

            # Output: Torch (dim_outputs)
            output = output_sequence[t]
            assert output.shape == torch.Size([dim_outputs])

            ############################################################################
            # Calculate the Log-likelihood #
            ############################################################################

            # Log-likelihood
            # Log-likelihood
            y_pred = self.emissions(state)
            log_lik += y_pred.log_prob(output).mean()

            ############################################################################
            # Next State #
            ############################################################################

            # next_state: Tensor (num_particles x state_dim)
            # state_d: Multivariate Normal (dim_states)
            # state: Tensor (dim_states x num_particles)
            state_d = next_state
            state = state_d.rsample().t()

        ################################################################################
        # Add KL Divergences #
        ################################################################################

        # There is 1 gp per dimension hence the sum.
        kl_u = self.gp.variational_strategy.kl_divergence().sum()
        kl_x1 = kl_divergence(qx1, px1)

        elbo = -(log_lik * self.loss_factors[0] / sequence_length
                 - kl_x1 * self.loss_factors[0] / sequence_length
                 - kl_u
                 )
        return elbo

    @torch.jit.export
    def forward(self, output_sequence: Tensor, input_sequence: Tensor = None
                ) -> MultivariateNormal:
        """Forward propagate the model.

        Parameters
        ----------
        output_sequence: Tensor.
            Tensor of output data [recognition_length x dim_outputs].

        input_sequence: Tensor.
            Tensor of input data [prediction_length x dim_inputs].

        Returns
        -------
        output_distribution: MultivariateNormal.
            MultivariateNormal of prediction_length x dim_outputs
        """
        num_particles = self.num_particles

        recognition_length, dim_outputs = output_sequence.shape
        sequence_length, dim_inputs = input_sequence.shape

        # state_d: Multivariate Normal (dim_states)
        px1 = self.prior_recognition(output_sequence, input_sequence)
        state_d = px1

        # Initial State: Tensor (num_particles x dim_states)
        state = state_d.sample(sample_shape=torch.Size([num_particles]))
        assert state.shape == torch.Size([num_particles, self.dim_states])

        output_loc = torch.empty((0, dim_outputs))
        output_covariance = torch.empty((0, dim_outputs, dim_outputs))

        for t in range(sequence_length):
            ############################################################################
            # Generate the Samples #
            ############################################################################

            # Input: Torch (num_particles x dim_inputs)
            u = input_sequence[t].expand(num_particles, dim_inputs)
            assert u.shape == torch.Size([num_particles, dim_inputs])

            # \hat{X}: Torch (num_particles x dim_states + dim_inputs)
            state_input = torch.cat((state, u), dim=-1)
            assert state_input.shape == torch.Size([num_particles,
                                                    dim_inputs + self.dim_states])

            # next_f: Multivariate Normal (state_dim x num_particles)
            next_f = self.gp(state_input)
            assert next_f.loc.shape == torch.Size([self.dim_states, num_particles])
            assert next_f.covariance_matrix.shape == torch.Size([self.dim_states,
                                                                 num_particles,
                                                                 num_particles])

            # next_state: Multivariate Normal(state_dim x num_particles)
            next_state = self.transitions(next_f)
            assert next_state.loc.shape == torch.Size([self.dim_states, num_particles])
            assert next_state.covariance_matrix.shape == torch.Size([self.dim_states,
                                                                     num_particles,
                                                                     num_particles])

            assert (next_state.loc == next_f.loc).all()
            assert not (next_state.covariance_matrix == next_f.covariance_matrix).all()

            # Output: Torch (dim_outputs)
            output = self.emissions(state_d)
            assert output.loc.shape == torch.Size([dim_outputs])
            assert output.covariance_matrix.shape == torch.Size([dim_outputs,
                                                                 dim_outputs])

            output_loc = torch.cat((output_loc, output.loc.unsqueeze(0)), dim=0)
            output_covariance = torch.cat((output_covariance,
                                           output.covariance_matrix.unsqueeze(0)),
                                          dim=0)

            # next_state: Tensor(num_particles x state_dim)
            # state_d: Multivariate Normal (dim_states)
            # state: Tensor (dim_states x num_particles)
            state_d = MultivariateNormal(
                next_state.loc.mean(dim=1),
                covariance_matrix=torch.diag(
                    next_state.covariance_matrix.mean(dim=(1, 2)))
            )
            state = state_d.sample(sample_shape=torch.Size([num_particles]))
            assert state.shape == torch.Size([num_particles, self.dim_states])

        return MultivariateNormal(output_loc, covariance_matrix=output_covariance)

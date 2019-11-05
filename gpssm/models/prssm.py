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


class PRSSM(SSMSVI):
    """Implementation of PR-SSM algorithm."""

    def __init__(self,
                 gp_model: ModelList,
                 transitions: Transitions,
                 emissions: Emissions,
                 recognition_model: Recognition,
                 loss_factors: list = None,
                 num_particles: int = 32
                 ) -> None:
        super().__init__()
        self.dim_states = gp_model.num_outputs
        self.gp = gp_model
        self.transitions = transitions
        self.emissions = emissions

        self.prior_recognition = recognition_model.copy()
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
    def elbo(self, output_sequence: Tensor, input_sequence: Tensor = None,
             state_sequence: Tensor = None) -> Tensor:
        """Calculate the ELBO for the given output/input/state data.

        Parameters
        ----------
        output_sequence: Tensor.
            Tensor of output data [batch_size x sequence_length x dim_outputs].
        input_sequence: Tensor, optional.
            Tensor of input data, if any [batch_size x sequence_length x dim_inputs].
        state_sequence: Tensor, optional.
            Tensor of state data, if any [batch_size x sequence_length x dim_states].

        Returns
        -------
        elbo: Tensor.
            Differentiable tensor with ELBO of sequence.
        """
        num_particles = self.num_particles
        dim_inputs = input_sequence.shape[-1]
        batch_size, sequence_length, dim_outputs = output_sequence.shape

        # Calculate x1 from prior and posterior
        qx1 = self.posterior_recognition(output_sequence, input_sequence)
        px1 = self.prior_recognition(output_sequence, input_sequence)

        # Initial State: Tensor (batch_size x num_particles x dim_states)
        state = qx1.rsample(sample_shape=torch.Size([num_particles]))
        state = state.permute(1, 0, 2)
        assert state.shape == torch.Size([batch_size, num_particles, self.dim_states])

        log_lik = torch.tensor(0.)
        for t in range(sequence_length):
            ############################################################################
            # Generate the Samples #
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

            # Output: Torch (dim_outputs)
            y = output_sequence[:, t].expand(num_particles, batch_size, dim_outputs)
            y = y.permute(1, 0, 2)
            assert y.shape == torch.Size([batch_size, num_particles, dim_inputs])

            ############################################################################
            # Calculate the Log-likelihoods #
            ############################################################################

            # Log-likelihoods
            # Log-likelihoods
            y_pred = self.emissions(state)

            for iy in range(dim_outputs):
                log_lik += y_pred[iy].log_prob(y[..., iy]).mean() / dim_outputs

            ############################################################################
            # Next State #
            ############################################################################

            # next_state: Tensor (num_particles x state_dim)
            # state_d: Multivariate Normal (dim_states)
            # state: Tensor (dim_states x num_particles)
            state_d = next_state
            state = torch.zeros((batch_size, num_particles, self.dim_states))
            for ix in range(self.dim_states):
                state[:, :, ix] = state_d[ix].rsample()

        ################################################################################
        # Add KL Divergences #
        ################################################################################

        # There is 1 gp per dimension hence the sum.
        kl_u = torch.tensor(0.)
        for model in self.gp.models:
            kl_u += model.variational_strategy.kl_divergence().sum()
        kl_x1 = kl_divergence(qx1, px1)

        elbo = -(log_lik * self.loss_factors[0] / sequence_length
                 - kl_x1 * self.loss_factors[0] / sequence_length
                 - kl_u
                 )
        return elbo.mean()

    @torch.jit.export
    def forward(self, output_sequence: Tensor, input_sequence: Tensor = None
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
        num_particles = self.num_particles

        batch_size, recognition_length, dim_outputs = output_sequence.shape
        _, sequence_length, dim_inputs = input_sequence.shape

        # state_d: Multivariate Normal (batch_size x dim_states)
        px1 = self.prior_recognition(output_sequence, input_sequence)
        state_d = px1
        assert state_d.loc.shape == torch.Size([batch_size, self.dim_states])
        assert state_d.covariance_matrix.shape == torch.Size(
            [batch_size, self.dim_states, self.dim_states])

        # State: Tensor (batch_size, num_particles x dim_states)
        state = state_d.sample(sample_shape=torch.Size([num_particles]))
        state = state.permute(1, 0, 2)
        assert state.shape == torch.Size([batch_size, num_particles, self.dim_states])

        output_loc = torch.zeros((batch_size, sequence_length, dim_outputs))
        output_cov = torch.zeros((batch_size, sequence_length, dim_outputs, dim_outputs)
                                 )

        for t in range(sequence_length):
            ############################################################################
            # Generate the Samples #
            ############################################################################

            # Input: Tensor (batch_size x num_particles x dim_inputs)
            u = input_sequence[:, t].expand(num_particles, batch_size, dim_inputs)
            u = u.permute(1, 0, 2)
            assert u.shape == torch.Size([batch_size, num_particles, dim_inputs])

            # StateInput: Tensor (batch_size x num_particles x dim_states + dim_inputs)
            state_input = torch.cat((state, u), dim=-1)
            assert state_input.shape == torch.Size(
                [batch_size, num_particles, dim_inputs + self.dim_states])

            # next_f: [Multivariate Normal(batch_size x num_particles)] x dim_output
            next_f = self.gp(state_input)
            for ix in range(self.dim_states):
                assert next_f[ix].loc.shape == torch.Size(
                    [batch_size, num_particles])
                assert next_f[ix].covariance_matrix.shape == torch.Size(
                    [batch_size, num_particles, num_particles])

            # next_state: [Multivariate Normal(batch_size x num_particles)] x dim_output
            next_state = self.transitions(next_f)
            for ix in range(self.dim_states):
                assert next_state[ix].loc.shape == torch.Size(
                    [batch_size, num_particles])
                assert next_state[ix].covariance_matrix.shape == torch.Size(
                    [batch_size, num_particles, num_particles])

                assert (next_state[ix].loc == next_f[ix].loc).all()
                assert not (next_state[ix].covariance_matrix == next_f[
                    ix].covariance_matrix).all()

            # Output: [MultivariateNormal (batch_size x num_particles)] x dim_outputs
            output = self.emissions(state_d)

            # Collapse particles!
            for iy in range(dim_outputs):
                output_loc[:, t, iy] = output[iy].loc.squeeze(dim=-1)
                output_cov[:, t, iy, iy] = output[iy].covariance_matrix.squeeze()

            # next_state: Tensor(num_particles x state_dim)
            # state_d: Multivariate Normal (dim_states)
            # state: Tensor (dim_states x num_particles)
            ############################################################################
            # Next State #
            ############################################################################

            # next_state: Tensor (num_particles x state_dim)
            # state_d: Multivariate Normal (dim_states)
            # state: Tensor (dim_states x num_particles)

            loc = torch.zeros(batch_size, self.dim_states)
            cov = torch.zeros(batch_size, self.dim_states, self.dim_states)
            for ix in range(self.dim_states):
                loc[:, ix] = next_state[ix].loc.mean(dim=-1)
                cov[:, ix, ix] = torch.diag(next_state[ix].covariance_matrix)

            state_d = MultivariateNormal(loc, covariance_matrix=cov)
            state = state_d.sample(sample_shape=torch.Size([num_particles]))
            state = state.permute(1, 0, 2)
            assert state.shape == torch.Size(
                [batch_size, num_particles, self.dim_states])

        return MultivariateNormal(output_loc, covariance_matrix=output_cov)

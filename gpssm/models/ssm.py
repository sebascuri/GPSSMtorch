"""Base Class for System Id using Variational Inference with SSMs."""
from abc import ABC, abstractmethod
from torch import Tensor
import torch
import torch.jit
import torch.nn as nn
# from torch.distributions import kl_divergence
from typing import List, Tuple
from gpytorch.distributions import MultivariateNormal

from .components.dynamics import Dynamics, ZeroDynamics
from .components.emissions import Emissions
from .components.transitions import Transitions
from .components.recognition_model import Recognition
from .utilities import diagonal_covariance

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
        Factor for soft conditioning.
    loss_factors: dict.
        Factors to multiply each term of the ELBO with.
    independent_particles: bool.
        Flag to indicate if particles are independent.
    dataset_size: int.
        Size of dataset.
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
                 k_factor: float = 100.,
                 loss_factors: dict = None,
                 independent_particles: bool = True,
                 dataset_size: int = 1) -> None:
        super().__init__()
        self.dim_states = forward_model.num_outputs
        self.forward_model = forward_model
        if backward_model is None:
            backward_model = ZeroDynamics(self.dim_states - emissions.dim_outputs)
        self.backward_model = backward_model

        self.transitions = transitions
        self.emissions = emissions

        self.recognition = recognition_model
        self.num_particles = num_particles
        self.cubic_sampling = cubic_sampling
        self.loss_key = loss_key
        self.k_factor = k_factor
        if loss_factors is None:
            loss_factors = dict(kl_u=1., kl_conditioning=1.)
        self.loss_factors = loss_factors
        self.independent_particles = independent_particles
        self.dataset_size = dataset_size

    def __str__(self) -> str:
        """Return string of object with parameters."""
        string = "Model Parameters: \n\n"
        string += "Forward Model\n{}\n".format(self.forward_model)
        string += "Backward Model\n{}\n".format(self.backward_model)
        string += "Emission {}\n\n".format(self.emissions)
        string += "Transition {}\n\n".format(self.transitions)
        string += "Recognition x1 {}\n\n".format(self.recognition)

        return string

    def properties(self) -> list:
        """Return list of learnable parameters."""
        return [
            {'params': self.forward_model.parameters()},
            {'params': self.backward_model.parameters()},
            {'params': self.emissions.parameters()},
            {'params': self.transitions.parameters()},
            {'params': self.recognition.parameters()},
        ]

    @torch.jit.export
    def forward(self, *inputs: Tensor, **kwargs
                ) -> Tuple[List[MultivariateNormal], Tensor]:
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
        # SAMPLE GP #
        ################################################################################
        self.forward_model.resample()
        self.backward_model.resample()

        ################################################################################
        # PERFORM Backward Pass #
        ################################################################################
        # if self.training:
        #     output_distribution = self.backward(output_sequence, input_sequence)

        ################################################################################
        # Initial State #
        ################################################################################
        state = self.recognition(output_sequence[:, :self.recognition.length],
                                 input_sequence[:, :self.recognition.length],
                                 num_particles=num_particles)

        ################################################################################
        # PREDICT Outputs #
        ################################################################################
        outputs = []
        y_pred = self.emissions(state)
        outputs.append(MultivariateNormal(y_pred.loc.detach(),
                                          y_pred.covariance_matrix.detach()))

        ################################################################################
        # INITIALIZE losses #
        ################################################################################

        # entropy = torch.tensor(0.)
        # if self.training:
        #     output_distribution.pop(0)
        #     # entropy += y_tilde.entropy().mean() / sequence_length

        y = output_sequence[:, 0].expand(num_particles, batch_size, dim_outputs
                                         ).permute(1, 2, 0)
        log_lik = y_pred.log_prob(y).sum(dim=1).mean()  # type: torch.Tensor
        l2 = ((y_pred.loc - y) ** 2).sum(dim=1).mean()  # type: torch.Tensor
        kl_cond = torch.tensor(0.)

        for t in range(sequence_length - 1):
            ############################################################################
            # PREDICT Next State #
            ############################################################################
            u = input_sequence[:, t].expand(num_particles, batch_size, dim_inputs)
            u = u.permute(1, 2, 0)  # Move last component to end.
            state_samples = state.rsample()
            state_input = torch.cat((state_samples, u), dim=1)

            next_f = self.forward_model(state_input)
            next_state = self.transitions(next_f)
            next_state.loc += state_samples

            if self.independent_particles:
                next_state = diagonal_covariance(next_state)
            ############################################################################
            # CONDITION Next State #
            ############################################################################
            # if self.training:
            #     y_tilde = output_distribution.pop(0)
            #     p_next_state = next_state
            #     next_state = self._condition(next_state, y_tilde)
            #     kl_cond += kl_divergence(next_state, p_next_state).mean()
            ############################################################################
            # RESAMPLE State #
            ############################################################################
            state = next_state

            ############################################################################
            # PREDICT Outputs #
            ############################################################################
            y_pred = self.emissions(state)
            outputs.append(y_pred)

            ############################################################################
            # COMPUTE Losses #
            ############################################################################
            y = output_sequence[:, t + 1].expand(
                num_particles, batch_size, dim_outputs).permute(1, 2, 0)
            log_lik += y_pred.log_prob(y).sum(dim=1).mean()
            l2 += ((y_pred.loc - y) ** 2).sum(dim=1).mean()
            # entropy += y_tilde.entropy().mean() / sequence_length

        assert len(outputs) == sequence_length

        # if self.training:
        #     del output_distribution
        ################################################################################
        # Compute model KL divergences Divergences #
        ################################################################################
        factor = 1  # batch_size / self.dataset_size
        kl_uf = self.forward_model.kl_divergence()
        kl_ub = self.backward_model.kl_divergence()

        if self.forward_model.independent:
            kl_uf *= sequence_length
        if self.backward_model.independent:
            kl_ub *= sequence_length

        kl_cond = kl_cond * self.loss_factors['kl_conditioning'] * factor
        kl_ub = kl_ub * self.loss_factors['kl_u'] * factor
        kl_uf = kl_uf * self.loss_factors['kl_u'] * factor

        if self.loss_key.lower() == 'loglik':
            loss = -log_lik
        elif self.loss_key.lower() == 'elbo':
            loss = -(log_lik - kl_uf - kl_ub - kl_cond)
            if kwargs.get('print', False):
                str_ = 'elbo: {}, log_lik: {}, kluf: {}, klub: {}, klcond: {}'
                print(str_.format(loss.item(), log_lik.item(), kl_uf.item(),
                                  kl_ub.item(), kl_cond.item()))
        elif self.loss_key.lower() == 'l2':
            loss = l2
        elif self.loss_key.lower() == 'rmse':
            loss = torch.sqrt(l2)
        else:
            raise NotImplementedError("Key {} not implemented".format(self.loss_key))

        return outputs, loss

    @torch.jit.export
    def backward(self, *inputs: Tensor) -> List[MultivariateNormal]:
        """Implement backwards pass."""
        output_sequence, input_sequence = inputs
        _, _, dim_inputs = input_sequence.shape
        batch_size, sequence_length, dim_outputs = output_sequence.shape
        dim_states = self.dim_states
        num_particles = self.num_particles
        dim_delta = dim_states - dim_outputs
        shape = (batch_size, dim_delta, num_particles)

        ################################################################################
        # Final Pseudo Measurement #
        ################################################################################
        y = output_sequence[:, -1].expand(num_particles, -1, -1).permute(1, 2, 0)
        x_tilde_obs = self.emissions(y)

        loc = torch.cat((x_tilde_obs.loc, torch.zeros(*shape)), dim=1)

        cov = torch.cat((x_tilde_obs.covariance_matrix,
                         torch.diag_embed(torch.ones(*shape))), dim=1)

        x_tilde = MultivariateNormal(loc, cov)
        outputs = [x_tilde]

        for t in reversed(range(sequence_length - 1)):
            ############################################################################
            # PREDICT Previous pseudo-measurement #
            ############################################################################
            y = output_sequence[:, t].expand(num_particles, -1, -1).permute(1, 2, 0)
            x_tilde_obs = self.emissions(y)
            u = input_sequence[:, t].expand(num_particles, batch_size, dim_inputs)
            u = u.permute(1, 2, 0)

            # Here change the order of y_tilde for identity dynamics (those that
            # return the first dim_output states). The reason for this is that we
            # already append the y_ from emissions in the first components.
            # We can check this by comparing before computing next_x_tilde
            # loc[0, :, 0], x.loc[0, :, 0], x_tilde[0, :, 0]

            delta_idx = torch.arange(dim_outputs, dim_states)
            idx = torch.cat((delta_idx, torch.arange(dim_outputs)))
            x_tilde_samples = x_tilde.rsample()[:, idx]  # exchange indexes
            x_tilde_u = torch.cat((x_tilde_samples, u), dim=1)
            next_x_tilde = self.backward_model(x_tilde_u)
            next_x_tilde.loc += x_tilde_samples[:, :dim_delta]

            loc = torch.cat((x_tilde_obs.loc, next_x_tilde.loc), dim=1)
            cov = torch.cat((x_tilde_obs.covariance_matrix,
                             next_x_tilde.covariance_matrix), dim=1)

            ############################################################################
            # PREDICT Outputs #
            ############################################################################
            x_tilde = MultivariateNormal(loc, cov)
            outputs.append(x_tilde)

        assert len(outputs) == sequence_length
        return outputs[::-1]

    @abstractmethod
    def _condition(self, next_x: MultivariateNormal, next_y: MultivariateNormal
                   ) -> MultivariateNormal:
        """Implement conditioning."""
        raise NotImplementedError


class PRSSM(SSM):
    """Implementation of PR-SSM Algorithm."""

    @torch.jit.export
    def _condition(self, next_x: MultivariateNormal, next_y: MultivariateNormal
                   ) -> MultivariateNormal:
        """Implement conditioning."""
        return next_x


class CBFSSM(SSM):
    """Conditional Backwards Forwards Algorithm."""

    @torch.jit.export
    def _condition(self, next_x: MultivariateNormal, next_y: MultivariateNormal
                   ) -> MultivariateNormal:
        """Condition the next_x distribution with the measurements of next_y.

        Next_x is a Multivariate Normal and the covariance matrix is between particles.
        However, between the x-coordinates are independent.

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
        error = next_y.loc - next_x.loc
        sigma_f = next_x.lazy_covariance_matrix.diag()
        sigma_y = next_y.lazy_covariance_matrix.diag()
        sigma_y += (self.k_factor - 1) * sigma_f

        gain = sigma_f / (sigma_f + sigma_y)
        loc = next_x.loc + gain * error
        cov = torch.diag_embed(sigma_y * gain.pow(2))

        neg_gain = torch.diag_embed(1 - gain)
        cov += neg_gain @ next_x.covariance_matrix @ neg_gain
        return MultivariateNormal(loc, cov)

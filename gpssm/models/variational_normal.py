"""Variational Gaussian Implementation."""
from gpytorch import Module, settings
from gpytorch.lazy import CholLazyTensor
from gpytorch.distributions import MultivariateNormal
import torch
from torch.distributions.kl import kl_divergence
from torch import Tensor


class VariationalNormal(Module):
    """Variational distribution."""

    def __init__(self, dim_states: int, trainable: bool = True):
        super().__init__()

        mean_init = torch.zeros(dim_states)
        covar_init = torch.eye(dim_states, dim_states)

        self.prior_distribution = MultivariateNormal(
            mean_init.detach(), covar_init.detach()
        )

        self.register_parameter(name="variational_mean",
                                parameter=torch.nn.Parameter(mean_init,
                                                             requires_grad=trainable))
        self.register_parameter(name="chol_variational_covar",
                                parameter=torch.nn.Parameter(covar_init,
                                                             requires_grad=trainable))

    @property
    def variational_distribution(self) -> MultivariateNormal:
        """
        Return the variational distribution q(u) that this module represents.

        In this simplest case, this involves directly returning the variational mean.
        For the variational covariance matrix, we consider the lower triangle of the
        registered variational covariance parameter, while also ensuring that the
        diagonal remains positive.
        """
        chol_variational_covar = self.chol_variational_covar
        dtype = chol_variational_covar.dtype
        device = chol_variational_covar.device

        # First make the cholesky factor is upper triangular
        lower_mask = torch.ones(self.chol_variational_covar.shape[-2:], dtype=dtype,
                                device=device).tril(0)
        chol_variational_covar = chol_variational_covar.mul(lower_mask)

        # Now construct the actual matrix
        variational_covar = CholLazyTensor(chol_variational_covar)
        return MultivariateNormal(self.variational_mean, variational_covar)

    def __call__(self) -> MultivariateNormal:
        """Call the variational distribution."""
        return self.forward()

    def rsample(self, sample_shape: torch.Size = torch.Size([])) -> Tensor:
        """Get a differentiable sample from the variational distribution."""
        return self.variational_distribution.rsample(sample_shape=sample_shape)

    def forward(self) -> MultivariateNormal:
        """Call the variational distribution."""
        return self.variational_distribution

    def kl_divergence(self):
        """Get KL divergence between prior and current distribution."""
        with settings.max_preconditioner_size(0):
            return kl_divergence(self.variational_distribution, self.prior_distribution)

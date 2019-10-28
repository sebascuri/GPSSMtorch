"""Variational Gaussian Implementation."""
from gpytorch import Module
from gpytorch.distributions import MultivariateNormal
import torch


class VariationalGaussian(Module):
    def __init__(self, state_dim, batch_shape):
        super().__init__()
        mean_init = torch.zeros(state_dim)
        covar_init = torch.eye(state_dim, state_dim)
        mean_init = mean_init.repeat(*batch_shape, 1)
        covar_init = covar_init.repeat(*batch_shape, 1, 1)

        self.register_parameter(name="variational_mean",
                                parameter=torch.nn.Parameter(mean_init,
                                                             requires_grad=True))
        self.register_parameter(name="chol_variational_covar",
                                parameter=torch.nn.Parameter(covar_init,
                                                             requires_grad=True))

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        """
        The :func:`~gpytorch.variational.VariationalStrategy.prior_distribution` method determines how to compute the
        GP prior distribution of the inducing points, e.g. :math:`p(u) \sim N(\mu(X_u), K(X_u, X_u))`. Most commonly,
        this is done simply by calling the user defined GP prior on the inducing point data directly.
        """
        out = self.model.forward(self.inducing_points)
        res = MultivariateNormal(
            out.mean, out.lazy_covariance_matrix.add_jitter()
        )
        return res


if __name__ == """__main__""":
    pass
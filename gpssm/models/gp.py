"""Implementation of GP Models."""

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from gpytorch.kernels import Kernel
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

__author__ = 'Sebastian Curi'
__all__ = ['ExactGPModel', 'VariationalGP']


class ExactGPModel(gpytorch.models.ExactGP):
    """An Exact GP Model implementation.

    Exact GP Models require that all states are measured and the dimension of y and x
    is the same.

    This model is only valid for one sequence of data.

    Parameters
    ----------
    train_inputs: torch.tensor.
        Tensor of states with shape [sequence_length x dimension].

    train_outputs: torch.tensor.
        Tensor of outputs with shape [dimension x sequence_length].

    likelihood: Likelihood.
        Only Gaussian Likelihoods are allowed in Exact GPs.

    mean: Mean
        Prior mean function of GP.

    kernel: Kernel
        Prior kernel function of GP.

    Examples
    --------
    >>> from gpytorch.means import ConstantMean
    >>> from gpytorch.kernels import ScaleKernel, RBFKernel
    >>> from gpytorch.likelihoods import GaussianLikelihood
    >>> from gpytorch.mlls import ExactMarginalLogLikelihood
    >>> import torch.testing
    >>> xu = torch.randn((64, 3))
    >>> dim_x, dim_u = 2, 1
    >>> dim_y = dim_x
    >>> y1 = (torch.sin(xu[:, 0]) + 2 * xu[:, 2] - xu[:, 1] ** 2).unsqueeze(dim=1)
    >>> y2 = (torch.cos(xu[:, 0]) - 2 * xu[:, 2] ** 2 + xu[:, 1]).unsqueeze(dim=1)
    >>> y = torch.cat((y1, y2), dim=-1) + 0.1 * torch.randn((64, 2))
    >>> y = y.t()
    >>> likelihood = GaussianLikelihood(batch_size=dim_x)
    >>> mean = ConstantMean(batch_size=dim_x)
    >>> kernel = ScaleKernel(RBFKernel(batch_size=dim_x), batch_size=dim_x)
    >>> train_xu, test_xu = xu[:32], xu[32:]
    >>> train_y, test_y = y[:, :32], y[:, 32:]
    >>> model = ExactGPModel(train_xu, train_y, likelihood, mean, kernel)
    >>> mll = ExactMarginalLogLikelihood(likelihood, model)
    >>> pred_f = model(train_xu)
    >>> loss = -mll(pred_f, train_y)
    >>> torch.testing.assert_allclose(loss, -likelihood(pred_f).log_prob(train_y) / 32)
    >>> m,l = model.eval(), likelihood.eval()
    >>> pred_y = likelihood(model(test_xu))
    >>> mll = -pred_y.log_prob(test_y).sum()
    """

    def __init__(self,
                 train_inputs: torch.tensor,
                 train_outputs: torch.tensor,
                 likelihood: Likelihood,
                 mean: Mean,
                 kernel: Kernel) -> None:
        assert train_inputs.shape[-2] == train_outputs.shape[-1], """
            Train inputs have to have shape [num_points x in_dim] or
            [out_dim x num_points x in_dim].
            Train outputs have to have shape [out_dim x num_points].
        """
        super().__init__(train_inputs, train_outputs, likelihood)
        self.num_outputs = train_outputs.shape[0]
        self.mean_module = mean
        self.covar_module = kernel

    def __call__(self, state_input: torch.tensor, **kwargs) -> MultivariateNormal:
        """Override call method to expand test inputs and not train inputs."""
        if torch.equal(self.train_inputs[0], state_input):
            return super().__call__(state_input, **kwargs)
        else:
            state_input = state_input.expand(self.num_outputs, *state_input.shape)
            return super().__call__(state_input, **kwargs)

    def forward(self, state_input: torch.tensor) -> MultivariateNormal:
        """Forward call of GP class."""
        mean_x = self.mean_module(state_input)
        covar_x = self.covar_module(state_input)
        return MultivariateNormal(mean_x, covar_x)


class VariationalGP(AbstractVariationalGP):
    """Sparse Variational GP Class.

    Parameters
    ----------
    inducing_points: torch.tensor.
        Tensor with size [output_dims x num_inducing x input_dims] with location of
        inducing points.
        Note that it is critical that the first dimension is output dim as it needs this
        to build the number of output distribution of the inducing points.

    mean: Mean
        Prior mean function of GP.

    kernel: Kernel
        Prior kernel function of GP.

    Methods
    -------
    sample_gp: None -> ExactGP.
        Get an exact GP by sampling a training set from the inducing points.
        This is useful when we do not want to marginalize the GP points.

    References
    ----------
    Titsias, M. (2009, April). Variational learning of inducing variables in sparse
    Gaussian processes. In Artificial Intelligence and Statistics (pp. 567-574).

    Examples
    --------
    >>> from gpytorch.means import ConstantMean
    >>> from gpytorch.kernels import ScaleKernel, RBFKernel
    >>> from gpytorch.likelihoods import GaussianLikelihood
    >>> from gpytorch.mlls import VariationalELBO, ExactMarginalLogLikelihood
    >>> import torch.testing
    >>> num_points = 64
    >>> xu = torch.randn((num_points, 3))
    >>> num_inducing_points = 25
    >>> learn_inducing_loc = True
    >>> dim_x, dim_u = 2, 1
    >>> dim_y = dim_x
    >>> y1 = (torch.sin(xu[:, 0]) + 2 * xu[:, 2] - xu[:, 1] ** 2).unsqueeze(dim=1)
    >>> y2 = (torch.cos(xu[:, 0]) - 2 * xu[:, 2] ** 2 + xu[:, 1]).unsqueeze(dim=1)
    >>> y = torch.cat((y1, y2), dim=-1) + 0.1 * torch.randn((64, 2))
    >>> y = y.t()
    >>> inducing_points = torch.randn((dim_y, num_inducing_points, dim_x + dim_u))
    >>> likelihood = GaussianLikelihood(batch_size=dim_x)
    >>> mean = ConstantMean(batch_size=dim_x)
    >>> kernel = ScaleKernel(RBFKernel(batch_size=dim_x), batch_size=dim_x)
    >>> model = VariationalGP(inducing_points, mean, kernel, learn_inducing_loc)
    >>> mll = VariationalELBO(likelihood, model, num_points, combine_terms=False)
    >>> pred_f = model(xu)
    >>> log_lik, kl_div, log_prior = mll(pred_f, y)
    >>> loss = -(log_lik - kl_div + log_prior).sum()
    >>> pred_y = likelihood(pred_f)
    >>> ell = likelihood.expected_log_prob(y, pred_f) / num_points
    >>> torch.testing.assert_allclose(ell, log_lik)
    >>> model_i = model.sample_gp(likelihood)
    >>> m = model_i.eval()
    >>> mll = ExactMarginalLogLikelihood(likelihood, model)
    >>> pred_f = model(xu)
    >>> loss = -mll(pred_f, y)
    >>> torch.testing.assert_allclose(loss, -likelihood(pred_f).log_prob(y) / 64)
    """

    def __init__(self, inducing_points: torch.tensor,
                 mean: Mean,
                 kernel: Kernel,
                 learn_inducing_loc: bool = True):
        num_outputs, num_inducing, input_dims = inducing_points.shape
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_size=num_outputs
        )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=learn_inducing_loc
        )
        super().__init__(variational_strategy)
        self.mean_module = mean
        self.covar_module = kernel
        self.num_outputs = num_outputs

    def __call__(self, state_input: torch.tensor, **kwargs) -> MultivariateNormal:
        """Override call method to expand test inputs and not train inputs."""
        if not torch.equal(state_input, self.variational_strategy.inducing_points):
            state_input = state_input.expand(self.num_outputs, *state_input.shape)
        return super().__call__(state_input, **kwargs)

    def forward(self, state_input: torch.tensor) -> MultivariateNormal:
        """Forward call of GP class."""
        mean_x = self.mean_module(state_input)
        covar_x = self.covar_module(state_input)
        return MultivariateNormal(mean_x, covar_x)

    def sample_gp(self, likelihood: Likelihood) -> ExactGPModel:
        """Sample an Exact GP from the variational distribution."""
        train_xu = self.variational_strategy.inducing_points
        d = self.variational_strategy.variational_distribution.variational_distribution
        train_y = d.rsample()  # Do not propagate the gradients?

        return ExactGPModel(train_xu, train_y, likelihood,
                            self.mean_module, self.covar_module)
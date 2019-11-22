"""Implementation of GP Models."""

# import torch
from torch import Tensor
from abc import ABC, abstractmethod
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from gpytorch.kernels import Kernel
from gpytorch.models import AbstractVariationalGP, ExactGP
from gpytorch.variational import CholeskyVariationalDistribution, \
    VariationalDistribution
from gpytorch.variational import VariationalStrategy

__author__ = 'Sebastian Curi'
__all__ = ['GPSSM', 'ExactGPModel', 'VariationalGP']


class GPSSM(ABC):
    """GPSSM's are GPs defined in different outputs.

    Parameters
    ----------
    mean: Mean
        Prior mean function of GP.

    kernel: Kernel
        Prior kernel function of GP.
    """

    def __init__(self, mean: Mean, kernel: Kernel) -> None:
        self.mean_module = mean
        self.covar_module = kernel

    @abstractmethod
    def __call__(self, state_input: Tensor, **kwargs) -> MultivariateNormal:
        """Call a GP-SSM at a given state-input pair."""
        raise NotImplementedError

    def __str__(self) -> str:
        """Return GP parameters as a string."""
        lengthscale = self.covar_module.base_kernel.lengthscale.detach()
        outputscale = self.covar_module.outputscale.detach()
        return "\toutputscale: {}\n \tlengthscale: {}".format(outputscale, lengthscale)


class ExactGPModel(GPSSM, ExactGP):
    """An Exact GP Model implementation.

    Exact GP Models require that all states are measured and the dimension of y and x
    is the same.

    This model is only valid for one sequence of data.

    Parameters
    ----------
    train_inputs: Tensor
        Tensor of states with shape [sequence_length x dimension].

    train_outputs: Tensor.
        Tensor of outputs with shape [dimension x sequence_length].

    likelihood: Likelihood.
        Only Gaussian Likelihoods are allowed in Exact GPs.

    mean: Mean
        Prior mean function of GP.

    kernel: Kernel
        Prior kernel function of GP.

    Examples
    --------
    >>> import torch
    >>> from gpssm.models.components.gp import ExactGPModel
    >>> from gpytorch.means import ConstantMean
    >>> from gpytorch.kernels import ScaleKernel, RBFKernel
    >>> from gpytorch.likelihoods import GaussianLikelihood
    >>> from gpytorch.mlls import ExactMarginalLogLikelihood
    >>> from torch import Size
    >>> from torch.testing import assert_allclose
    >>> data_size = 32
    >>> dim_x = 2
    >>> x = torch.randn((data_size, dim_x))
    >>> assert_allclose(x.shape, Size([data_size, dim_x]))
    >>> y = torch.sin(x[:, 0]) + 2 * x[:, 1] - x[:, 1] ** 2
    >>> y += 0.1 * torch.randn(data_size)
    >>> likelihoods = GaussianLikelihood()
    >>> mean = ConstantMean()
    >>> kernel = ScaleKernel(RBFKernel())
    >>> train_size = data_size // 2
    >>> train_x, test_x = x[:train_size], x[train_size:]
    >>> train_y, test_y = y[:train_size], y[train_size:]
    >>> model = ExactGPModel(train_x, train_y, likelihoods, mean, kernel)
    >>> mll = ExactMarginalLogLikelihood(likelihoods, model)
    >>> pred_f = model(train_x)
    >>> loss = -mll(pred_f, train_y)
    >>> assert_allclose(loss, -likelihoods(pred_f).log_prob(train_y) / train_size)
    >>> m,l = model.eval(), likelihoods.eval()
    >>> pred_y = likelihoods(model(test_x))
    >>> loss = -pred_y.log_prob(test_y) / train_size
    >>> batch = model(torch.randn(8, 4, dim_x))
    >>> assert_allclose(batch.loc.shape, torch.Size([8, 4]))
    >>> assert_allclose(batch.covariance_matrix.shape, torch.Size([8, 4, 4]))
    """

    def __init__(self,
                 train_inputs: Tensor,
                 train_outputs: Tensor,
                 likelihood: Likelihood,
                 mean: Mean,
                 kernel: Kernel) -> None:
        assert train_inputs.shape[-2] == train_outputs.shape[-1], """
            Train inputs have to have shape [num_points x in_dim] or
            [out_dim x num_points x in_dim].
            Train outputs have to have shape [out_dim x num_points].
        """
        ExactGP.__init__(self, train_inputs, train_outputs, likelihood)
        GPSSM.__init__(self, mean, kernel)

    def __call__(self, state_input: Tensor, **kwargs) -> MultivariateNormal:
        """Override call method to expand test inputs and not train inputs."""
        return ExactGP.__call__(self, state_input, **kwargs)

    def forward(self, state_input: Tensor) -> MultivariateNormal:
        """Forward call of GP class."""
        mean_x = self.mean_module(state_input)
        covar_x = self.covar_module(state_input)
        return MultivariateNormal(mean_x, covar_x)


class VariationalGP(GPSSM, AbstractVariationalGP):
    """Sparse Variational GP Class.

    Parameters
    ----------
    inducing_points: Tensor.
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
    >>> import torch
    >>> from gpytorch.means import ConstantMean
    >>> from gpytorch.kernels import ScaleKernel, RBFKernel
    >>> from gpytorch.likelihoods import GaussianLikelihood
    >>> from gpytorch.mlls import VariationalELBO, ExactMarginalLogLikelihood
    >>> from torch.testing import assert_allclose
    >>> data_size = 64
    >>> dim_x = 2
    >>> x = torch.randn((data_size, dim_x))
    >>> num_inducing_points = 25
    >>> learn_inducing_loc = True
    >>> y = torch.sin(x[:, 0]) + 2 * x[:, 1] - x[:, 1] ** 2 + torch.randn(data_size)
    >>> inducing_points = torch.randn((num_inducing_points, dim_x))
    >>> likelihoods = GaussianLikelihood()
    >>> mean = ConstantMean()
    >>> kernel = ScaleKernel(RBFKernel())
    >>> model = VariationalGP(inducing_points, mean, kernel, learn_inducing_loc)
    >>> mll = VariationalELBO(likelihoods, model, data_size, combine_terms=False)
    >>> pred_f = model(x)
    >>> log_lik, kl_div, log_prior = mll(pred_f, y)
    >>> loss = -(log_lik - kl_div + log_prior).sum()
    >>> pred_y = likelihoods(pred_f)
    >>> ell = likelihoods.expected_log_prob(y, pred_f) / data_size
    >>> assert_allclose(ell, log_lik)
    >>> model_i = model.sample_gp(likelihoods)
    >>> m = model_i.eval()
    >>> mll = ExactMarginalLogLikelihood(likelihoods, model)
    >>> pred_f = model(x)
    >>> loss = -mll(pred_f, y)
    >>> torch.testing.assert_allclose(loss, -likelihoods(pred_f).log_prob(y)/data_size)
    >>> batch = model(torch.randn(8, 4, dim_x))
    >>> assert_allclose(batch.loc.shape, torch.Size([8, 4]))
    >>> assert_allclose(batch.covariance_matrix.shape, torch.Size([8, 4, 4]))
    """

    def __init__(self, inducing_points: Tensor,
                 mean: Mean,
                 kernel: Kernel,
                 learn_inducing_loc: bool = True,
                 variational_distribution: VariationalDistribution = None) -> None:

        if variational_distribution is None:
            batch_k, num_inducing, input_dims = inducing_points.shape
            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=num_inducing,
                batch_size=batch_k
            )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=learn_inducing_loc
        )
        self.num_outputs = inducing_points.shape[0]
        AbstractVariationalGP.__init__(self, variational_strategy)
        GPSSM.__init__(self, mean, kernel)

    def __call__(self, state_input: Tensor, **kwargs) -> MultivariateNormal:
        """Override call method to expand test inputs and not train inputs."""
        batch_size, dim_inputs, num_particles = state_input.shape
        if batch_size == 1:
            state_input = state_input[0].expand(
                self.num_outputs, dim_inputs, num_particles).permute(0, 2, 1)
        else:
            state_input = state_input.expand(
                self.num_outputs, batch_size, dim_inputs, num_particles
            ).permute(1, 0, 3, 2)

        f = AbstractVariationalGP.__call__(self, state_input, **kwargs)
        if batch_size == 1:
            f.loc = f.loc.unsqueeze(0)
            f.covariance_matrix = f.covariance_matrix.unsqueeze(0)
        return f

    def forward(self, state_input: Tensor) -> MultivariateNormal:
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

    def kl_divergence(self) -> Tensor:
        """Get the KL-Divergence of the Model."""
        return self.variational_strategy.kl_divergence().mean()

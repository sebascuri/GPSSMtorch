"""Implementation of GP Models."""

import torch
from abc import ABC, abstractmethod
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from gpytorch.kernels import Kernel
from gpytorch.models import AbstractVariationalGP, ExactGP
from gpytorch.variational import CholeskyVariationalDistribution, \
    VariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.models.model_list import AbstractModelList
from torch.nn import ModuleList
from typing import List

__author__ = 'Sebastian Curi'
__all__ = ['GPSSM', 'ExactGPModel', 'VariationalGP', 'ModelList']


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
    def __call__(self, state_input: torch.tensor, **kwargs) -> MultivariateNormal:
        """Call a GP-SSM at a given state-input pair."""
        raise NotImplementedError

    def __str__(self) -> str:
        """Return GP parameters as a string."""
        # TODO: also think about other kernels + Mean functions.
        lengthscale = self.covar_module.base_kernel.lengthscale.detach()
        outputscale = self.covar_module.outputscale.detach()
        return "outputscale: {}, lengthscale: {}".format(outputscale, lengthscale)


class ExactGPModel(GPSSM, ExactGP):
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
        ExactGP.__init__(self, train_inputs, train_outputs, likelihood)
        GPSSM.__init__(self, mean, kernel)

    def __call__(self, state_input: torch.tensor, **kwargs) -> MultivariateNormal:
        """Override call method to expand test inputs and not train inputs."""
        return ExactGP.__call__(self, state_input, **kwargs)

    def forward(self, state_input: torch.tensor) -> MultivariateNormal:
        """Forward call of GP class."""
        mean_x = self.mean_module(state_input)
        covar_x = self.covar_module(state_input)
        return MultivariateNormal(mean_x, covar_x)


class VariationalGP(GPSSM, AbstractVariationalGP):
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

    def __init__(self, inducing_points: torch.tensor,
                 mean: Mean,
                 kernel: Kernel,
                 learn_inducing_loc: bool = True,
                 variational_distribution: VariationalDistribution = None) -> None:

        if variational_distribution is None:
            num_inducing, input_dims = inducing_points.shape
            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=num_inducing,
            )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=learn_inducing_loc
        )
        AbstractVariationalGP.__init__(self, variational_strategy)
        GPSSM.__init__(self, mean, kernel)

    def __call__(self, state_input: torch.tensor, **kwargs) -> MultivariateNormal:
        """Override call method to expand test inputs and not train inputs."""
        return AbstractVariationalGP.__call__(self, state_input, **kwargs)

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

    def kl_divergence(self) -> torch.Tensor:
        """Get the KL-Divergence of the Model."""
        return self.variational_strategy.kl_divergence().mean()


class ModelList(AbstractModelList):
    """List of variational models.

    Properties
    ----------
    models: list of GPSSMs.

    Examples
    --------
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
    >>> y1 = torch.sin(x[:, 0]) + 2 * x[:, 1] - x[:, 1] ** 2 + torch.randn(data_size)
    >>> y2 = torch.cos(x[:, 0]) - 2 * x[:, 1] + x[:, 1] ** 2 + torch.randn(data_size)
    >>> y = [y1, y2]
    >>> dim_y = len(y)
    >>> ip = torch.randn((num_inducing_points, dim_x))
    >>> mean = ConstantMean()
    >>> kernel = ScaleKernel(RBFKernel())
    >>> model = ModelList([VariationalGP(ip, mean, kernel, True) for _ in range(dim_y)])
    >>> likelihoods = [GaussianLikelihood() for _ in range(dim_y)]
    >>> debug_str = str(model)
    >>> assert model.num_outputs == dim_y
    >>> assert type(model(x)) is list
    >>> assert type(model(x)[0]) is MultivariateNormal
    >>> mean_shape = torch.Size([data_size])
    >>> cov_shape = torch.Size([data_size, data_size])
    >>> for i in range(dim_y):
    ...     assert_allclose(model(x)[i].loc.shape, mean_shape)
    ...     assert_allclose(model(x)[i].covariance_matrix.shape, cov_shape)
    >>> x = torch.randn((8, 4, dim_x))
    >>> mean_shape = torch.Size([8, 4])
    >>> cov_shape = torch.Size([8, 4, 4])
    >>> for i in range(dim_y):
    ...     assert_allclose(model(x)[i].loc.shape, mean_shape)
    ...     assert_allclose(model(x)[i].covariance_matrix.shape, cov_shape)
    >>> for i in range(dim_y):
    ...     assert_allclose(model.forward_i(i, x).loc, model(x)[i].loc)
    >>> sampled_model = model.sample_gp(likelihoods)
    >>> assert sampled_model.num_outputs == dim_y
    >>> assert type(sampled_model.models[0]) == ExactGPModel
    >>> for i in range(dim_y):
    ...     model_lik = sampled_model.likelihood_i(i, x)
    ...     lik = likelihoods[i](x)
    ...     assert_allclose(model_lik.loc, lik.loc)
    ...     assert_allclose(model_lik.scale, lik.scale)
    """

    def __init__(self, models: List[GPSSM]) -> None:
        super().__init__()
        self.models = ModuleList(models)

    def __str__(self) -> str:
        """Return GP parameters as a string."""
        string = ""
        for i in range(self.num_outputs):
            string += "component {} {}".format(i, str(self.models[i]))
        return string

    @property
    def num_outputs(self) -> int:
        """Get the number of outputs."""
        return len(self.models)

    def forward_i(self, i, *args, **kwargs):
        """Forward propagate model i."""
        return self.models[i].forward(*args, **kwargs)

    def likelihood_i(self, i, *args, **kwargs):
        """Likelihood of model i."""
        return self.models[i].likelihood(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward propagate all models."""
        return [model(*args, **kwargs) for model in self.models]

    def __call__(self, *args, **kwargs):
        """Forward propagate all models."""
        return self.forward(*args, **kwargs)

    def sample_gp(self, likelihood: List[Likelihood]) -> 'ModelList':
        """Sample an Exact GP from the variational distribution."""
        m = []
        for iy in range(self.num_outputs):
            m.append(self.models[iy].sample_gp(likelihood[iy]))
        return ModelList(m)

    def kl_divergence(self) -> torch.Tensor:
        """Get the KL-Divergence of the Model List."""
        kl_u = torch.tensor(0.)
        for model in self.models:
            kl_u += model.kl_divergence()
        return kl_u

    def get_fantasy_model(self, inputs, targets, **kwargs) -> 'ModelList':
        """Get a New GP with the inputs/targets."""
        models = [model.get_fantasy_model(inputs, target_.rsample(), **kwargs)
                  for (model, target_) in zip(self.models, targets)]
        return ModelList(models)

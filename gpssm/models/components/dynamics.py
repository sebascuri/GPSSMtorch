"""Implementation of GP Models."""

import torch
from torch import nn as nn
from torch import Tensor
from abc import ABC
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from gpytorch.kernels import Kernel
from gpytorch.models import AbstractVariationalGP, ExactGP
from .variational import VariationalStrategy
from .variational import ApproxCholeskyVariationalDistribution as AppCholVaDi
from .variational import CholeskyMeanVariationalDistribution as CholMeanVaDi
from .variational import DeltaVariationalDistribution as DeltaVaDi
from .variational import CholeskySampleVariationalDistribution as CholSamVaDi

__author__ = 'Sebastian Curi'
__all__ = ['Dynamics', 'ZeroDynamics',
           'GPDynamics', 'ExactGPModel', 'VariationalGP']


class Dynamics(ABC):
    """Dynamics Model.

    A dynamical model is a function that receives a tensor (x, u) and returns a
    distribution over the next states.

    """

    def __init__(self, num_outputs: int):
        self.num_outputs = num_outputs

    def parameters(self):
        """Get learnable parameters."""
        raise NotImplementedError

    def __call__(self, *args: Tensor, **kwargs) -> MultivariateNormal:
        """Call a Dynamical System at a given state-input pair."""
        raise NotImplementedError

    def kl_divergence(self) -> Tensor:
        """Get the KL-Divergence of the Model with the Prior."""
        return torch.tensor(0.0)

    @property
    def independent(self):
        """Return true if the function calls are independent of each other."""
        return True

    def resample(self):
        """Resample the variational distribution approximation points."""
        pass


class NNDynamics(nn.Module, Dynamics):
    """GPDynamics is a Dynamical model defined over GPs.

    Parameters
    ----------
    num_outputs: int.
        Number of outputs in model.
    """

    def __init__(self, num_outputs: int) -> None:
        Dynamics.__init__(self, num_outputs)
        nn.Module.__init__(self)

    def __str__(self) -> str:
        """Return GP parameters as a string."""
        return ""


class ZeroDynamics(NNDynamics):
    """Dynamics that returns a zero next state."""

    def forward(self, *args: Tensor, **kwargs) -> MultivariateNormal:
        """Call a Dynamical System at a given state-input pair."""
        state_input = args[0]
        batch_size, _, num_particles = state_input.shape
        loc = torch.zeros(batch_size, self.num_outputs, num_particles)
        scale = torch.ones(batch_size, self.num_outputs, num_particles)
        cov = torch.diag_embed(scale)

        return MultivariateNormal(loc, cov)


class GPDynamics(Dynamics):
    """GPDynamics is a Dynamical model defined over GPs.

    Parameters
    ----------
    mean: Mean
        Prior mean function of GP.

    kernel: Kernel
        Prior kernel function of GP.
    """

    def __init__(self, num_outputs: int, mean: Mean, kernel: Kernel) -> None:
        Dynamics.__init__(self, num_outputs)
        self.mean_module = mean
        self.covar_module = kernel

    def __str__(self) -> str:
        """Return GP parameters as a string."""
        lengthscale = self.covar_module.base_kernel.lengthscale.detach()
        outputscale = self.covar_module.outputscale.detach()
        return "\toutputscale: {}\n \tlengthscale: {}".format(outputscale, lengthscale)


class ExactGPModel(ExactGP, GPDynamics):
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
    >>> from gpssm.models.components.dynamics import ExactGPModel
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
        GPDynamics.__init__(self, train_outputs.shape[0], mean, kernel)

    def __call__(self, *args: Tensor, **kwargs) -> MultivariateNormal:
        """Override call method to expand test inputs and not train inputs."""
        state_input = args[0]
        return ExactGP.__call__(self, state_input, **kwargs)

    def forward(self, state_input: Tensor) -> MultivariateNormal:
        """Forward call of GP class."""
        mean_x = self.mean_module(state_input)
        covar_x = self.covar_module(state_input)
        return MultivariateNormal(mean_x, covar_x)


class VariationalGP(AbstractVariationalGP, GPDynamics):
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
    >>> x = torch.randn((dim_x, data_size))
    >>> num_inducing_points = 25
    >>> learn_inducing_loc = True
    >>> y = torch.sin(x[0]) + 2 * x[1] - x[1] ** 2 + torch.randn(data_size)
    >>> inducing_points = torch.randn((1, num_inducing_points, dim_x))
    >>> likelihoods = GaussianLikelihood()
    >>> mean = ConstantMean()
    >>> kernel = ScaleKernel(RBFKernel())
    >>> model = VariationalGP(inducing_points, mean, kernel, learn_inducing_loc)
    >>> mll = VariationalELBO(likelihoods, model, data_size, combine_terms=False)
    >>> pred_f = model(x.unsqueeze(0))
    >>> log_lik, kl_div, log_prior = mll(pred_f, y)
    >>> loss = -(log_lik - kl_div + log_prior).sum()
    >>> pred_y = likelihoods(pred_f)
    >>> ell = likelihoods.expected_log_prob(y, pred_f) / data_size
    >>> assert_allclose(ell, log_lik)
    """

    def __init__(self, inducing_points: Tensor,
                 mean: Mean,
                 kernel: Kernel,
                 learn_inducing_loc: bool = True,
                 variational_distribution: AppCholVaDi = None
                 ) -> None:
        if variational_distribution is None:
            batch_k, num_inducing, input_dims = inducing_points.shape
            variational_distribution = AppCholVaDi(
                num_inducing_points=num_inducing,
                batch_size=batch_k
            )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=learn_inducing_loc
        )
        AbstractVariationalGP.__init__(self, variational_strategy)
        GPDynamics.__init__(self, inducing_points.shape[0], mean, kernel)

    def __call__(self, *args: Tensor, **kwargs) -> MultivariateNormal:
        """Override call method to expand test inputs and not train inputs."""
        state_input = args[0]
        batch_size, dim_inputs, num_particles = state_input.shape
        state_input = state_input.expand(self.num_outputs, batch_size, dim_inputs,
                                         num_particles).permute(1, 0, 3, 2)

        return AbstractVariationalGP.__call__(self, state_input, **kwargs)

    def forward(self, state_input: Tensor) -> MultivariateNormal:
        """Forward call of GP class."""
        mean_x = self.mean_module(state_input)
        covar_x = self.covar_module(state_input)
        return MultivariateNormal(mean_x, covar_x)

    def resample(self):
        """Resample the variational distribution approximation points."""
        self.variational_strategy.resample()

    def kl_divergence(self) -> Tensor:
        """Get the KL-Divergence of the Model."""
        return self.variational_strategy.kl_divergence().sum(dim=1).mean()

    @property
    def independent(self):
        type_var_dist = type(self.variational_strategy.variational_distribution)
        return not (type_var_dist is CholMeanVaDi
                    or type_var_dist is DeltaVaDi
                    or type_var_dist is CholSamVaDi)
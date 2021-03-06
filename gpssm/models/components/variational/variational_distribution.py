"""Delta Variational Distribution."""
import numbers
import torch
from torch.distributions.kl import register_kl
from gpytorch.distributions.distribution import Distribution
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.lazy import CholLazyTensor
from gpytorch.variational import CholeskyVariationalDistribution


class Delta(Distribution):
    """Degenerate discrete distribution (a single point).

    Discrete distribution that assigns probability one to the single element in
    its support. Delta distribution parameterized by a random choice should not
    be used with MCMC based inference, as doing so produces incorrect results.

    :param torch.Tensor v: The single support element.
    :param torch.Tensor log_density: An optional density for this Delta. This
        is useful to keep the class of :class:`Delta` distributions closed
        under differentiable transformation.
    :param int event_dim: Optional event dimension, defaults to zero.
    """

    has_rsample = True

    def __init__(self, v, log_density=0.0, event_dim=0, validate_args=None):
        if event_dim > v.dim():
            raise ValueError("Expected event_dim <= v.dim(), actual {} vs {}".format(
                event_dim, v.dim()))
        batch_dim = v.dim() - event_dim
        batch_shape = v.shape[:batch_dim]
        event_shape = v.shape[batch_dim:]
        if isinstance(log_density, numbers.Number):
            log_density = torch.full(batch_shape, log_density, dtype=v.dtype,
                                     device=v.device)
        elif validate_args and log_density.shape != batch_shape:
            raise ValueError("Expected log_density.shape = {}, actual {}".format(
                log_density.shape, batch_shape))
        self.v = v
        self.log_density = log_density
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape: torch.Size, _instance=None):
        """Expand distribution to a given batch size."""
        new = self._get_checked_instance(Delta, _instance)
        batch_shape = torch.Size(batch_shape)
        new.v = self.v.expand(*(batch_shape + self.event_shape), -1)
        new.log_density = self.log_density.expand(*batch_shape, -1)
        super().__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def lazy_covariance_matrix(self):
        """Get lazy covariance matrix."""
        return CholLazyTensor(torch.diag_embed(self.variance))

    def rsample(self, sample_shape=torch.Size()):
        """Sample with reparametrization trick."""
        shape = sample_shape + self.v.shape
        return self.v.expand(shape)

    def log_prob(self, x):
        """Get the log probability of a point x."""
        v = self.v.expand(self.batch_shape + self.event_shape)
        log_prob = (x == v).type(x.dtype).log()
        if len(self.event_shape):
            log_prob = log_prob.sum(list(range(-1, -len(self.event_shape) - 1, -1)))
        return log_prob + self.log_density

    @property
    def mean(self):
        """Get the mean of the distribution."""
        return self.v

    @property
    def variance(self):
        """Get the variance of the distribution."""
        return torch.zeros_like(self.v)


@register_kl(Delta, MultivariateNormal)
def kl_mvn_mvn(p_dist, q_dist):
    """Register KL between Delta and Multivariate Normal."""
    return q_dist.log_prob(p_dist.mean)


class ApproxCholeskyVariationalDistribution(CholeskyVariationalDistribution):
    """Approximate Cholesky variational distribution.

    variational_distribution: N(mean, covariance)
    approx_variational_distribution: N(mean, covariance)
    """

    def __init__(self, num_inducing_points, batch_shape=torch.Size([]), **kwargs):
        super().__init__(num_inducing_points, batch_shape, **kwargs)
        self.sample = None

    @property
    def approx_variational_distribution(self):
        """Approximate variaitonal distribution."""
        return self.variational_distribution

    def resample(self):
        """Resample approximation."""
        pass


class DeltaVariationalDistribution(ApproxCholeskyVariationalDistribution):
    """Variational distribution approximated with a single particle.

    It is equivalent to doing MAP inference.

    variational_distribution: Delta(mean)
    approx_variational_distribution: Delta(mean)

    """

    def __init__(self, num_inducing_points: int, batch_size=None,
                 mean_init_std=1e-3, **kwargs):
        super().__init__(num_inducing_points=num_inducing_points,
                         batch_size=batch_size)
        batch_shape = torch.Size([batch_size])
        mean_init = torch.zeros(num_inducing_points)
        mean_init = mean_init.repeat(*batch_shape, 1)
        self.mean_init_std = mean_init_std
        self.register_parameter(name="variational_mean",
                                parameter=torch.nn.Parameter(mean_init, True))

    @property
    def variational_distribution(self):
        """Build and return variational distribution."""
        return Delta(self.variational_mean)

    @property
    def approx_variational_distribution(self):
        """Build and return variational distribution."""
        return self.variational_distribution

    def initialize_variational_distribution(self, prior_dist):
        """Initialize variational distribution."""
        self.variational_mean.data.copy_(prior_dist.mean)
        self.variational_mean.data.add_(self.mean_init_std,
                                        torch.randn_like(prior_dist.mean))


class CholeskySampleVariationalDistribution(ApproxCholeskyVariationalDistribution):
    """Variational distribution approximated with a set of inducing points.

    variational_distribution: N(mean, covariance)
    approx_variational_distribution: Delta(sample from variational_distribution)

    """

    def resample(self):
        """Resample approximation."""
        self.sample = self.variational_distribution.rsample()

    @property
    def approx_variational_distribution(self):
        """Return the variational distribution q(u) that this module represents."""
        if self.sample is None:
            self.resample()
        return Delta(self.sample)


class CholeskyMeanVariationalDistribution(ApproxCholeskyVariationalDistribution):
    """Variational distribution approximated with a set of inducing points.

    variational_distribution: N(mean, covariance)
    approx_variational_distribution: Delta(mean)
    """

    @property
    def approx_variational_distribution(self):
        """Return the variational distribution q(u) that this module represents."""
        return Delta(self.variational_distribution.loc)

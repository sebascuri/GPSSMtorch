"""Emission model for GPSSM's."""
from gpytorch.likelihoods import Likelihood
from torch.nn import ModuleList
from torch import Tensor
from gpytorch.distributions import MultivariateNormal
from typing import List


class Transitions(Likelihood):
    """Transition module."""

    def __init__(self, likelihoods: List[Likelihood]) -> None:
        super().__init__()
        self.likelihoods = ModuleList(likelihoods)

    def __str__(self) -> str:
        """Return transition model parameters as a string."""
        string = ""
        for i in range(len(self.likelihoods)):
            noise_str = str(self.likelihoods[i].noise_covar.noise.detach())
            string += "component {} {}".format(i, noise_str)
        return string

    def forward(self, function_samples: List[Tensor], *args, **kwargs
                ) -> List[MultivariateNormal]:
        """Compute the conditional distribution p(y|f) that defines the likelihoods.

        Parameters
        ----------
        function_samples: list of Tensor.
            Samples from function `f`.

        Returns
        -------
        y: list of MultivariateNormal.
            Distribution object (with same shape as `function_samples`).
        """
        return [l(y) for l, y in zip(self.likelihoods, function_samples)]

    def __call__(self, next_f: List[MultivariateNormal]) -> List[MultivariateNormal]:
        """Call module."""
        return [l(next_f_) for l, next_f_ in zip(self.likelihoods, next_f)]

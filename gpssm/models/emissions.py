"""Implementation of emission models in gpytorch."""
# from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
# import torch
# from torch.distributions import MultivariateNormal
# from torch.nn.functional import softplus


# class Gaussian(GaussianLikelihood):
#     """A gaussian Model.
#
#     It returns N(Cx + d, R)
#
#     Parameters
#     ----------
#     dim_x: int.
#         Dimension of input.
#
#     dim_y: int.
#         Dimension of output.
#
#     noise: torch.tensor, optional (default=identity).
#         Noise matrix.
#
#     trainable_noise: bool, optional (default=True).
#         Flag to indicate if the noise covariance R should be trained.
#
#     trainable_params: bool, optional (default=False).
#         Flag to indicate if the parameter C and d should be trained.
#
#     """
#     def __init__(self, dim_x: int, dim_y: int, noise: torch.tensor = None,
#                  trainable_noise: bool = True, trainable_params: bool = False):
#         assert dim_y <= dim_x, 'The measurements are not more than the states.'
#         self.C = torch.nn.Parameter(torch.zeros((dim_y, dim_x)),
#                                     requires_grad=trainable_params)
#         self.C[:dim_y, :dim_y] = torch.eye(dim_y)
#         self.d = torch.nn.Parameter(torch.zeros(dim_y),
#                                     requires_grad=trainable_params)
#
#         if noise is None:
#             noise = torch.ones(dim_y)
#         if noise.ndimension() == 2:
#             noise = torch.diag(noise)
#
#         noise = inv_softplus(noise)
#         self.noise = torch.nn.Parameter(noise, requires_grad=trainable_noise)
#
#         super().__init__()
#
#     @property
#     def covariance(self) -> torch.tensor:
#         """Return covariance estimate."""
#         return torch.diag(softplus(self.noise))
#
#     def forward(self, state: torch.tensor) -> MultivariateNormal:
#         """Call the emission model.
#
#         Parameters
#         ----------
#         state: torch.tensor.
#             Tensor with shape [batch x state_dim].
#
#         Returns
#         -------
#         measurements: List of torch.Distributions.
#
#         """
#         output = self.C @ state.t()
#         return MultivariateNormal(loc=output, covariance_matrix=self.covariance)

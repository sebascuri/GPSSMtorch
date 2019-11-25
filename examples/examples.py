"""Python Script Template."""
from gpssm.models.components.gp import VariationalGP
from gpytorch.means import ZeroMean
from gpytorch.kernels import RBFKernel, ScaleKernel
import torch

batch_size = 32
inducing_points = 20
num_particles = 10

dim_x, dim_u = 3, 1
batch_k = 1

ip = torch.randn(dim_x, inducing_points, dim_x + dim_u)
mean = ZeroMean()
kernel = ScaleKernel(RBFKernel(batch_size=batch_k, ard_num_dims=dim_x + dim_u), batch_size=batch_k)


gp = VariationalGP(ip, mean, kernel)
print(gp.covar_module.base_kernel.lengthscale.detach())
print(gp.covar_module.outputscale.detach())

gp.train()
x = torch.randn(batch_size, dim_x, dim_x + dim_u, num_particles)
print(gp(x.transpose(-1, -2)))


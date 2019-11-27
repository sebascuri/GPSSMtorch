"""Python Script Template."""
from gpssm.utilities import Experiment, load
from gpssm.models.ssm import SSM
from gpssm.models.components.dynamics import ExactGPModel
import torch
from gpytorch.likelihoods import GaussianLikelihood
import gpytorch
import matplotlib.pyplot as plt


model = load(Experiment('PRSSM', 'Actuator', 0), 'model')[0]  # type: SSM
gp = model.forward_model
# print(model)

gp.eval()
state_input = torch.randn(1, 5, 20)
var_gp_loc = gp(state_input).loc


ip = gp.variational_strategy.inducing_points

var_dist = gp.variational_strategy.variational_distribution.variational_distribution
mu = var_dist.loc
sample = var_dist.rsample()

with gpytorch.settings.debug(False):
    gp_s = ExactGPModel(ip, mu, GaussianLikelihood(), gp.mean_module, gp.covar_module)

    state_input = state_input.expand(4, 1, 5, 20).permute(1, 0, 3, 2)
    print(ip.shape)
    print(state_input.shape)
    sample_gp_loc = gp_s(state_input).loc

    plt.plot(ip[0, :, 0].detach().numpy(),
             var_gp_loc[0, 0].detach().numpy(), '*')
    plt.plot(ip[0, :, 0].detach().numpy(),
             sample_gp_loc[0, 0].detach().numpy(), '*')
    plt.show()
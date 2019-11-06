"""Project main runner file."""

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel

from gpssm.dataset.dataset import Actuator
from gpssm.models.components.gp import VariationalGP, ModelList
from gpssm.models.components.transitions import Transitions
from gpssm.models.components.emissions import Emissions
from gpssm.models.components.recognition_model import OutputRecognition
from gpssm.models.prssm import PRSSM
from gpssm.models.utilities import get_inducing_points
from gpssm.plotters.plot_sequences import plot_predictions
from gpssm.plotters.plot_learning import plot_loss

if __name__ == "__main__":
    # Set hyper-parameters
    inducing_points_conf = {
        'number': 20,
        'strategy': 'normal',
        'scale': 2.0
    }

    sequence_length = 50
    recognition_length = 1
    num_particles = 20
    batch_size = 16
    learn_inducing_loc = True
    dim_states = 4
    learning_rate = 0.1
    loss_factors = np.array([1.0, 0])
    num_epochs = 5
    loss_key = 'elbo'

    train_set = Actuator(train=True, sequence_length=sequence_length)

    test_set = Actuator(train=False, sequence_length=512)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    dim_inputs = train_set.dim_inputs
    dim_outputs = train_set.dim_outputs

    # Initialize Components
    gps = []
    transitions = []

    mean = ConstantMean()
    kernel = ScaleKernel(RBFKernel(ard_num_dims=dim_states + dim_inputs))
    for _ in range(dim_states):
        inducing_points = get_inducing_points(inducing_points_conf['number'],
                                              dim_states + dim_inputs,
                                              inducing_points_conf['strategy'],
                                              inducing_points_conf['scale'])

        gp = VariationalGP(inducing_points, mean, kernel, learn_inducing_loc)

        # gp.variational_strategy.variational_distribution.variational_mean = (
        #     nn.Parameter(0.05 ** 2 * torch.ones(num_inducing_points))
        # )
        #
        # gp.variational_strategy.variational_distribution.chol_variational_covar = (
        #     nn.Parameter(0.01 ** 2 * torch.eye(num_inducing_points))
        # )

        gp.covar_module.outputscale = 0.5 ** 2
        gp.covar_module.base_kernel.lengthscale = torch.tensor([2.] * (
                dim_states + dim_inputs))
        gps.append(gp)

        transition = GaussianLikelihood()
        transition.noise_covar.noise = 0.02 ** 2
        transitions.append(transition)

    gps = ModelList(gps)
    transitions = Transitions(transitions)

    emissions = []
    for _ in range(dim_outputs):
        emission = GaussianLikelihood()
        emission.noise_covar.noise = 0.1
        emissions.append(emission)

    emissions = Emissions(likelihoods=emissions)

    recognition = OutputRecognition(dim_states=dim_states)  # , sd_noise=0.1)

    # Initialize Model and Optimizer
    model = PRSSM(
        gp_model=gps,
        transitions=transitions,
        emissions=emissions,
        recognition_model=recognition,
        loss_factors=loss_factors,
        num_particles=num_particles
    )
    print(model)
    optimizer = torch.optim.Adam(model.properties(), lr=learning_rate)

    # Train
    losses = []
    for epochs in tqdm(range(num_epochs)):
        for idx, (inputs, outputs, states) in enumerate(train_loader):
            # Zero the gradients of the Optimizer
            optimizer.zero_grad()

            # Compute the elbo
            loss = model.loss(outputs, inputs, states, loss_key)

            # Back-propagate
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            # print(idx, loss.item())
        print(model)

    fig = plot_loss(losses, ylabel='ELBO')
    fig.show()

    # Predict
    with torch.no_grad():
        for inputs, outputs, states in test_loader:
            model.gp.eval()
            predicted_outputs = model.forward(outputs[:, :recognition_length], inputs)

            mean = predicted_outputs.loc.detach().numpy()
            var = predicted_outputs.covariance_matrix.detach().numpy()
            var = np.diagonal(var, axis1=-2, axis2=-1)

            fig = plot_predictions(mean[0].T, np.sqrt(var[0]).T,
                                   outputs[0].detach().numpy().T,
                                   inputs[0].detach().numpy().T)
            fig.show()

            break

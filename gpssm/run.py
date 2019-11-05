"""Project main runner file."""

import numpy as np
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
    num_epochs = 1
    num_inducing_points = 20
    strategy = 'random'
    sequence_length = 50
    recognition_length = 1
    num_particles = 50
    batch_size = 16
    learn_inducing_loc = True
    dim_states = 1
    learning_rate = 0.1
    loss_factors = [0.5, 0]

    train_set = Actuator(train=True, sequence_length=sequence_length)
    test_set = Actuator(train=False, sequence_length=sequence_length)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=False)

    dim_inputs = train_set.dim_inputs
    dim_outputs = train_set.dim_outputs

    # Initialize Components
    gps = []
    transitions = []
    for _ in range(dim_states):
        inducing_points = get_inducing_points(num_inducing_points,
                                              dim_states + dim_inputs,
                                              strategy)
        mean = ConstantMean()
        kernel = ScaleKernel(RBFKernel(ard_num_dims=dim_states + dim_inputs))
        gp = VariationalGP(inducing_points, mean, kernel, learn_inducing_loc)

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
        emission.noise_covar.noise = 1.
        emissions.append(emission)

    emissions = Emissions(likelihoods=emissions)

    recognition = OutputRecognition(dim_states=dim_states)

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
    for epochs in range(num_epochs):
        for idx, (inputs, outputs, states) in enumerate(train_loader):
            # Zero the gradients of the Optimizer
            optimizer.zero_grad()

            # Compute the elbo
            elbo = model.elbo(outputs, inputs, states)

            # Back-propagate
            elbo.backward()
            optimizer.step()

            losses.append(elbo.item())
            print(idx, elbo.item())
            break
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

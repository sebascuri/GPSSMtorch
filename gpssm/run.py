"""Project main runner file."""

import numpy as np
import torch
from torch import Size
from torch.utils.data import DataLoader
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel

from gpssm.dataset.dataset import Actuator
from gpssm.models.components.gp import VariationalGP
from gpssm.models.components.emissions import GaussianEmission
from gpssm.models.components.recognition_model import OutputRecognition
from gpssm.models.prssm import PRSSM
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Set hyper-parameters
    num_epochs = 50
    num_inducing_points = 50
    sequence_length = 50
    recognition_length = 1
    num_particles = 64
    batch_size = 16
    learn_inducing_loc = True
    dim_states = 1  # dataset.dim_states
    learning_rate = 0.01
    loss_factors = [1, 0]

    train_set = Actuator(train=True, sequence_length=sequence_length)
    test_set = Actuator(train=False, sequence_length=512)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    dim_inputs = train_set.dim_inputs
    dim_outputs = train_set.dim_outputs

    # Initialize Components
    inducing_points = torch.randn(
        (dim_states, num_inducing_points, dim_states + dim_inputs))

    mean = ConstantMean(batch_shape=Size([dim_states]))
    kernel = ScaleKernel(
        RBFKernel(batch_shape=Size([dim_states]),
                  ard_num_dims=dim_states + dim_inputs),
        batch_shape=Size([dim_states]))

    gp = VariationalGP(inducing_points, mean, kernel, learn_inducing_loc)

    emissions = GaussianEmission(dim_states=dim_states, dim_outputs=dim_outputs)
    transition = GaussianLikelihood(batch_size=dim_states)
    recognition = OutputRecognition(dim_states=dim_states)

    # Initialize Model and Optimizer
    model = PRSSM(
        gp_model=gp,
        transitions=transition,
        emissions=emissions,
        recognition_model=recognition,
        loss_factors=loss_factors,
        num_particles=num_particles
    )
    optimizer = torch.optim.Adam(model.properties(), lr=learning_rate)

    # Train
    losses = []
    for epochs in range(num_epochs):
        for idx, (inputs, outputs, states) in enumerate(train_loader):
            # Zero the gradients of the Optimizer
            batch_size = inputs.shape[0]
            optimizer.zero_grad()

            # Compute the elbo
            elbo = torch.tensor(0.)
            for i_batch in range(batch_size):
                elbo += model.elbo(outputs[i_batch], inputs[i_batch], states[i_batch])
            elbo = elbo / batch_size

            # Back-propagate
            elbo.backward()
            optimizer.step()

            losses.append(elbo.item())
            print(idx, elbo.item())

    plt.plot(losses)
    plt.show()

    # Predict
    with torch.no_grad():
        for inputs, outputs, states in test_loader:
            predicted_outputs = model.forward(outputs[0, :recognition_length],
                                              inputs[0])

            mean = predicted_outputs.loc.detach().numpy()
            std = predicted_outputs.covariance_matrix.detach().numpy()
            std = np.diagonal(std, axis1=1, axis2=2)
        plt.plot(mean, 'b')
        plt.plot(outputs[0].numpy(), 'r')
        plt.fill_between(np.arange(512), (mean - std)[:, 0], (mean + std)[:, 0],
                         alpha=0.2)
        plt.show()

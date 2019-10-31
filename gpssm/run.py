"""Project main runner file."""

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
    num_epochs = 4
    num_inducing_points = 25
    sequence_length = 50
    num_particles = 64
    batch_size = 16
    learn_inducing_loc = True
    dim_states = 1  # dataset.dim_states
    learning_rate = 0.01
    loss_factors = [1, 0]

    dataset = Actuator(train=True, sequence_length=sequence_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dim_inputs = dataset.dim_inputs
    dim_outputs = dataset.dim_outputs

    # Initialize Components
    inducing_points = torch.randn(
        (dim_states, num_inducing_points, dim_states + dim_inputs))

    mean = ConstantMean(batch_shape=Size([dim_states]))
    kernel = ScaleKernel(
        RBFKernel(batch_shape=Size([dim_states]),
                  ard_num_dims=dim_states + dim_inputs),
        batch_shape=Size([dim_states]))

    gp = VariationalGP(inducing_points, mean, kernel, batch_size, learn_inducing_loc)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    for epochs in range(num_epochs):
        for idx, (inputs, outputs, states) in enumerate(data_loader):
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

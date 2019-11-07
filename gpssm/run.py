"""Project main runner file."""

import numpy as np
from tqdm import tqdm
import torch

from torch.utils.data import DataLoader

from gpssm.evaluator import Evaluator
from gpssm.dataset.dataset import Actuator
from gpssm.models.components.recognition_model import OutputRecognition
from gpssm.models.prssm import PRSSM
from gpssm.models.utilities import init_emissions, init_transmissions, init_gps
from gpssm.plotters.plot_sequences import plot_predictions
from gpssm.plotters.plot_learning import plot_loss
import yaml


if __name__ == "__main__":
    # Set hyper-parameters
    config_file = 'experiments/small_scale.yaml'
    config = yaml.load(open(config_file), Loader=yaml.SafeLoader)
    sequence_length = config.get('sequence_length', 20)
    recognition_length = config.get('recognition_length', 10)
    num_particles = 20
    batch_size = 16
    learn_inducing_loc = True
    dim_states = 4
    learning_rate = 0.1
    loss_factors = np.array([1.0, 0])
    num_epochs = 1
    loss_key = 'elbo'

    train_set = Actuator(train=True, sequence_length=sequence_length)

    test_set = Actuator(train=False, sequence_length=512)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    dim_inputs = train_set.dim_inputs
    dim_outputs = train_set.dim_outputs

    gps = init_gps(dim_states, dim_inputs,
                   inducing_points_number=20,
                   inducing_points_strategy='normal',
                   inducing_points_scale=2.0,
                   kernel_str='rbf',
                   kernel_lengthscale=2.0,
                   kernel_outputscale=0.1,
                   mean_str='constant',
                   shared=False
                   )
    transitions = init_transmissions(dim_states, initial_variance=0.0001,
                                     learnable=True, shared=False)
    emissions = init_emissions(dim_outputs, initial_variance=0.1,
                               learnable=True, shared=False)

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
    evaluator = Evaluator()
    with torch.no_grad():
        for inputs, outputs, states in test_loader:
            model.gp.eval()
            predicted_outputs = model.forward(outputs[:, :recognition_length], inputs)
            mean = predicted_outputs.loc.detach().numpy()
            var = predicted_outputs.covariance_matrix.detach().numpy()

            print(evaluator.evaluate(predicted_outputs, outputs))
            var = np.diagonal(var, axis1=-2, axis2=-1)

            fig = plot_predictions(mean[0].T, np.sqrt(var[0]).T,
                                   outputs[0].detach().numpy().T,
                                   inputs[0].detach().numpy().T)
            fig.show()

            break

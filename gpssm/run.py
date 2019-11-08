"""Project main runner file."""

import numpy as np
from tqdm import tqdm
import torch

from torch.utils.data import DataLoader

from gpssm.evaluator import Evaluator
from gpssm.dataset import get_dataset
from gpssm.models.prssm import PRSSM
from gpssm.models.initializers import init_emissions, init_transmissions, init_gps, \
    init_recognition
from gpssm.models.utilities import approximate_with_normal
from gpssm.plotters.plot_sequences import plot_predictions
from gpssm.plotters.plot_learning import plot_loss
import yaml

if __name__ == "__main__":
    # Set hyper-parameters
    config_file = 'experiments/small_scale.yaml'
    config = yaml.load(open(config_file), Loader=yaml.SafeLoader)
    sequence_length = config.get('sequence_length', 50)
    dim_states = config.get('dim_state', 1)
    batch_size = config.get('batch_size', 32)
    num_epochs = config.get('num_epochs', 5)
    learning_rate = config.get('learning_rate', 0.1)
    loss_key = config.get('loss_key', 'elbo')

    for dataset in config.get('dataset'):
        dataset_ = get_dataset(dataset)
        train_set = dataset_(train=True, sequence_length=sequence_length)
        test_set = dataset_(train=False)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        dim_inputs = train_set.dim_inputs
        dim_outputs = train_set.dim_outputs

        gps = init_gps(dim_states, dim_inputs, **config.get('forward_gps'))
        transitions = init_transmissions(dim_states, **config.get('transitions'))
        emissions = init_emissions(dim_outputs, **config.get('emissions'))
        recognition = init_recognition(dim_outputs, dim_inputs, dim_states,
                                       **config.get('recognition'))

        # Initialize Model and Optimizer
        model = PRSSM(
            forward_model=gps,
            transitions=transitions,
            emissions=emissions,
            recognition_model=recognition,
            num_particles=config['model']['num_particles']
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
                predicted_outputs = model(outputs, inputs)
                loss = model.loss(predicted_outputs, outputs, inputs, loss_key)

                # Back-propagate
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
            print(model)

        fig = plot_loss(losses, ylabel='ELBO')
        fig.show()

        # Predict
        evaluator = Evaluator()
        with torch.no_grad():
            model.eval()
            for inputs, outputs, states in test_loader:
                predicted_outputs = model(outputs, inputs)
                predicted_outputs = approximate_with_normal(predicted_outputs)
                mean = predicted_outputs.loc.detach().numpy()
                var = predicted_outputs.covariance_matrix.detach().numpy()

                print(dataset)
                print(evaluator.evaluate(predicted_outputs, outputs))
                var = np.diagonal(var, axis1=-2, axis2=-1)

                for i in range(mean.shape[0]):
                    fig = plot_predictions(mean[i].T, np.sqrt(var[i]).T,
                                           outputs[i].detach().numpy().T,
                                           inputs[i].detach().numpy().T)
                    fig.axes[0].set_title(dataset + ' Predictions')
                    fig.show()

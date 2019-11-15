"""Project main runner file."""

import torch
from torch.utils.data import DataLoader

from gpssm.dataset import get_dataset
from gpssm.models import get_model
from gpssm.models.initializers import init_emissions, init_transmissions, init_gps, \
    init_recognition
from gpssm.utilities.utilities import train, evaluate
from gpssm.plotters.plot_learning import plot_loss
import yaml

if __name__ == "__main__":
    # Set hyper-parameters
    torch.manual_seed(0)

    config_file = 'experiments/small_scale.yaml'
    config = yaml.load(open(config_file), Loader=yaml.SafeLoader)
    sequence_length = config.get('sequence_length', 50)
    dim_states = config.get('dim_state', 1)
    batch_size = config.get('batch_size', 32)
    num_epochs = config.get('num_epochs', 5)
    num_particles = config.get('num_particles', 50)
    learning_rate = config.get('learning_rate', 0.1)
    loss_key = config.get('loss_key', 'elbo')

    for dataset in config.get('dataset'):
        dataset_ = get_dataset(dataset)
        dim_inputs = dataset_.dim_inputs
        dim_outputs = dataset_.dim_outputs

        gps = init_gps(dim_states, dim_inputs, **config.get('forward_gps'))
        transitions = init_transmissions(dim_states, **config.get('transitions'))
        emissions = init_emissions(dim_outputs, **config.get('emissions'))
        recognition = init_recognition(dim_outputs, dim_inputs, dim_states,
                                       **config.get('recognition'))

        # Initialize Model and Optimizer
        model = get_model(config.get('model'))(
            forward_model=gps,
            transitions=transitions,
            emissions=emissions,
            recognition_model=recognition,
            num_particles=num_particles
        )
        optimizer = torch.optim.Adam(model.properties(), lr=learning_rate)

        # Train
        train_set = dataset_(train=True, sequence_length=sequence_length)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        losses = train(model, train_loader, optimizer, num_epochs)
        fig = plot_loss(losses, ylabel='ELBO')
        fig.show()

        # Predict
        test_set = dataset_(train=False)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        evaluate(model, test_loader)

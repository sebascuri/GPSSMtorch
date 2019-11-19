"""Project main runner file."""

import torch
from torch.utils.data import DataLoader
from gpssm.dataset import get_dataset
from gpssm.models import get_model
from gpssm.utilities.utilities import train, evaluate
from gpssm.plotters.plot_learning import plot_loss


def main(model_name: str, dataset_name: str, config: dict, seed: int = 0,
         num_threads: int = 2):
    """Run GPSSM inference.

    Parameters
    ----------
    model_name: str.
        Name of GP-SSM model.
    dataset_name: str.
        Name of dataset.
    config: dict.
        Dictionary with configurations.
    seed: int.
        Random seed.
    num_threads: int.
        Number of threads.

    # TODO: implement device.
    """
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)

    # Dataset Parameters
    dataset_config = config.get('dataset', {})

    # Optimizer Parameters
    optim_config = config.get('optimization', {})
    batch_size = optim_config.get('batch_size', 32)
    num_epochs = optim_config.get('num_epochs', 5)
    learning_rate = optim_config.get('learning_rate', 0.01)

    # Model Parameters
    model_config = config.get('model', {})

    # Plot Parameters  # TODO: Change to outputs
    plot_config = config.get('plots', ['prediction'])

    # Initialize dataset, model, and optimizer.
    dataset = get_dataset(dataset_name)
    model = get_model(model_name, dataset.dim_outputs, dataset.dim_inputs,
                      **model_config)
    optimizer = torch.optim.Adam(model.properties(), lr=learning_rate)

    # Train
    train_set = dataset(train=True, **dataset_config)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    losses = train(model, train_loader, optimizer, num_epochs)
    fig = plot_loss(losses, ylabel='ELBO')
    fig.show()

    # Predict  # TODO: Predict with two different lengths.
    test_set = dataset(train=False, **dataset_config)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # evaluate(model, test_loader, config['plots'])

    test_set.sequence_length = test_set.experiment_length
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    evaluate(model, test_loader, plot_config)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='Run GP-SSM.')
    parser.add_argument('--model', type=str, default='PRSSM', help='GPSSM Model.')
    parser.add_argument('--dataset', type=str, default='Actuator', help='Dataset.')
    parser.add_argument('--config-file', type=str, default=None, help='Dataset.')
    parser.add_argument('--seed', type=int, default=0, help='Seed.')
    parser.add_argument('--num-threads', type=int, default=2, help='Number Threads.')
    parser.add_argument('--device', type=str, default='cpu', help='Device.')
    args = parser.parse_args()

    if args.config_file is not None:
        with open(args.config_file, 'r') as file:
            configs = yaml.load(file, Loader=yaml.SafeLoader)
        configs.get('model', {}).pop('name', {})
        configs.get('dataset', {}).pop('name', {})
    else:
        configs = {}

    main(args.model, args.dataset, configs, args.seed)

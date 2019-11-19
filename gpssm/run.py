"""Project main runner file."""

import torch
from torch.utils.data import DataLoader
from gpssm.dataset import get_dataset
from gpssm.models import get_model
from gpssm.utilities.utilities import train, evaluate, Experiment, save, experiment_dir
from gpssm.plotters.plot_learning import plot_loss


def main(experiment: Experiment, num_threads: int = 2):
    """Run GPSSM inference.

    Parameters
    ----------
    experiment: Experiment.
        Experiment identifier.
    num_threads: int.
        Number of threads.

    # TODO: implement device.
    """
    exp_dir = experiment_dir(experiment)
    torch.manual_seed(experiment.seed)
    torch.set_num_threads(num_threads)

    # Dataset Parameters
    dataset_config = experiment.configs.get('dataset', {})

    # Optimizer Parameters
    optim_config = experiment.configs.get('optimization', {})
    batch_size = optim_config.get('batch_size', 32)
    num_epochs = optim_config.get('num_epochs', 5)
    learning_rate = optim_config.get('learning_rate', 0.01)

    # Model Parameters
    model_config = experiment.configs.get('model', {})

    # Plot Parameters
    eval_config = experiment.configs.get('evaluation', {})
    plot_list = eval_config.get('plots', [])

    # Initialize dataset, model, and optimizer.
    dataset = get_dataset(experiment.dataset)
    model = get_model(experiment.model, dataset.dim_outputs, dataset.dim_inputs,
                      **model_config)
    optimizer = torch.optim.Adam(model.properties(), lr=learning_rate)

    # Train
    train_set = dataset(train=True, **dataset_config)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    losses = train(model, train_loader, optimizer, num_epochs)
    save(experiment, model=model)

    if 'training_loss' in plot_list:
        fig = plot_loss(losses, ylabel=model.loss_key.upper())
        fig.show()
        fig.savefig('{}training_loss_{}.png'.format(exp_dir, experiment.seed))

    # Evaluate with different sequence lengths.
    test_set = dataset(train=False, **dataset_config)
    eval_lengths = [train_set.sequence_length, test_set.experiment_length]
    eval_lengths += eval_config.get('length', [])
    for eval_length in eval_lengths:
        test_set.sequence_length = eval_length
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        evaluator = evaluate(model, test_loader, experiment, plot_list.copy())
        save(experiment, **{'eval_{}'.format(test_set.sequence_length): evaluator})


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

    main(Experiment(args.model, args.dataset, args.seed, configs), args.num_threads)

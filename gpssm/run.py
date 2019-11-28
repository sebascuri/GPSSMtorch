"""Project main runner file."""

import torch
from torch.utils.data import DataLoader
from gpssm.dataset import get_dataset
from gpssm.models import get_model
from gpssm.utilities import train, evaluate, Experiment, save


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
    torch.manual_seed(experiment.seed)
    torch.set_num_threads(num_threads)

    # Dataset Parameters
    dataset_config = experiment.configs.get('dataset', {})

    # Optimizer Parameters
    optim_config = experiment.configs.get('optimization', {})
    batch_size = optim_config.get('batch_size', 32)
    num_epochs = optim_config.get('num_epochs', 1)
    learning_rate = optim_config.get('learning_rate', 0.1)

    # Model Parameters
    model_config = experiment.configs.get('model', {})

    # Plot Parameters
    eval_config = experiment.configs.get('evaluation', {})
    plot_list = eval_config.get('plots', ['prediction', 'training_loss'])

    # Initialize dataset, model, and optimizer.
    dataset = get_dataset(experiment.dataset)
    model = get_model(experiment.model, dataset.dim_outputs, dataset.dim_inputs,
                      **model_config)
    model.dump(experiment.fig_dir + 'model_initial.txt')
    optimizer = torch.optim.Adam(model.properties(), lr=learning_rate)

    # Train
    train_set = dataset(train=True, **dataset_config)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train(model, train_loader, optimizer, num_epochs, experiment)
    model.dump(experiment.fig_dir + 'model_final.txt')
    save(experiment, model=model)

    # Evaluate on Train Set.
    train_set.sequence_length = train_set.experiment_length
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    evaluator = evaluate(model, train_loader, experiment, plot_list.copy(), 'train')
    save(experiment, eval_train=evaluator)
    evaluator.dump(experiment.fig_dir + 'train_results.txt')

    # Evaluate on Test Set.
    test_set = dataset(train=False, **dataset_config)
    test_set.sequence_length = test_set.experiment_length
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    evaluator = evaluate(model, test_loader, experiment, plot_list.copy(), 'test')
    save(experiment, eval_test=evaluator)
    evaluator.dump(experiment.fig_dir + 'test_results.txt')


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='Run GP-SSM.')
    parser.add_argument('--config-file', type=str, default=None, help='Dataset.')
    parser.add_argument('--seed', type=int, default=0, help='Seed.')
    parser.add_argument('--num-threads', type=int, default=2, help='Number Threads.')
    parser.add_argument('--device', type=str, default='cpu', help='Device.')
    args = parser.parse_args()

    if args.config_file is not None:
        with open(args.config_file, 'r') as file:
            configs = yaml.load(file, Loader=yaml.SafeLoader)
        model = configs.get('model').pop('name')
        dataset = configs.get('dataset').pop('name')
    else:
        configs = {'experiment': {'name': 'experiments/sample/'},
                   'model': {'dim_states': 4}}
        model = 'PRSSMDiag'
        dataset = 'Actuator'

    main(Experiment(model, dataset, args.seed, configs), args.num_threads)

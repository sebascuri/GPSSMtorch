"""Project main runner file."""

import torch
import torch.optim
from torch.utils.data import DataLoader
from gpssm.dataset import get_dataset
from gpssm.models import get_model
from gpssm.utilities import train, evaluate, Experiment, save
import math

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
    if 'max_iter' in optim_config:
        num_epochs = None
    else:
        num_epochs = optim_config.get('num_epochs', 1)
    max_iter = optim_config.get('max_iter', 1)

    learning_rate = optim_config.get('learning_rate', 0.1)
    eval_length = optim_config.get('eval_length', [None])

    # Model Parameters
    model_config = experiment.configs.get('model', {})

    # Plot Parameters
    plot_list = configs.get('plots', ['prediction', 'training_loss'])

    # Initialize dataset, model, and optimizer.
    dataset = get_dataset(experiment.dataset)
    model = get_model(experiment.model, dataset.dim_outputs, dataset.dim_inputs,
                      **model_config)
    model.dump(experiment.fig_dir + 'model_initial.txt')
    optimizer = torch.optim.Adam(model.properties(), lr=learning_rate)

    # Train.
    train_set = dataset(train=True, **dataset_config)
    print(train_set)
    model.dataset_size = len(train_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    if num_epochs is None:
        num_epochs = max(1, math.floor(max_iter * batch_size / len(train_set)))
        print(num_epochs)
    train(model, train_loader, optimizer, num_epochs, experiment)
    model.dump(experiment.fig_dir + 'model_final.txt')
    save(experiment, model=model)

    # Evaluate.
    for key in ['train', 'test']:
        dataset_ = dataset(train=key == 'train', **dataset_config)
        model.dataset_size = len(dataset_)
        for seq_len in eval_length:
            if seq_len is None:
                seq_len = dataset_.experiment_length
            eval_key = '{}_{}'.format(key, seq_len)

            dataset_.sequence_length = seq_len
            loader = DataLoader(dataset_, batch_size=batch_size, shuffle=False)
            evaluator = evaluate(model, loader, experiment, plot_list.copy(), eval_key)
            save(experiment, eval_train=evaluator)
            evaluator.dump(experiment.fig_dir + '{}_results_{}.txt'.format(eval_key,
                                                                           args.seed))


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
                   'model': {'dim_states': 4},
                   'dataset': {'sequence_length': 40},
                   'optimization': {'eval_length': [50],
                                    'max_iter': 1}}
        model = 'softcbfssm'
        dataset = 'Actuator'

    main(Experiment(model, dataset, args.seed, configs), args.num_threads)

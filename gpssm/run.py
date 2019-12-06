"""Project main runner file."""

import torch
import torch.optim
from torch.utils.data import DataLoader
from gpssm.dataset import get_dataset
from gpssm.models import get_model
from gpssm.utilities import train, evaluate, Experiment, save, dump
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

    # Optimization Parameters
    opt_config = experiment.configs.get('optimization', {})
    batch_size = opt_config.get('batch_size', 10)
    num_epochs = None if 'max_iter' in opt_config else opt_config.get('num_epochs', 1)
    max_iter = opt_config.get('max_iter', 1)
    learning_rate = opt_config.get('learning_rate', 0.01)
    eval_length = opt_config.get('eval_length', [None])

    # Model Parameters
    model_config = experiment.configs.get('model', {})

    # Initialize dataset
    dataset = get_dataset(experiment.dataset)
    train_set = dataset(train=True, **dataset_config)
    test_set = dataset(train=False, **dataset_config)
    test_set.sequence_length = test_set.experiment_length
    print(train_set)
    print(test_set)

    # Initialize model
    model = get_model(experiment.model, dataset.dim_outputs, dataset.dim_inputs,
                      **model_config)
    dump(str(model), experiment.fig_dir + 'model_initial.txt')

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.properties(), lr=learning_rate)

    # Train.
    model.dataset_size = len(train_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True
                              # sampler=torch.utils.data.RandomSampler(train_set),
                              )
    if num_epochs is None:
        num_epochs = max(1, math.floor(max_iter * batch_size / len(train_set)))

    losses = train(model, train_loader, optimizer, num_epochs, experiment,
                   test_set=test_set)
    save(experiment, model=model)
    dump(str(model), experiment.fig_dir + 'model_final_{}.txt'.format(experiment.seed))
    dump(str(losses), experiment.fig_dir + 'losses_{}.txt'.format(experiment.seed))

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
            evaluator = evaluate(model, loader, experiment, eval_key)
            save(experiment, eval_train=evaluator)
            dump(str(evaluator), experiment.fig_dir + '{}_results_{}.txt'.format(
                eval_key, experiment.seed))


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
                   'print_iter': 50,
                   'model': {'dim_states': 4,
                             'num_particles': 16,
                             'forward': {
                                 'mean': {'kind': 'zero'},
                                 'kernel': {'shared': True,
                                            'outputscale': 0.25,
                                            'lengthscale': 2.},
                                 'inducing_points': {
                                     'strategy': 'uniform',
                                     'scale': 4.0,
                                     'learnable': False,
                                     'number_points': 20}
                             },
                             'emissions': {'variance': .1, 'learnable': True}
                             },
                   'dataset': {'sequence_length': 50},
                   'optimization': {'eval_length': [None],
                                    'max_iter': 300}}
        model = 'CBFSSM'
        dataset = 'Actuator'

    main(Experiment(model, dataset, args.seed, configs), args.num_threads)

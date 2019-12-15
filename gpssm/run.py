"""Project main runner file."""

import torch
import torch.optim
from gpssm.dataset import get_dataset
from gpssm.models import get_model
from gpssm.utilities import train, Experiment


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
    learning_rate = opt_config.get('learning_rate', 0.01)
    eval_length = opt_config.get('eval_length', None)

    # Model Parameters
    model_config = experiment.configs.get('model', {})

    # Initialize dataset
    dataset = get_dataset(experiment.dataset)
    train_set = dataset(train=True, **dataset_config)
    test_set = dataset(train=False, **dataset_config)
    if eval_length is None:
        test_set.sequence_length = test_set.experiment_length
    else:
        test_set.sequence_length = eval_length
    print(train_set)
    print(test_set)

    # Initialize model
    model = get_model(experiment.model, dataset.dim_outputs, dataset.dim_inputs,
                      **model_config)
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.properties(), lr=learning_rate)

    # Train & Evaluate.
    model.dataset_size = len(train_set)
    train(model, optimizer, experiment, train_set=train_set, test_set=test_set)


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
        model_ = configs.get('model').pop('name')
        dataset_ = configs.get('dataset').pop('name')
    else:
        configs = {'experiment': {'name': 'experiments/sample/'},
                   'verbose': 2,
                   'model': {'dim_states': 4,
                             'num_particles': 50,
                             'loss_factors': {'kl_u': .5, 'kl_conditioning': 1.},
                             'recognition': {'kind': 'conv', 'length': 16,
                                             'variance': 0.1 ** 2},
                             'forward': {
                                 'mean': {'kind': 'zero'},
                                 'kernel': {'shared': True,
                                            'outputscale': .5 ** 2,
                                            'lengthscale': 2.},
                                 'inducing_points': {
                                     'strategy': 'uniform',
                                     'scale': 4.0,
                                     'learnable': True,
                                     'number_points': 20},
                                 'variational_distribution': {
                                     'mean': 0.05 ** 2,
                                     'variance': 0.01 ** 2,
                                 }
                             },
                             'emissions': {'variance': 1., 'learnable': True},
                             'transitions': {'variance': 0.002 ** 2, 'learnable': True}
                             },
                   'dataset': {'sequence_length': 50},
                   'optimization': {'learning_rate': 0.005,
                                    'batch_size': 10,
                                    'num_epochs': 50}
                   }
        model_ = 'PRSSM'
        dataset_ = 'Actuator'

    main(Experiment(model_, dataset_, args.seed, configs), args.num_threads)

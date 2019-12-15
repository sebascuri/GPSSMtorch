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
                   'model': {'dim_states': 1,
                             'independent_particles': True,
                             'num_particles': 100,
                             'loss_key': 'elbo',
                             'loss_factors': {'kl_u': .1, 'kl_conditioning': 1.},
                             'recognition': {'kind': 'conv', 'length': 20,
                                             'learnable': True,
                                             'variance': 0.004},
                             'forward': {
                                 'mean': {'kind': 'zero'},
                                 'kernel': {'shared': True,
                                            'kind': 'matern32',
                                            'outputscale': 0.5,
                                            'lengthscale': 2.0
                                            },
                                 'inducing_points': {
                                     'strategy': 'normal',
                                     'scale': 2.0,
                                     'learnable': False,
                                     'number_points': 20},
                                 'variational_distribution': {
                                     'kind': 'mean',
                                     # 'learn_mean': False,
                                 }
                             },
                             'emissions': {'variance':  0.04, 'learnable': True},
                             'transitions': {'variance': 0.0025, 'learnable': True}
                             },
                   'dataset': {'sequence_length': 20},
                   'optimization': {'learning_rate': 0.1,
                                    'batch_size': 1,
                                    'num_epochs': 200}
                   }
        model_ = 'VCDT'
        dataset_ = 'KinkFunction'

    main(Experiment(model_, dataset_, args.seed, configs), args.num_threads)

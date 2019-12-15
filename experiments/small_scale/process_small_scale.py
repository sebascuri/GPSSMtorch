"""Python Script Template."""
from gpssm.plotters import plot_evaluation_datasets
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

SPLITS = ['test', 'train', 'last']


def parse_file(file_name):
    """Parse a file."""
    with open(file_name) as file:
        data = file.readlines()
    test = data[-2].split('. ')
    train = data[-1].split('. ')
    last_epoch = data[-3].split('. ')

    test[0] = 'L' + test[0].split(' L')[1]
    train[0] = 'L' + train[0].split(' L')[1]
    last_epoch[0] = 'L' + last_epoch[0].split(' L')[1]


    results = {}
    for split, value in zip(SPLITS, [test, train, last_epoch]):
        results[split] = {}
        for key_val in value:
            key, val = key_val.split(':')
            results[split][key.lower()] = float(val)
    return results


def process_method(method, datasets, keys, seeds):
    print(method)
    losses = {}
    for dataset in datasets:
        if dataset not in losses:
            losses[dataset] = {}

        for k, seed in product(keys, seeds):
            key = ('{}/'*len(k)).format(*k)
            if key not in losses[dataset]:
                losses[dataset][key] = {}

            file_name = '{}/{}/{}train_epoch_{}.txt'.format(method, dataset, key, seed)
            results = parse_file(file_name)

            for split in results:
                for loss, val in results[split].items():
                    if loss not in losses[dataset][key]:
                        losses[dataset][key][loss] = {}
                    if split not in losses[dataset][key][loss]:
                        losses[dataset][key][loss][split] = []

                    losses[dataset][key][loss][split].append(val)

    for dataset in datasets:
        print(dataset)
        min_loss = float('inf')
        min_key = None
        for key in losses[dataset]:
            if -np.mean(losses[dataset][key]['log-lik']['test']) < min_loss:
                min_key = key
                min_loss = -np.mean(losses[dataset][key]['log-lik']['last'])
        rmse = losses[dataset][min_key]['rmse']['test']
        loglik = losses[dataset][min_key]['log-lik']['test']
        print(min_key, '{:.3} ({:.2})'.format(np.mean(rmse), np.std(rmse, ddof=1)),
              '{:.3} ({:.2})'.format(-np.mean(loglik), np.std(loglik, ddof=1))
              )


# def process_method(method_name, datasets, var_dists, k_us, k_factor):

datasets = ['Actuator', 'BallBeam', 'Drive', 'Dryer', 'GasFurnace', 'Flutter', 'Tank']
var_dists = ['full', 'delta', 'full', 'mean', 'sample']
k_us = ['0.1', '0.01', '0.05']
k_factors = ['1', '10', '50']

process_method('CBFSSM', datasets, list(product(
    ['sample', 'delta', 'full', 'mean'],
    # ['sample'],
    ['0.1', '0.01', '0.05'],
    ['1', '10', '50'],
    # ['50']
)),
               [0, 1, 2])
process_method('VCDT', datasets, list(product(
    ['sample'],
    # [ sample, 'mean', 'delta'],
    ['0.1'],
    # ['0.1', '0.01', '0.05'],
    )),
            [0, 1, 2, 3, 4]
               )

process_method('PRSSM', datasets, list(product(
    ['full'], #, 'mean', 'delta'],
    ['0.1'] #, '0.01', '0.05']
    )),
               [0, 1, 2, 3, 4]
               )

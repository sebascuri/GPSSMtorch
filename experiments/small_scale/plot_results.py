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


def process_method(method, datasets, keys):
    print(method)
    losses = {}
    for dataset in datasets:
        if dataset not in losses:
            losses[dataset] = {}

        for k in keys:
            key = ('{}/'*len(k)).format(*k)
            if key not in losses[dataset]:
                losses[dataset][key] = {}

            try:
                file_name = '{}/{}/{}train_epoch.txt'.format(method, dataset, key)
                results = parse_file(file_name)
            except FileNotFoundError:
                file_name = '{}/{}/{}train_epoch_0.txt'.format(method, dataset, key)
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
            if np.mean(losses[dataset][key]['rmse']['last']) < min_loss:
                min_key = key
                min_loss = np.mean(losses[dataset][key]['rmse']['test'])
        print(min_key,
              losses[dataset][min_key]['rmse'],
              losses[dataset][min_key]['log-lik'])


# def process_method(method_name, datasets, var_dists, k_us, k_factor):

datasets = ['Actuator', 'BallBeam', 'Drive', 'Dryer', 'GasFurnace', 'Flutter', 'Tank']
var_dists = ['full', 'delta', 'full', 'mean', 'sample']
k_us = ['0.1', '0.01', '0.05']
k_factors = ['1', '10', '50', '100']

process_method('CBFSSM', datasets, list(product(
    ['delta', 'full', 'mean', 'sample'],
    ['0.1', '0.01', '0.05'],
    ['1', '10', '50', '100'])))
process_method('VCDT', datasets, list(product(
    ['sample'],
    ['0.1', '0.01', '0.05'])))

process_method('PRSSM', datasets, [('',)])

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

    results = {split: {'log-lik': [], 'nrmse': [], 'rmse': []} for split in SPLITS}
    min_val = 0

    for line in reversed(data):
        data_line = line.split('. ')
        split = data_line[0].split(' ')[0]
        data_line[0] = 'L' + data_line[0].split(' L')[1]

        for key_val in data_line:
            key, val = key_val.split(':')

            if split.lower() == 'train':
                results['train'][key.lower()].append(float(val))
            elif split.lower() == 'test':
                results['test'][key.lower()].append(float(val))
            else:
                val = int(split)
                if val < min_val:
                    return results
                min_val = val
                results['last'][key.lower()].append(float(val))


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
            if losses[dataset][key]['rmse']['test'] == [[]]:
                continue
            if np.mean(losses[dataset][key]['rmse']['test']) < min_loss:
                min_key = key
                min_loss = np.mean(losses[dataset][key]['rmse']['test'])
        rmse = losses[dataset][min_key]['rmse']['test']
        loglik = losses[dataset][min_key]['log-lik']['test']
        print(min_key,
              np.mean(rmse), np.std(rmse, ddof=1),
              np.mean(loglik), np.std(loglik, ddof=1)
              )


# def process_method(method_name, datasets, var_dists, k_us, k_factor):

datasets = ['RobomoveSimple', 'Robomove']
var_dists = ['full', 'delta', 'full', 'mean', 'sample']
k_factors = ['1', '10', '50']

process_method('CBFSSM', datasets, list(product(
    ['delta', 'full', 'mean', 'sample'],
    ['bi-lstm', 'conv', 'lstm'],  #, 'output'],
    ['1', '10', '50'],
)), [0])

process_method('VCDT', datasets, list(product(
    ['delta', 'mean', 'sample'],
    ['bi-lstm', 'conv', 'lstm'],  #, 'output'],
)), [0])

process_method('PRSSM', datasets, list(product(
    ['full', 'mean'],
    ['bi-lstm', 'conv', 'lstm'], #, 'output'],
)), [0])

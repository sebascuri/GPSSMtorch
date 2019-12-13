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


def process_method(method, keys, seeds):
    print(method)
    losses = {}

    for k, seed in product(keys, seeds):
        key = ('{}/'*len(k)).format(*k)
        if key not in losses:
            losses[key] = {}

        file_name = '{}/{}train_epoch_{}.txt'.format(method, key, seed)
        results = parse_file(file_name)

        for split in results:
            for loss, val in results[split].items():
                if loss not in losses[key]:
                    losses[key][loss] = {}
                if split not in losses[key][loss]:
                    losses[key][loss][split] = []

                losses[key][loss][split].append(np.mean(val))

    min_loss = float('inf')
    min_key = None
    for key in losses:
        if np.mean(losses[key]['rmse']['test']) < min_loss:
            min_key = key
            min_loss = np.mean(losses[key]['rmse']['test'])
    rmse = losses[min_key]['rmse']['test']
    loglik = losses[min_key]['log-lik']['test']
    print(min_key,
          np.mean(rmse), np.std(rmse, ddof=1),
          np.mean(loglik), np.std(loglik, ddof=1)
          )


process_method('CBFSSM', list(product(
    ['full', 'sample', 'mean', 'delta'],
    ['0.01'],  # ['0.01', '0.001'],
    ['10'],  # ['5', '10'],
    ['0.01'],  # ['0.1', '0.01', '1.0'],
    ['1.0', '10.0', '50.0'],  # ['1.0', '10.0', '50.0']
)),
               [0, 1, 2] #, 3, 4]
)
process_method('VCDT', list(product(
    ['sample', 'mean', 'delta'],
    ['0.01'],  # ['0.01', '0.001'],
    ['10'],  # ['5', '10'],
    ['0.1'],  # ['0.1', '0.01', '1.0'],
    )),
               [0, 1, 2, 3, 4]
               )
process_method('PRSSM', list(product(
    ['full'],  #, 'mean', 'delta'],
    ['0.01'],  # ['0.01', '0.001'],
    ['10'],  # ['5', '10'],
    ['0.1'],  # ['0.1', '0.01', '1.0'],

)),
               [0, 1, 2, 3, 4]
               )


# CBFSSM
# full/0.01/10/0.01/50.0/
# VCDT
# delta/0.01/10/0.1/
# PRSSM
# mean/0.01/10/0.1/

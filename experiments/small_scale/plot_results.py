"""Python Script Template."""
from gpssm.plotters import plot_evaluation_datasets
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def parse_file(file_name):
    """Parse a file."""
    with open(file_name) as file:
        data = file.readlines()
    test = data[-2].split('. ')
    test[0] = 'L' + test[0].split(' L')[1]

    results = {}
    for key_val in test:
        key, val = key_val.split(':')
        results[key.lower()] = float(val)
    return results

methods = ['PRSSM'] #, 'VCDT', 'CBFSSM']
datasets = ['Actuator', 'BallBeam', 'Drive', 'Dryer', 'GasFurnace', 'Flutter', 'Tank']
# seeds = [0, 1, 2, 3, 4]

# fig, axes = plt.subplots(len(datasets), 1, sharex=True)
fig = plt.figure()
losses = {}
for method, dataset in product(methods, datasets):
    if method not in losses:
        losses[method] = {}
    if dataset not in losses[method]:
        losses[method][dataset] = {'rmse': [], 'nrmse': [], 'log-lik': []}

    file_name = '{}/{}/train_epoch.txt'.format(method, dataset)
    results = parse_file(file_name)
    for key, val in results.items():
        losses[method][dataset][key].append(val)

for key in ['rmse', 'log-lik']:
    print(key)
    for method, dataset in product(methods, datasets):
        print(method, dataset, losses[method][dataset][key])

"""Python Script Template."""
from gpssm.plotters import plot_evaluation_datasets
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

datasets = {'Actuator': 512, 'BallBeam': 500, 'Drive': 250, 'Dryer': 500,
            'GasFurnace': 148, 'Flutter': 512, 'Tank': 1250}
methods = ['PRSSM', 'VCDT', 'CBFSSM']
seeds = [0, 1, 2, 3, 4]

# fig, axes = plt.subplots(len(datasets), 1, sharex=True)
fig = plt.figure()
losses = {'rmse': {}, 'nrmse': {}, 'log-lik': {}}
for dataset, length in datasets.items():
    for key in losses:
        losses[key][dataset] = {}

    for method in methods:
        for key in losses:
            if method not in losses[key][dataset]:
                losses[key][dataset][method] = [0, 0]
        aux = []
        for i, seed in enumerate(seeds):
            file_name = '{}/{}/test_{}_results_{}.txt'.format(dataset, method,
                                                              length, seed)
            with open(file_name) as file:
                data = file.readline().split('. ')
                for key_val in data:
                    key, val = key_val.split(':')
                    old_mean = losses[key.lower()][dataset][method][0]
                    old_var = losses[key.lower()][dataset][method][1] ** 2

                    delta = float(val) - old_mean
                    new_mean = old_mean + delta / (i + 1)
                    new_var = delta ** 2 / (i + 1)
                    if i > 0:
                        new_var += old_var * (i-1) / i

                    losses[key.lower()][dataset][method][0] = new_mean
                    losses[key.lower()][dataset][method][1] = np.sqrt(new_var)
                    if key.lower() == 'rmse':
                        aux.append(float(val))

for key in ['rmse', 'log-lik']:
    print(key)
    for dataset in datasets:
        print(dataset, losses[key][dataset])
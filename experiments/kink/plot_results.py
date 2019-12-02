"""Python Script Template."""
from gpssm.plotters import plot_evaluation_datasets
import matplotlib.pyplot as plt
from itertools import product

datasets = ['Kink']
methods = ['PRSSM', 'VCDT', 'CBFSSM']
samplings = ['delta', 'full', 'mean', 'sample']

# fig, axes = plt.subplots(len(datasets), 1, sharex=True)
fig = plt.figure()
losses = {'rmse': {}, 'nrmse': {}, 'log-lik': {}}
for dataset in datasets:
    for key in losses:
        losses[key][dataset] = {}

    for method, sampling in product(methods, samplings):
        for key in losses:
            losses[key][dataset][method + sampling] = {}

        file_name = '{}/{}/{}/test_results.txt'.format(dataset, method, sampling)
        try:
            with open(file_name) as file:
                data = file.readline().split('. ')
                for key_val in data:
                    key, val = key_val.split(':')
                    losses[key.lower()][dataset][method] = [float(val), 0]
        except FileNotFoundError:
            print('File {} not found'.format(file_name))


for key, val in losses.items():
    fig = plot_evaluation_datasets(val)
    fig.gca().set_title(key)
    fig.show()
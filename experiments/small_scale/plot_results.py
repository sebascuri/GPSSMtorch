"""Python Script Template."""
from gpssm.plotters import plot_evaluation_datasets
import matplotlib.pyplot as plt
from itertools import product

datasets = ['Actuator', 'BallBeam', 'Drive', 'Dryer', 'GasFurnace', 'Flutter'] #, 'Tank']
methods = ['PRSSM', 'VCDT', 'CBFSSM']

# fig, axes = plt.subplots(len(datasets), 1, sharex=True)
fig = plt.figure()
losses = {'rmse': {}, 'nrmse': {}, 'log-lik': {}}
for dataset in datasets:
    for key in losses:
        losses[key][dataset] = {}

    for method, k in product(methods, [0.01, 0.1, 1.0]):
        mk = '{}{}'.format(method, k)
        for key in losses:
            losses[key][dataset][mk] = {}

        file_name = '{}/{}/{}/test_50_results.txt'.format(dataset, method, k)
        try:
            with open(file_name) as file:
                data = file.readline().split('. ')
                for key_val in data:
                    key, val = key_val.split(':')
                    losses[key.lower()][dataset][mk] = [float(val), 0]
        except FileNotFoundError:
            print('File {} not found'.format(file_name))
            for key in losses:
                losses[key][dataset].pop(mk)

print(losses)
for key, val in losses.items():
    fig = plot_evaluation_datasets(val)
    fig.gca().set_title(key)
    fig.show()
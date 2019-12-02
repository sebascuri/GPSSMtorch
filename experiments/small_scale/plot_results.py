"""Python Script Template."""
from gpssm.plotters import plot_evaluation_datasets
import matplotlib.pyplot as plt

datasets = ['Actuator', 'BallBeam', 'Drive', 'Dryer', 'GasFurnace', 'Flutter'] #, 'Tank']
methods = ['PRSSM', 'VCDT', 'CBFSSM']

# fig, axes = plt.subplots(len(datasets), 1, sharex=True)
fig = plt.figure()
losses = {'rmse': {}, 'nrmse': {}, 'log-lik': {}}
for dataset in datasets:
    for key in losses:
        losses[key][dataset] = {}

    for method in methods:
        for key in losses:
            losses[key][dataset][method] = {}

        file_name = '{}/{}/test_results.txt'.format(dataset, method)
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
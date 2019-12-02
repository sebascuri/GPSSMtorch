"""Python Script Template."""
from itertools import product
import matplotlib.pyplot as plt

datasets = ['Actuator', 'BallBeam', 'Drive', 'Dryer', 'GasFurnace']
methods = ['CBFSSM']
recognition = ['bi-lstm', 'conv', 'lstm', 'nn', 'output', 'zero']

# fig, axes = plt.subplots(len(datasets), 1, sharex=True)
for dataset, method in product(datasets, methods):
    fig = plt.figure()
    vals = []
    for rec in recognition:
        # keys.append(method + rec)
        file_name = '{}/{}/{}/test_results.txt'.format(dataset, method, rec)
        try:
            with open(file_name) as file:
                data = file.readline()
                vals.append(float(data.split('NRMSE: ')[1][:4]))

        except FileNotFoundError:
            print('File {} not found'.format(file_name))
            vals.append(0)
    print(vals)
    plt.bar(recognition, vals)
        # axes[i].set_ylabel('NRMSE')

    # axes[i].set_title(dataset)
    plt.title(dataset)
    plt.ylabel('Normalized RMSE')
    plt.xlabel('Methods')
    plt.show()

# USE Output or LSTM


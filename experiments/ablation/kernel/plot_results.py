"""Python Script Template."""
from itertools import product
import matplotlib.pyplot as plt

datasets = ['Actuator', 'BallBeam', 'Drive', 'Dryer', 'GasFurnace']
method = 'PRSSM'
kernels = ['rbf', 'matern12', 'matern32', 'matern52']
shared = [True, False]

# fig, axes = plt.subplots(len(datasets), 1, sharex=True)
for i, dataset in enumerate(datasets):
    fig = plt.figure()
    vals = []
    keys = []
    for kernel, sh in product(kernels, shared):
        keys.append(kernel + str(sh))
        file_name = '{}/{}/{}/{}/test_results.txt'.format(dataset, method, kernel, sh)
        try:
            with open(file_name) as file:
                data = file.readline()
                vals.append(float(data.split('NRMSE: ')[1][:5]))

        except FileNotFoundError:
            print('File {} not found'.format(file_name))
            vals.append(0)
    print(vals)
    plt.bar(keys, vals)
        # axes[i].set_ylabel('NRMSE')

    # axes[i].set_title(dataset)
    plt.title(dataset)
    plt.ylabel('Normalized RMSE')
    plt.xlabel('Methods')
    plt.show()

# RBF performs best


"""Python Script Template."""
from itertools import product
import matplotlib.pyplot as plt

datasets = ['Actuator', 'BallBeam', 'Drive', 'Dryer', 'GasFurnace']
methods = ['PRSSM', 'VCDT', 'CBFSSM']

entropies = [0.0, 0.1, 1.0, 10.0]
nrmse = {}

# fig, axes = plt.subplots(len(datasets), 1, sharex=True)
for i, dataset in enumerate(datasets):
    fig = plt.figure()
    for method in methods:
        vals = []
        for entropy in entropies:
            file_name = '{}/{}/{}/test_results.txt'.format(dataset, method, entropy)
            try:
                with open(file_name) as file:
                    data = file.readline()
                    vals.append(float(data.split('NRMSE: ')[1][:4]))

            except FileNotFoundError:
                print('File {} not found'.format(file_name))
                vals.append(0)

        plt.plot(entropies, vals, label='{}'.format(method))
        # axes[i].set_ylabel('NRMSE')

    # axes[i].set_title(dataset)
    plt.title(dataset)
    plt.ylabel('Normalized RMSE')
    plt.xlabel('Entropy')
    plt.legend(loc='best')
    plt.show()

# NO ENTROPY (not clear)

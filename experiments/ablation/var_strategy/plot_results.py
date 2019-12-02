"""Python Script Template."""
from itertools import product
import matplotlib.pyplot as plt

datasets = ['Actuator', 'BallBeam', 'Drive', 'Dryer', 'GasFurnace']
methods = ['PRSSM', 'VCDT', 'CBFSSM']
var_methods = ['delta', 'full', 'mean', 'sample']

# fig, axes = plt.subplots(len(datasets), 1, sharex=True)
for i, dataset in enumerate(datasets):
    fig = plt.figure()
    vals = []
    keys = []
    for method, var in product(methods, var_methods):
        keys.append(method + var)
        file_name = '{}/{}/{}/test_results.txt'.format(dataset, method, var)
        try:
            with open(file_name) as file:
                data = file.readline()
                try:
                    vals.append(float(data.split('NRMSE: ')[1][:4]))
                except ValueError:
                    vals.append(float(data.split('NRMSE: ')[1][:3]))

        except FileNotFoundError:
            print('File {} not found'.format(file_name))
            vals.append(0)
    print(vals)
    plt.barh(keys, vals)
        # axes[i].set_ylabel('NRMSE')

    # axes[i].set_title(dataset)
    plt.title(dataset)
    plt.ylabel('Normalized RMSE')
    plt.xlabel('Methods')
    plt.show()

# Use either MEAN or FULL strategies.

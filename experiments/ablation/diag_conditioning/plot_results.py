"""Python Script Template."""
from itertools import product
import matplotlib.pyplot as plt

datasets = ['Actuator', 'BallBeam', 'Drive', 'Dryer', 'GasFurnace']
methods = ['PRSSM', 'PRSSMDiag', 'VCDT', 'VCDTDiag', 'CBFSSM', 'CBFSSMDiag']

# fig, axes = plt.subplots(len(datasets), 1, sharex=True)
for i, dataset in enumerate(datasets):
    fig = plt.figure()
    vals = []
    for method in methods:
        file_name = '{}/{}/test_results.txt'.format(dataset, method)
        try:
            with open(file_name) as file:
                data = file.readline()
                vals.append(float(data.split('NRMSE: ')[1][:5]))

        except FileNotFoundError:
            print('File {} not found'.format(file_name))
            vals.append(0)
    print(vals)
    plt.bar(methods, vals)
        # axes[i].set_ylabel('NRMSE')

    # axes[i].set_title(dataset)
    plt.title(dataset)
    plt.ylabel('Normalized RMSE')
    plt.xlabel('Methods')
    plt.show()

# Do not use Diag Conditioning


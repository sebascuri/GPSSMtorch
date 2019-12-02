"""Python Script Template."""
from itertools import product
import matplotlib.pyplot as plt

datasets = ['Actuator', 'BallBeam', 'Drive', 'Dryer', 'GasFurnace']
methods = ['PRSSM', 'VCDT', 'CBFSSM']

ips = [4, 8, 20, 50]
learnable = [True, False]
nrmse = {}

# fig, axes = plt.subplots(len(datasets), 1, sharex=True)
for i, dataset in enumerate(datasets):
    fig = plt.figure()
    for method, l in product(methods, learnable):
        vals = []
        for ip in ips:
            file_name = '{}/{}/{}/{}/test_results.txt'.format(dataset, method, ip,l)
            try:
                with open(file_name) as file:
                    data = file.readline()
                    vals.append(float(data.split('NRMSE: ')[1][:5]))

            except FileNotFoundError:
                print('File {} not found'.format(file_name))
                vals.append(0)

        plt.plot(ips, vals, label='{} {}'.format(method, 'adaptive' if l else 'fixed'))
        # axes[i].set_ylabel('NRMSE')

    # axes[i].set_title(dataset)
    plt.title(dataset)
    plt.ylabel('Normalized RMSE')
    plt.xlabel('Inducing Points')
    plt.legend(loc='best')
    plt.show()

# Conclusion: Learnable IP, at least 20, the more the merrier.
# File Dryer/CBFSSM/50/False/test_results.txt not found

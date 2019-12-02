"""Python Script Template."""
from itertools import product
import matplotlib.pyplot as plt

datasets = ['Actuator', 'BallBeam', 'Drive', 'Dryer', 'GasFurnace']
methods = ['PRSSM', 'VCDT', 'CBFSSM']

seq_length = [10, 20, 50, 100]
# fig, axes = plt.subplots(len(datasets), 1, sharex=True)
for i, dataset in enumerate(datasets):
    fig = plt.figure()
    for method in methods:
        vals = []
        for seq_len in seq_length:
            file_name = '{}/{}/{}/test_results.txt'.format(dataset, method, seq_len)
            try:
                with open(file_name) as file:
                    data = file.readline()
                    vals.append(float(data.split('NRMSE: ')[1][:4]))

            except FileNotFoundError:
                print('File {} not found'.format(file_name))
                vals.append(0)

        plt.plot(seq_length, vals, label=method)
        # axes[i].set_ylabel('NRMSE')

    # axes[i].set_title(dataset)
    plt.title(dataset)
    plt.ylabel('Normalized RMSE')
    plt.xlabel('Sequence Length')
    plt.legend(loc='best')
    plt.show()

# Conclusion sequence length = 50


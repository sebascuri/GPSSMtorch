"""Python Script Template."""
from itertools import product
import matplotlib.pyplot as plt

datasets = ['Actuator', 'BallBeam', 'Drive', 'Dryer', 'GasFurnace']
methods = ['VCDT', 'CBFSSM']
k_factor = ['1.0', '2.0', '5.0', '10.0', '50.0']
xlabel = [1, 2, 5, 10, 50]
for i, dataset in enumerate(datasets):
    fig = plt.figure()
    for method in methods:
        vals = []
        for k_fact in k_factor:
            file_name = '{}/{}/{}/test_results.txt'.format(dataset, method, k_fact)
            try:
                with open(file_name) as file:
                    data = file.readline()
                    vals.append(float(data.split('NRMSE: ')[1][:5]))

            except FileNotFoundError:
                print('File {} not found'.format(file_name))
                vals.append(0)

        plt.plot(xlabel, vals, label=method)
        # axes[i].set_ylabel('NRMSE')

    # axes[i].set_title(dataset)
    plt.title(dataset)
    plt.ylabel('Normalized RMSE')
    plt.xlabel('k-Factor')
    plt.legend(loc='best')
    plt.show()

# k-Factor Might help a lot.
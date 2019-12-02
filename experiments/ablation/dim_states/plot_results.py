"""Python Script Template."""
from itertools import product
import matplotlib.pyplot as plt

datasets = ['Actuator', 'BallBeam', 'Drive', 'Dryer', 'GasFurnace']
methods = ['PRSSM', 'VCDT', 'CBFSSM']
dim_states = [1, 4, 10]

# fig, axes = plt.subplots(len(datasets), 1, sharex=True)
for i, dataset in enumerate(datasets):
    fig = plt.figure()
    for method in methods:
        vals = []
        for dim_state in dim_states:
            file_name = '{}/{}/{}/test_results.txt'.format(dataset, method, dim_state)
            try:
                with open(file_name) as file:
                    data = file.readline()
                    vals.append(float(data.split('NRMSE: ')[1][:4]))

            except FileNotFoundError:
                print('File {} not found'.format(file_name))
                vals.append(0)

        plt.plot(dim_states, vals, label=method)
        # axes[i].set_ylabel('NRMSE')

    # axes[i].set_title(dataset)
    plt.title(dataset)
    plt.ylabel('Normalized RMSE')
    plt.xlabel('State Dimensionality')
    plt.legend(loc='best')
    plt.show()

# Conclusion 20 IP/learnable = True
# File Actuator/PRSSM/10/test_results.txt not found
# File Actuator/VCDT/10/test_results.txt not found
# File Actuator/CBFSSM/1/test_results.txt not found
# File Actuator/CBFSSM/4/test_results.txt not found
# File Actuator/CBFSSM/10/test_results.txt not found
# File BallBeam/PRSSM/10/test_results.txt not found
# File BallBeam/VCDT/10/test_results.txt not found
# File BallBeam/CBFSSM/1/test_results.txt not found
# File BallBeam/CBFSSM/4/test_results.txt not found
# File BallBeam/CBFSSM/10/test_results.txt not found
# File Drive/PRSSM/10/test_results.txt not found
# File Drive/VCDT/10/test_results.txt not found
# File Drive/CBFSSM/1/test_results.txt not found
# File Drive/CBFSSM/4/test_results.txt not found
# File Drive/CBFSSM/10/test_results.txt not found
# File Dryer/PRSSM/10/test_results.txt not found
# File Dryer/VCDT/10/test_results.txt not found
# File Dryer/CBFSSM/1/test_results.txt not found
# File Dryer/CBFSSM/4/test_results.txt not found
# File Dryer/CBFSSM/10/test_results.txt not found
# File GasFurnace/PRSSM/10/test_results.txt not found
# File GasFurnace/VCDT/10/test_results.txt not found
# File GasFurnace/CBFSSM/1/test_results.txt not found
# File GasFurnace/CBFSSM/4/test_results.txt not found
# File GasFurnace/CBFSSM/10/test_results.txt not found

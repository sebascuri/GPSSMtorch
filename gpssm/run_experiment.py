"""Run an Experiment from a config file."""
from gpssm.runner import init_runner
from gpssm.utilities import make_dir
import argparse
from itertools import product
import sys
import yaml
from copy import deepcopy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GP-SSM from config file.')
    parser.add_argument('config_file', type=str, help='config file.')
    parser.add_argument('--num-threads', type=int, default=2, help='threads to use.')
    parser.add_argument('-W', '--wall-time', type=int, default=None,
                        help='wall-time in minutes.')
    parser.add_argument('--memory', type=int, default=None, help='memory in MB.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0],
                        help='random seeds to use.')

    args = parser.parse_args()

    interpreter_script = sys.executable
    base_cmd = interpreter_script + ' ' + 'gpssm/run.py'

    with open(args.config_file, 'r') as file:
        configs = yaml.load(file, Loader=yaml.SafeLoader)

    experiment_name = configs['experiment']['name']
    base_dir = 'experiments/' + experiment_name
    make_dir(base_dir)

    experiment_keys = configs['experiment']['splits']
    experiment_values = []
    for keys in experiment_keys:
        values = configs.copy()
        for key in keys:
            values = values[key]
        experiment_values.append(values)

    cmd_list = []
    for values in product(*experiment_values):
        folder = (base_dir + '/' + '{}/' * len(values)).format(*values)
        make_dir(folder)
        configs_ = deepcopy(configs)
        configs_['experiment']['name'] = folder
        configs_['experiment'].pop('splits')

        for idx, keys in enumerate(experiment_keys):
            aux = configs_
            for key in keys[:-1]:
                aux = aux[key]
            aux[keys[-1]] = values[idx]

        config_file = folder + 'config.yaml'
        with open(config_file, 'w') as file:
            yaml.dump(configs_, file, default_flow_style=False)

        for seed in args.seeds:
            cmd = base_cmd + ' --config-file {} --seed {} --num-threads {}'.format(
                config_file, seed, args.num_threads)
            cmd_list.append(cmd)

    runner = init_runner(num_threads=args.num_threads, wall_time=args.wall_time,
                         memory=args.memory)
    runner.run(cmd_list)

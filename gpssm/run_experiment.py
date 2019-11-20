"""Run an Experiment from a config file."""
from gpssm.runner import init_runner
import argparse
from itertools import product
import sys
import yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GP-SSM from config file.')
    parser.add_argument('config_file', type=str, default='experiments/small_scale.yaml',
                        help='config file with hyper-parameters.')
    parser.add_argument('--num-threads', type=int, default=2,
                        help='number of threads to use.')
    parser.add_argument('-W', '--wall-time', type=int, default=None,
                        help='wall-time in minutes.')
    parser.add_argument('--memory', type=int, default=None,
                        help='memory in MB.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0],
                        help='random seeds to use.')

    args = parser.parse_args()

    interpreter_script = sys.executable
    base_cmd = interpreter_script + ' ' + 'gpssm/run.py'

    with open(args.config_file, 'r') as file:
        configs = yaml.load(file, Loader=yaml.SafeLoader)

    models = configs.get('model').pop('name')
    datasets = configs.get('dataset').pop('name')

    cmd_list = []
    for dataset, model, seed in product(datasets, models, args.seeds):
        cmd = base_cmd + ' --dataset {} --model {} --config-file {} --seed {}'.format(
            dataset, model, args.config_file, seed)
        cmd += ' --num-threads {}'.format(args.num_threads)
        cmd_list.append(cmd)

    runner = init_runner(num_threads=args.num_threads, wall_time=args.wall_time,
                         memory=args.memory)
    runner.run(cmd_list)

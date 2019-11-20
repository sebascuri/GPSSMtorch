"""Post Process the Experiments from a config file."""
import argparse
from itertools import product
import yaml
from gpssm.utilities import Experiment, load
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GP-SSM from config file.')
    parser.add_argument('config_file', type=str, default='experiments/small_scale.yaml',
                        help='config file with hyper-parameters.')
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        configs = yaml.load(file, Loader=yaml.SafeLoader)

    models = configs.get('model').pop('name')
    datasets = configs.get('dataset').pop('name')

    results = {}  # type: ignore
    lengths = ['all', 'train']
    keys = ['loglik', 'rmse']

    for dataset, model in product(datasets, models):
        if dataset not in results:
            results[dataset] = {}
        if model not in results[dataset]:
            results[dataset][model] = {length: {key: [] for key in keys}
                                       for length in lengths}

        for length in lengths:
            experiment = Experiment(model, dataset, -1, {})
            evaluators = load(experiment, 'eval_{}'.format(length))
            for evaluator, key in product(evaluators, keys):
                results[dataset][model][length][key].append(np.mean(evaluator[key]))

        print(dataset, model, results[dataset][model])

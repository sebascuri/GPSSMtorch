"""Post Process the Experiments from a config file."""
import argparse
from itertools import product
import yaml
from .dataset import get_dataset
from .utilities import Experiment, load
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GP-SSM from config file.')
    parser.add_argument('config_file', type=str,
                        default='experiments/small_scale/config.yaml',
                        help='config file with hyper-parameters.')
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        configs = yaml.load(file, Loader=yaml.SafeLoader)
        configs['name'] = args.config_file.split('/')[1]

    models = configs.get('model').pop('name')
    datasets = configs.get('dataset').pop('name')

    results = {}  # type: dict
    splits = ['train']
    keys = ['loglik', 'rmse']

    for dataset, model in product(datasets, models):
        if dataset not in results:
            results[dataset] = {}
        if model not in results[dataset]:
            results[dataset][model] = {split: {key: [] for key in keys}
                                       for split in splits}

        for split in splits:
            ds = get_dataset(dataset)(train=split == 'train', **configs['dataset'])
            scale = ds.output_normalizer.sd[0]
            experiment = Experiment(model, dataset, -1, configs)
            evaluators = load(experiment, 'eval_{}'.format(split))
            for evaluator, key in product(evaluators, keys):
                if key == 'rmse':
                    val = np.mean(evaluator[key]) * scale  # Rescale
                else:
                    val = np.mean(evaluator[key])

                results[dataset][model][split][key].append(val)

        print(dataset, model, results[dataset][model])

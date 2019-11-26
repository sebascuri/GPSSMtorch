"""Integration test."""
from gpssm.run import main, Experiment
import yaml


def test_integration():
    """Test project running integration."""
    config_file = 'experiments/test/config.yaml'
    with open(config_file, 'r') as file:
        configs = yaml.load(file, Loader=yaml.SafeLoader)
    configs.get('model', {}).pop('name', {})
    configs.get('dataset', {}).pop('name', {})
    configs['name'] = config_file.split('/')[1]

    main(Experiment('CBFSSM', 'Actuator', 0, configs), 2)

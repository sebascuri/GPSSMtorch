"""Integration test."""
import pytest
from gpssm.run import main, Experiment
import yaml


@pytest.fixture(params=['PRSSM', 'CBFSSM', 'CBFSSM-HALF'])
def method(request):
    return request.param


def test_integration(method):
    """Test project running integration."""
    config_file = 'experiments/test/config.yaml'
    with open(config_file, 'r') as file:
        configs = yaml.load(file, Loader=yaml.SafeLoader)
    configs.get('model', {}).pop('name', {})
    configs.get('dataset', {}).pop('name', {})
    configs['name'] = config_file.split('/')[1]

    main(Experiment(method, 'Actuator', 0, configs))

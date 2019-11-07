"""Testing Kink Function dataset."""
import pytest
from gpssm.dataset.dataset import KinkFunction
import numpy as np
import torch

datasets = [
    (KinkFunction, 1, 1, 60, 60, 0, 1, 1)
]

@pytest.fixture(params=[True, False])
def train(request):
    return request.param


@pytest.fixture(params=datasets)
def dataset(request):
    return request.param


@pytest.fixture(params=[None, 1, 24])
def sequence_length(request):
    return request.param


@pytest.fixture(params=[1, 2, 15])
def sequence_stride(request):
    return request.param


def test_dataset_shapes(train, dataset, sequence_length, sequence_stride):
    dataset_, n_train, n_test, len_train, len_test, dim_u, dim_y, dim_x = dataset
    dataset = dataset_(train=train, sequence_length=sequence_length,
                       sequence_stride=sequence_stride)

    print(dataset)

    assert dataset.inputs.dtype == np.float64
    assert dataset.outputs.dtype == np.float64
    assert dataset.states.dtype == np.float64

    if train:
        n_exp = n_train
        data_size = len_train
    else:
        n_exp = n_test
        data_size = len_test

    if sequence_length is None:
        sequence_length = data_size

    num_seq = (data_size - sequence_length) // sequence_stride + 1
    if (data_size - sequence_length) % sequence_stride > 0:
        num_seq += 1

    assert dataset.inputs.shape == (n_exp * num_seq, sequence_length, dim_u)
    assert dataset.outputs.shape == (n_exp * num_seq, sequence_length, dim_y)
    assert dataset.states.shape == (n_exp * num_seq, sequence_length, dim_x)

    assert len(dataset) == n_exp * num_seq

    inputs, outputs, states = dataset[np.random.choice(len(dataset))]
    assert inputs.shape == (sequence_length, dim_u)
    assert outputs.shape == (sequence_length, dim_y)
    assert states.shape == (sequence_length, dim_x)

    for tensor in [inputs, outputs, states]:
        assert type(tensor) == torch.Tensor
        assert tensor.dtype == torch.float

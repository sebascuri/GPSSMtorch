"""Testing recognition module."""
import pytest
import torch
from gpssm.models.components.recognition_model import OutputRecognition, \
    ZeroRecognition, NNRecognition, ConvRecognition, LSTMRecognition


@pytest.fixture(params=[OutputRecognition, ZeroRecognition, NNRecognition,
                        ConvRecognition, LSTMRecognition])
def model(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def dim_y(request):
    return request.param


@pytest.fixture(params=[0, 1, 4])
def dim_u(request):
    return request.param


@pytest.fixture(params=[4])
def dim_x(request):
    return request.param


def test_recognition(model, dim_u, dim_y, dim_x):
    if model == OutputRecognition or model == ZeroRecognition:
        length = 1
    else:
        length = 10

    recognition = model(dim_y, dim_u, dim_x, length, variance=1.0)

    in_seq = torch.randn(32, 10, dim_u)
    out_seq = torch.randn(32, 10, dim_y)
    out = recognition(out_seq, in_seq)

    assert out.loc.shape == torch.Size([32, dim_x])
    assert out.covariance_matrix.shape == torch.Size([32, dim_x, dim_x])

    other = recognition.copy()
    assert other is not recognition
    assert type(other) is type(recognition)
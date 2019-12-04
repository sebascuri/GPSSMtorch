"""Testing template."""
from gpssm.models.components.utilities import safe_softplus, inverse_softplus
import torch
from torch.testing import assert_allclose


def test_safety():
    x = torch.rand(32, 4, 10)
    assert torch.all(safe_softplus(x) > torch.nn.functional.softplus(x))
    assert_allclose(safe_softplus(x, 0), torch.nn.functional.softplus(x))


def test_shape():
    x = torch.rand(32, 4, 10)
    assert safe_softplus(x).shape == torch.Size([32, 4, 10])
    assert inverse_softplus(x).shape == torch.Size([32, 4, 10])


def test_circle():
    x = torch.rand(32, 4, 10)
    assert_allclose(x, safe_softplus(inverse_softplus(x), 0))
    assert_allclose(x, inverse_softplus(safe_softplus(x, 1e-12)))

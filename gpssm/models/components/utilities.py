"""Python Script Template."""
import torch


def safe_softplus(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Safe softplus to return a softplus larger than epsilon.

    Parameters
    ----------
    x: torch.Tensor.
        Input tensor to transform.
    eps: float.
        Safety jitter.

    Returns
    -------
    output: torch.Tensor.
        Transformed tensor.
    """
    return torch.nn.functional.softplus(x) + eps


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Inverse function to torch.functional.softplus.

    Parameters
    ----------
    x: torch.Tensor.
        Input tensor to transform.

    Returns
    -------
    output: torch.Tensor.
        Transformed tensor.
    """
    return torch.log(torch.exp(x) - 1.)

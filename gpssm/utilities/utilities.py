"""Utilities for training and evaluating."""
import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.distributions import Normal
import gpytorch
from gpytorch.distributions import MultivariateNormal
from tqdm import tqdm
from typing import List
from gpssm.models.gpssm_vi import GPSSM
from gpssm.plotters.plot_sequences import plot_predictions
from .evaluator import Evaluator

__author__ = 'Sebastian Curi'
__all__ = ['approximate_with_normal', 'train', 'evaluate']


def approximate_with_normal(predicted_outputs: List[MultivariateNormal]) -> Normal:
    """Approximate a particle distribution with a Normal by moment matching."""
    sequence_length = len(predicted_outputs)
    dim_outputs, batch_size, _ = predicted_outputs[0].loc.shape

    output_loc = torch.zeros((batch_size, sequence_length, dim_outputs))
    output_cov = torch.zeros((batch_size, sequence_length, dim_outputs))
    for t, y_pred in enumerate(predicted_outputs):
        # Collapse particles!
        output_loc[:, t] = y_pred.loc.mean(dim=-1).t()
        output_cov[:, t] = y_pred.scale.mean(dim=-1).t()

    return Normal(output_loc, output_cov)


def train(model: GPSSM, dataloader: DataLoader, optimizer: Optimizer, num_epochs: int
          ) -> List[float]:
    """Train a model.

    Parameters
    ----------
    model: GPSSM.
        Model to train.
    dataloader: DataLoader.
        Loader to iterate data.
    optimizer: Optimizer.
        Model Optimizer.
    num_epochs: int.
        Number of epochs.

    Returns
    -------
    losses: list of int.
        List of losses encountered during training.
    """
    losses = []
    for _ in tqdm(range(num_epochs)):
        for idx, (inputs, outputs, states) in enumerate(dataloader):
            # Zero the gradients of the Optimizer
            optimizer.zero_grad()

            # Compute the elbo
            predicted_outputs = model(outputs, inputs)
            loss = model.loss(predicted_outputs, outputs, inputs)

            # Back-propagate
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

    return losses


def evaluate(model: GPSSM, dataloader: DataLoader) -> None:
    """Evaluate a model.

    Parameters
    ----------
    model: GPSSM.
        Model to train.
    dataloader: DataLoader.
        Loader to iterate data.

    """
    evaluator = Evaluator()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        model.eval()
        for inputs, outputs, states in dataloader:
            predicted_outputs = model(outputs, inputs)
            predicted_outputs = approximate_with_normal(predicted_outputs)

            mean = predicted_outputs.loc.detach().numpy()
            scale = predicted_outputs.scale.detach().numpy()

            print(evaluator.evaluate(predicted_outputs, outputs))

            fig = plot_predictions(mean[0].T, np.sqrt(scale[0]).T,
                                   outputs[0].detach().numpy().T,
                                   inputs[0].detach().numpy().T)
            fig.axes[0].set_title(dataloader.dataset + ' Predictions')
            fig.show()

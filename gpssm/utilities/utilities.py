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
from gpssm.plotters.plot_sequences import plot_pred, plot_2d, plot_transition
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

        print(model)
    return losses


def evaluate(model: GPSSM, dataloader: DataLoader, plot_list: list = None) -> Evaluator:
    """Evaluate a model.

    Parameters
    ----------
    model: GPSSM.
        Model to train.
    dataloader: DataLoader.
        Loader to iterate data.
    plot_list: list of str.
        list of plotters.

    """
    plot_list = [] if plot_list is None else plot_list

    model_name = model.__class__.__name__
    data_name = dataloader.dataset.__class__.__name__

    evaluator = Evaluator()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # model.eval()
        for inputs, outputs, states in dataloader:
            predicted_outputs = model(outputs, inputs)
            predicted_outputs = approximate_with_normal(predicted_outputs)

            mean = predicted_outputs.loc.detach().numpy()
            scale = predicted_outputs.scale.detach().numpy()

            evaluator.evaluate(predicted_outputs, outputs)

            if 'prediction' in plot_list:
                fig = plot_pred(mean[0].T, np.sqrt(scale[0]).T, outputs[0].numpy().T)
                fig.gca().set_title('{} {} Prediction'.format(model_name, data_name))
                fig.show()
            if '2d' in plot_list:
                fig = plot_2d(mean[0].T, np.sqrt(scale[0]).T, outputs[0].numpy().T)
                fig.gca().set_title('{} {} Prediction'.format(model_name, data_name))
                fig.show()
            if 'transition' in plot_list:  # only implemented for 1d.
                gp = model.forward_model.models[0]
                transition = model.transitions.likelihoods[0]
                x = torch.arange(-3, 1, 0.1)
                true_next_x = dataloader.dataset.f(x.numpy())  # type: ignore
                pred_next_x = transition(gp(x))

                fig = plot_transition(
                    x.numpy(), true_next_x, pred_next_x.loc.numpy(),
                    torch.diag(pred_next_x.covariance_matrix).sqrt().numpy())
                fig.show()

        print(np.array(evaluator['loglik']).mean(),
              np.array(evaluator['rmse']).mean())

    return evaluator

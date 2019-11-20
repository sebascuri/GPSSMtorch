"""Utilities for training and evaluating."""
import numpy as np
import torch
import os
import pickle
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.distributions import Normal
import gpytorch
from gpytorch.distributions import MultivariateNormal
from tqdm import tqdm
from typing import List, Tuple
from gpssm.dataset import get_dataset, Dataset
from gpssm.models import get_model
from gpssm.models.gpssm_vi import GPSSM
from gpssm.plotters.plot_sequences import plot_pred, plot_2d, plot_transition
from .evaluator import Evaluator
from collections import namedtuple


__author__ = 'Sebastian Curi'
__all__ = ['Experiment', 'approximate_with_normal', 'train', 'evaluate', 'save', 'load']


class Experiment(namedtuple('Experiment', ['model', 'dataset', 'seed', 'configs',
                                           'log_dir', 'fig_dir'])):
    """Experiment Named Tuple."""

    def __new__(cls, model: str, dataset: str, seed: int, configs: dict = None):
        """Create new named experiment."""
        log_dir = get_dir(model, dataset, fig_dir=False)
        fig_dir = get_dir(model, dataset, fig_dir=True)
        return super(Experiment, cls).__new__(
            cls, model, dataset, seed, configs, log_dir, fig_dir)


def get_dir(model: str, dataset: str, fig_dir: bool = False) -> str:
    """Get the log or figure directory.

    If the directory does not exist, create it.

    Parameters
    ----------
    model: str.
        Name of model.
    dataset: str.
        Name of dataset.
    fig_dir: bool, optional.
        Flag that indicates if the directory is

    Returns
    -------
    dir: string

    """
    if 'SCRATCH' not in os.environ or fig_dir:
        base_dir = os.getcwd()
    else:
        base_dir = os.environ['SCRATCH']

    log_directory = base_dir + '/experiments/{}/{}/'.format(dataset, model)

    try:
        os.makedirs(log_directory)
    except FileExistsError:
        pass
    return log_directory


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


def evaluate(model: GPSSM, dataloader: DataLoader, experiment: Experiment,
             plot_list: list = None) -> Evaluator:
    """Evaluate a model.

    Parameters
    ----------
    model: GPSSM.
        Model to train.
    dataloader: DataLoader.
        Loader to iterate data.
    experiment: Experiment.
        Experiment meta-data.
    plot_list: list of str.
        list of plotters.

    """
    plot_list = [] if plot_list is None else plot_list
    evaluator = Evaluator()
    dataset = dataloader.dataset  # type: Dataset  # type: ignore
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # model.eval()
        for inputs, outputs, states in dataloader:
            predicted_outputs = model(outputs, inputs)
            predicted_outputs = approximate_with_normal(predicted_outputs)

            mean = predicted_outputs.loc.detach().numpy()
            scale = predicted_outputs.scale.detach().numpy()

            evaluator.evaluate(predicted_outputs, outputs)

            if 'prediction' in plot_list:
                plot_list.remove('prediction')
                fig = plot_pred(mean[0].T, np.sqrt(scale[0]).T, outputs[0].numpy().T)
                fig.gca().set_title('{} {} Prediction'.format(
                    experiment.model, experiment.dataset))
                fig.show()
                fig.savefig('{}prediction_{}_{}.png'.format(
                    experiment.fig_dir, dataset.sequence_length, experiment.seed))

            if '2d' in plot_list:
                plot_list.remove('2d')
                fig = plot_2d(mean[0].T, np.sqrt(scale[0]).T, outputs[0].numpy().T)
                fig.gca().set_title('{} {} Prediction'.format(
                    experiment.model, experiment.dataset))
                fig.show()
                fig.savefig('{}prediction2d_{}_{}.png'.format(
                    experiment.fig_dir, dataset.sequence_length, experiment.seed))

            if 'transition' in plot_list:  # only implemented for 1d.
                plot_list.remove('transition')
                gp = model.forward_model.models[0]
                transition = model.transitions.likelihoods[0]
                x = torch.arange(-3, 1, 0.1)
                true_next_x = dataset.f(x.numpy())
                pred_next_x = transition(gp(x))

                fig = plot_transition(
                    x.numpy(), true_next_x, pred_next_x.loc.numpy(),
                    torch.diag(pred_next_x.covariance_matrix).sqrt().numpy())
                fig.show()
                fig.savefig('{}transition_{}.png'.format(
                    experiment.fig_dir, experiment.seed))

        print('Sequence Length: {}. Log-Lik: {}. RMSE: {}'.format(
            dataset.sequence_length,
            np.array(evaluator['loglik']).mean(),
            np.array(evaluator['rmse']).mean()
        ))
    return evaluator


def save(experiment: Experiment, **kwargs) -> None:
    """Save Model and Experiment.

    Parameters
    ----------
    experiment: Experiment.
        Experiment data to save.

    """
    save_dir = experiment.log_dir
    file_name = save_dir + 'experiment_{}.obj'.format(experiment.seed)
    with open(file_name, 'wb') as file:
        pickle.dump(experiment, file)

    for key, value in kwargs.items():
        if key == 'model':
            file_name = save_dir + 'model_{}.pt'.format(experiment.seed)
            torch.save(value.state_dict(), file_name)
        else:
            file_name = save_dir + '{}_{}.obj'.format(key, experiment.seed)
            with open(file_name, 'wb') as file:
                pickle.dump(value, file)


def load(experiment: Experiment) -> Tuple[Experiment, GPSSM]:
    """Load Experiment data and Model.

    Parameters
    ----------
    experiment: Experiment.
        Experiment meata-data.

    Returns
    -------
    experiment: Experiment.
        Experiment with configs.

    model: GPSSM.
        Initialized model.

    """
    save_dir = experiment.log_dir
    file_name = save_dir + 'experiment_{}.obj'.format(experiment.seed)
    with open(file_name, 'rb') as file:
        experiment = pickle.load(file)

    dataset = get_dataset(experiment.dataset)
    model = get_model(experiment.model, dataset.dim_outputs, dataset.dim_inputs,
                      **experiment.configs['model'])
    file_name = save_dir + 'model_{}.pt'.format(experiment.seed)
    model.load_state_dict(torch.load(file_name))

    return experiment, model

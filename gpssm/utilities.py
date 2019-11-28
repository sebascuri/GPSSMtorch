"""Utilities for training and evaluating."""
import numpy as np
import torch
import os
import pickle
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.distributions import Normal
import gpytorch
from gpytorch.distributions import MultivariateNormal
from tqdm import tqdm
from typing import List
from .dataset import get_dataset, Dataset
from .models import get_model, SSM
from .plotters import plot_pred, plot_2d, plot_transition, plot_loss
from collections import namedtuple

__author__ = 'Sebastian Curi'
__all__ = ['Experiment', 'approximate_with_normal', 'train', 'evaluate', 'save', 'load']


class Evaluator(dict):
    """Object that evaluates the predictive performance of a model."""

    def __init__(self):
        self.criteria = ['loglik', 'rmse']
        super().__init__({criterion: [] for criterion in self.criteria})

    def dump(self, file_name):
        """Dump evaluations to a file."""
        with open(file_name, 'w') as file:
            file.write('Log-Lik: {}. RMSE: {}'.format(
                np.array(self['loglik']).mean(), np.array(self['rmse']).mean()
            ))

    def evaluate(self, predictions: Normal, true_values: Tensor) -> None:
        """Return the RMS error between the true values and the mean predictions.

        Parameters
        ----------
        predictions: MultivariateNormal.
            A multivariate normal with loc [time x dim] and covariance (or scale)
            [time x dim x dim] or [time x dim].
        true_values: Tensor.
            A tensor with shape [time x dim].

        Returns
        -------
        log_likelihood: float.
        """
        for criterion in self.criteria:
            self[criterion].append(getattr(self, criterion)(predictions, true_values))

    @staticmethod
    def loglik(predictions: Normal, true_values: Tensor) -> float:
        """Return the log likelihood of the true values under the predictions.

        Parameters
        ----------
        predictions: MultivariateNormal.
            A multivariate normal with loc [time x dim] and covariance (or scale)
            [time x dim x dim] or [time x dim].
        true_values: Tensor.
            A tensor with shape [time x dim].

        Returns
        -------
        log_likelihood: float.
        """
        return predictions.log_prob(true_values).mean().item()

    @staticmethod
    def rmse(predictions: Normal, true_values: Tensor) -> float:
        """Return the RMS error between the true values and the mean predictions.

        Parameters
        ----------
        predictions: MultivariateNormal.
            A multivariate normal with loc [time x dim] and covariance (or scale)
            [time x dim x dim] or [time x dim].
        true_values: Tensor.
            A tensor with shape [time x dim].

        Returns
        -------
        log_likelihood: float.
        """
        l2 = (predictions.loc - true_values).pow(2).mean(dim=(1, 2))
        return l2.sqrt().mean().item()


_experiment = namedtuple('Experiment',
                         ['model', 'dataset', 'seed', 'configs', 'log_dir', 'fig_dir'])


class Experiment(_experiment):
    """Experiment Named Tuple."""

    def __new__(cls, model: str, dataset: str, seed: int, configs: dict = None,
                log_dir: str = None, fig_dir: str = None):
        """Create new named experiment."""
        configs = {} if configs is None else configs
        if log_dir is None:
            log_dir = get_dir(model, dataset, configs.get('name', ''), fig_dir=False)
        if fig_dir is None:
            fig_dir = get_dir(model, dataset, configs.get('name', ''), fig_dir=True)
        return super(Experiment, cls).__new__(cls, model, dataset, seed, configs,
                                              log_dir, fig_dir)


def get_dir(model: str, dataset: str, exp_name: str, fig_dir: bool = False) -> str:
    """Get the log or figure directory.

    If the directory does not exist, create it.

    Parameters
    ----------
    model: str.
        Name of model.
    dataset: str.
        Name of dataset.
    exp_name:
        Name of experiment.
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

    log_directory = base_dir + '/experiments/{}/{}/{}/'.format(exp_name, dataset, model)

    try:
        os.makedirs(log_directory)
    except FileExistsError:
        pass
    return log_directory


def approximate_with_normal(predicted_outputs: List[MultivariateNormal]) -> Normal:
    """Approximate a particle distribution with a Normal by moment matching."""
    sequence_length = len(predicted_outputs)
    batch_size, dim_outputs, _ = predicted_outputs[0].loc.shape

    output_loc = torch.zeros((batch_size, sequence_length, dim_outputs))
    output_cov = torch.zeros((batch_size, sequence_length, dim_outputs))
    for t, y_pred in enumerate(predicted_outputs):
        # Collapse particles!
        output_loc[:, t, :] = y_pred.loc.mean(dim=-1)
        output_cov[:, t, :] = y_pred.scale.mean(dim=-1) + y_pred.loc.var(dim=-1)
    return Normal(output_loc, output_cov)


def train(model: SSM, dataloader: DataLoader, optimizer: Optimizer, num_epochs: int,
          experiment: Experiment) -> List[float]:
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
    experiment: Experiment.
        Experiment meta-data.

    Returns
    -------
    losses: list of int.
        List of losses encountered during training.
    """
    losses = []
    for _ in tqdm(range(num_epochs)):
        for inputs, outputs, states in tqdm(dataloader):
            # Zero the gradients of the Optimizer
            optimizer.zero_grad()

            # Compute the loss.
            predicted_outputs, loss = model.forward(outputs, inputs)

            # Back-propagate
            loss.backward(retain_graph=True)
            optimizer.step()

            losses.append(loss.item())

        print(model)

    fig = plot_loss(losses, ylabel=model.loss_key.upper())
    fig.gca().set_title('{} {} Training Loss'.format(
        experiment.model, experiment.dataset))
    fig.show()
    fig.savefig('{}training_loss.png'.format(experiment.fig_dir))

    return losses


def evaluate(model: SSM, dataloader: DataLoader, experiment: Experiment,
             plot_list: list = None, key: str = '') -> Evaluator:
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
        List of plotters.
    key: str.
        Key to end files with.

    """
    plot_list = [] if plot_list is None else plot_list
    dataset = dataloader.dataset  # type: Dataset  # type: ignore
    evaluator = Evaluator()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        model.eval()
        for inputs, outputs, states in dataloader:
            predicted_outputs, _ = model(outputs, inputs)
            predicted_outputs = approximate_with_normal(predicted_outputs)

            mean = predicted_outputs.loc.detach().numpy()
            scale = predicted_outputs.scale.detach().numpy()

            evaluator.evaluate(predicted_outputs, outputs)

            if 'prediction' in plot_list:
                plot_list.remove('prediction')
                fig = plot_pred(mean[0].T, np.sqrt(scale[0]).T, outputs[0].numpy().T)
                fig.axes[0].set_title('{} {} {} Prediction'.format(
                    experiment.model, experiment.dataset, key.capitalize()))
                fig.show()
                fig.savefig('{}prediction_{}.png'.format(experiment.fig_dir, key))

            if '2d' in plot_list:
                plot_list.remove('2d')
                fig = plot_2d(mean[0].T, np.sqrt(scale[0]).T, outputs[0].numpy().T)
                fig.axes[0].set_title('{} {} {} Prediction'.format(
                    experiment.model, experiment.dataset, key.capitalize()))
                fig.show()
                fig.savefig('{}prediction2d_{}.png'.format(experiment.fig_dir, key))

            if 'transition' in plot_list:  # only implemented for 1d.
                plot_list.remove('transition')
                gp = model.forward_model
                transition = model.transitions
                x = torch.arange(-3, 1, 0.1)
                true_next_x = dataset.f(x.numpy())
                pred_next_x = transition(gp(x))

                fig = plot_transition(
                    x.numpy(), true_next_x, pred_next_x.loc.numpy(),
                    torch.diag(pred_next_x.covariance_matrix).sqrt().numpy())
                fig.show()
                fig.savefig('{}transition.png'.format(experiment.fig_dir))

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


def load(experiment: Experiment, key: str) -> list:
    """Load kwarg from experiments.

    Parameters
    ----------
    experiment: Experiment.
        Experiment meata-data.

    key: str.
        Key to load.

    Returns
    -------
    data: list of data.

    """
    save_dir = experiment.log_dir
    values = []

    files = list(filter(lambda x: key in x, os.listdir(save_dir)))
    for file_name in files:
        if key == 'model':
            if file_name[-2:] == 'pt':
                if experiment.configs == {}:
                    configs = load(experiment, 'experiment')[0].configs
                else:
                    configs = experiment.configs
                configs.get('model', {}).pop('name', {})
                dataset = get_dataset(experiment.dataset)
                model = get_model(experiment.model, dataset.dim_outputs,
                                  dataset.dim_inputs, **configs.get('model', {}))
                model.load_state_dict(torch.load(save_dir + file_name))
                values.append(model)
        else:
            with open(save_dir + file_name, 'rb') as file:
                val = pickle.load(file)

            values.append(val)

    return values

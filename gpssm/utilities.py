"""Utilities for training and evaluating."""
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import pickle
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.distributions import Normal
from gpytorch import settings
from gpytorch.distributions import MultivariateNormal
from tqdm import tqdm
from typing import List
from .dataset import get_dataset, Dataset
from .models import get_model, SSM
from .plotters import plot_pred, plot_2d, plot_transition, plot_loss
from collections import namedtuple

from gpssm.dataset.dataset import KinkFunction

__author__ = 'Sebastian Curi'
__all__ = ['Experiment', 'approximate_with_normal', 'train', 'evaluate', 'save', 'load',
           'make_dir', 'dump']


class Evaluator(dict):
    """Object that evaluates the predictive performance of a model."""

    def __init__(self):
        self.criteria = ['loglik', 'nrmse', 'rmse']
        super().__init__({criterion: [] for criterion in self.criteria})
        self._last = {criterion: None for criterion in self.criteria}

    def __str__(self):
        return 'Log-Lik: {:.4}. NRMSE: {:.4}. RMSE: {:.4} '.format(
            np.array(self['loglik']).mean(),
            np.array(self['nrmse']).mean(),
            np.array(self['rmse']).mean()
        )

    @property
    def last(self) -> str:
        return 'Log-Lik: {:.4}. NRMSE: {:.4}. RMSE: {:.4} '.format(
            self._last['loglik'], self._last['nrmse'], self._last['rmse'])

    def evaluate(self, predictions: Normal, true_values: Tensor, scale: Tensor) -> None:
        """Return the RMS error between the true values and the mean predictions.

        Parameters
        ----------
        predictions: MultivariateNormal.
            A multivariate normal with loc [time x dim] and covariance (or scale)
            [time x dim x dim] or [time x dim].
        true_values: Tensor.
            A tensor with shape [time x dim].
        scale: Tensor.
            Output scale.

        Returns
        -------
        criteria: dict.
        """
        for criterion in self.criteria:
            val = getattr(self, criterion)(predictions, true_values, scale)
            self._last[criterion] = val
            self[criterion].append(val)

    @staticmethod
    def loglik(predictions: Normal, true_values: Tensor, _: Tensor = None) -> float:
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
    def nrmse(predictions: Normal, true_values: Tensor, _: Tensor = None) -> float:
        """Return the Normalized RMSE between the true values and the mean predictions.

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

    @staticmethod
    def rmse(predictions: Normal, true_values: Tensor, scale: Tensor = None) -> float:
        """Return the RMSE between the true values and the mean predictions.

        Parameters
        ----------
        predictions: MultivariateNormal.
            A multivariate normal with loc [time x dim] and covariance (or scale)
            [time x dim x dim] or [time x dim].
        true_values: Tensor.
            A tensor with shape [time x dim].
        scale: Tensor.
            A tensor with the scale of each of the dimensions of shape [dim].

        Returns
        -------
        log_likelihood: float.
        """
        l2 = ((predictions.loc - true_values) * scale).pow(2).mean(dim=(1, 2))
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
            log_dir = get_dir(configs['experiment']['name'], fig_dir=False)
        if fig_dir is None:
            fig_dir = get_dir(configs['experiment']['name'], fig_dir=True)
        return super(Experiment, cls).__new__(cls, model, dataset, seed, configs,
                                              log_dir, fig_dir)


def get_dir(exp_name: str, fig_dir: bool = False) -> str:
    """Get the log or figure directory.

    If the directory does not exist, create it.

    Parameters
    ----------
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

    log_directory = base_dir + '/' + exp_name
    make_dir(log_directory)
    return log_directory


def make_dir(name):
    """Make a directory."""
    try:
        os.makedirs(name)
    except FileExistsError:
        pass


def approximate_with_normal(predicted_outputs: List[MultivariateNormal]) -> Normal:
    """Approximate a particle distribution with a Normal by moment matching."""
    sequence_length = len(predicted_outputs)
    batch_size, dim_outputs, _ = predicted_outputs[0].loc.shape

    output_loc = torch.zeros((batch_size, sequence_length, dim_outputs))
    output_cov = torch.zeros((batch_size, sequence_length, dim_outputs))
    for t, y_pred in enumerate(predicted_outputs):
        # Collapse particles!
        output_loc[:, t, :] = y_pred.loc.mean(dim=-1)
        output_cov[:, t, :] = torch.diagonal(y_pred.covariance_matrix, dim1=-1, dim2=-2
                                             ).mean(dim=-1) + y_pred.loc.var(dim=-1)
    return Normal(output_loc, output_cov)


def train(model: SSM, dataloader: DataLoader, optimizer: Optimizer, num_epochs: int,
          experiment: Experiment, test_set: Dataset) -> List[float]:
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
    test_set: Dataset
        Dataset to evaluate model on.

    Returns
    -------
    losses: list of int.
        List of losses encountered during training.
    """
    losses = []
    best_rmse = float('inf')
    output_scale = torch.tensor(test_set.output_normalizer.sd).float()
    model_file = experiment.log_dir + 'model_{}.pt'.format(experiment.seed)
    evaluator = Evaluator()

    train_file = '{}train_epoch.txt'.format(experiment.fig_dir)
    if os.path.exists(train_file):
        os.remove(train_file)

    for i_epoch in tqdm(range(num_epochs)):
        for i_iter, (inputs, outputs) in enumerate(tqdm(dataloader)):
            # Zero the gradients of the Optimizer
            optimizer.zero_grad()

            # Compute the loss.
            predicted_outputs, loss = model.forward(outputs, inputs, print=not i_iter)

            # Back-propagate
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Evaluate
        with torch.no_grad():
            model.eval()
            inputs, outputs = test_set[0]
            inputs, outputs = inputs.unsqueeze(0), outputs.unsqueeze(0)
            _evaluate(model, outputs, inputs, output_scale, evaluator,
                      experiment, 'epoch_{}'.format(i_epoch))
            dump(str(i_epoch + 1) + ' ' + evaluator.last + '\n', train_file, 'a+')
            if evaluator['rmse'][-1] < best_rmse:
                best_rmse = evaluator['rmse'][-1]
                torch.save(model.state_dict(), model_file)
            model.train()
        print(model)

    fig = plot_loss(losses, ylabel=model.loss_key.upper())
    fig.gca().set_title('{} {} Training Loss'.format(
        experiment.model, experiment.dataset))
    fig.show()
    fig.savefig('{}training_loss.png'.format(experiment.fig_dir))
    plt.close(fig)
    model.load_state_dict(torch.load(model_file))

    return losses


def evaluate(model: SSM, dataloader: DataLoader, experiment: Experiment, key: str = ''
             ) -> Evaluator:
    """Evaluate a model.

    Parameters
    ----------
    model: GPSSM.
        Model to train.
    dataloader: DataLoader.
        Loader to iterate data.
    experiment: Experiment.
        Experiment meta-data.
    key: str.
        Key to end files with.

    """
    dataset = dataloader.dataset  # type: Dataset  # type: ignore
    output_scale = torch.tensor(dataset.output_normalizer.sd).float()
    evaluator = Evaluator()

    with torch.no_grad():
        model.eval()
        for inputs, outputs in dataloader:
            _evaluate(model, outputs, inputs, output_scale, evaluator, experiment,
                      key)
    return evaluator


def _evaluate(model: SSM, outputs: Tensor, inputs: torch.Tensor, output_scale: Tensor,
              evaluator: Evaluator, experiment: Experiment, key: str) -> None:
    with settings.fast_pred_samples(state=True), settings.fast_pred_var(state=True):
        # predicted_outputs = model.predict(outputs, inputs)
        predicted_outputs, _ = model.forward(outputs, inputs)
    collapsed_predicted_outputs = approximate_with_normal(predicted_outputs)

    mean = collapsed_predicted_outputs.loc.detach().numpy()
    scale = collapsed_predicted_outputs.scale.detach().numpy()

    evaluator.evaluate(collapsed_predicted_outputs, outputs, output_scale)
    print('\n' + evaluator.last)

    fig = plot_pred(mean[-1].T, np.sqrt(scale[-1]).T, outputs[-1].numpy().T)
    fig.axes[0].set_title('{} {} {} Prediction'.format(
        experiment.model, experiment.dataset, key.capitalize()))
    fig.show()
    fig.savefig('{}prediction_{}.png'.format(experiment.fig_dir, key))
    plt.close(fig)

    if 'robomove' in experiment.dataset.lower():
        fig = plot_2d(mean[-1].T, outputs[-1].numpy().T)
        fig.axes[0].set_title('{} {} {} Prediction'.format(
            experiment.model, experiment.dataset, key.capitalize()))
        fig.show()
        fig.savefig('{}prediction2d_{}.png'.format(experiment.fig_dir, key))
        plt.close(fig)

    if 'kink' in experiment.dataset.lower():
        gp = model.forward_model
        transition = model.transitions
        x = torch.arange(-3, 1, 0.1)
        true_next_x = KinkFunction.f(x.numpy())
        pred_next_x = transition(gp(x.expand(1, model.dim_states, -1)))

        fig = plot_transition(
            x.numpy(), true_next_x, pred_next_x.loc[-1, -1].numpy(),
            torch.diag(pred_next_x.covariance_matrix[-1, -1]).sqrt().numpy())
        fig.axes[0].set_title('{} {} Learned Function'.format(
            experiment.model, experiment.dataset))
        fig.show()
        fig.savefig('{}transition.png'.format(experiment.fig_dir))
        plt.close(fig)


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


def dump(string: str, file_name: str, mode='w') -> None:
    """Dump string to file."""
    with open(file_name, mode) as file:
        file.write(string)

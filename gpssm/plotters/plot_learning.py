"""Plotting functions for learning-related stats."""
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Iterable


def plot_loss(losses: Union[Iterable, dict], ylabel: str = 'Losses') -> plt.Figure:
    """Plot losses encountered during training.

    Parameters
    ----------
    losses: iterable or dict.
        List or array of size [num_iterations] with losses encountered during training.
        Dict of lists of size [num_iterations] with different losses encountered during
        training. The key of the dict will be used as legend.

    ylabel: str, optional.
        Label of y-axes, default=Losses.

    Returns
    -------
    fig: plt.Figure

    """
    fig, ax = plt.subplots()
    if isinstance(losses, Union[np.ndarray, list].__args__):
        ax.plot(losses)
    elif isinstance(losses, dict):
        for key, loss in losses.items():
            ax.plot(loss, label=key)
        ax.legend(loc='best')
    else:
        raise TypeError

    ax.set_xlabel('Iterations')
    ax.set_ylabel(ylabel)

    return fig


def plot_evaluation_datasets(losses: dict) -> plt.Figure:
    """Plot the evaluation of datasets.

    This is a bar chart, each x-label is a dataset and each dataset is divided into
    columns. Each column is each of the evaluation criterion.

    Parameters
    ----------
    losses: dict.
        Dictionary of losses. Each key is a dataset.
        Each dataset has a dictionary with the different evaluation criteria.
        Each criteria contains a list [mean, standard deviation].

    Returns
    -------
    fig: plt.Figure

    """
    datasets = list(losses.keys())
    criteria = list(losses[datasets[0]].keys())
    x = np.arange(len(datasets))

    values = []
    for dataset in datasets:
        values.append(losses[dataset])

    fig, ax = plt.subplots()
    width = 0.3  # the width of the bars
    for idx, criterion in enumerate(criteria):
        ax.bar(x + (idx - 1) * width,
               list(a[criterion][0] for a in values),
               width,
               yerr=list(a[criterion][1] for a in values),
               label=criterion
               )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=0)
    ax.legend(loc='best')

    return fig

"""Plotting functions for sequences of data."""
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Sebastian Curi'
__all__ = ['plot_pred', 'plot_2d', 'plot_transition']

TRUE_COLOR = 'C0'
PRED_COLOR = 'C1'


def plot_pred(pred_mean: np.ndarray, pred_std: np.ndarray = None,
              true_outputs: np.ndarray = None, sigmas: float = 1.) -> plt.Figure:
    """Plot predictions made by the model.

    Parameters
    ----------
    pred_mean: np.ndarray.
        Predicted mean of shape [dim_outputs, time].
    pred_std: np.ndarray, optional.
        Predicted standard deviation of shape [dim_outputs, time].
    true_outputs: np.ndarray, optional.
        True outputs of shape [dim_outputs, time].
    sigmas: float.
        Number of standard deviations to plot.

    Returns
    -------
    figure: plt.Figure
    """
    legend = False

    dim_outputs, time = pred_mean.shape
    if pred_std is not None:
        assert pred_mean.shape == pred_std.shape, """
        Mean and standard deviation must have the same shape.
        """

    if true_outputs is not None:
        legend = True
        assert pred_mean.shape == true_outputs.shape, """
        Prediction and Target must have the same shape.
        """

    fig, axes = plt.subplots(dim_outputs, 1, sharex='all')
    if dim_outputs == 1:
        axes = [axes]

    for idx in range(dim_outputs):
        if true_outputs is not None:
            axes[idx].plot(true_outputs[idx], color=TRUE_COLOR, label='Ground Truth')

        axes[idx].plot(pred_mean[idx], color=PRED_COLOR, label='Predicted Output')
        if pred_std is not None:
            axes[idx].fill_between(
                np.arange(time),
                (pred_mean - sigmas * pred_std)[idx],
                (pred_mean + sigmas * pred_std)[idx],
                alpha=0.2, facecolor=PRED_COLOR)
        axes[idx].set_ylabel('y_{}'.format(idx + 1))
        if legend and idx == 0:
            axes[idx].legend(loc='best')

    axes[-1].set_xlabel('Time')
    return fig


def plot_2d(pred_mean: np.ndarray, true_outputs: Optional[np.ndarray] = None) -> plt.Figure:
    """Plot predictions made by the model in 2d.

    Parameters
    ----------
    pred_mean: np.ndarray.
        Predicted mean of shape [dim_outputs, time].
    true_outputs: np.ndarray.
        True outputs of shape [dim_outputs, time].

    Returns
    -------
    figure: plt.Figure.
    """
    legend = False

    dim_outputs, time = pred_mean.shape
    assert dim_outputs >= 2, """At least two outputs."""

    if true_outputs is not None:
        legend = True
        assert pred_mean.shape == true_outputs.shape, """
        Prediction and Target must have the same shape.
        """

    fig, ax = plt.subplots()
    if true_outputs is not None:
        ax.plot(true_outputs[0], true_outputs[1], color=TRUE_COLOR, label='Ground Truth')
    ax.plot(pred_mean[0], pred_mean[1], color=PRED_COLOR, label='Predicted Output')

    if legend:
        ax.legend(loc='best')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return fig


def plot_transition(state: np.ndarray, true_next_state: np.ndarray,
                    pred_next_state_mu: np.ndarray, pred_next_state_std: np.ndarray,
                    sigmas: float = 3.) -> plt.Figure:
    """Plot the predicted transition function from samples.

    Parameters
    ----------
    state: np.ndarray.
        State sequence of shape [time]
    true_next_state: np.ndarray.
        Next state sequence of shape [time]
    pred_next_state_mu: np.ndarray.
        Mean predicted next state sequence of shape [time]
    pred_next_state_std: np.ndarray.
        Standard devaiation of predicted next state sequence of shape [time]
    sigmas: float.
        Number of standard deviations to plot.

    Returns
    -------
    fig: plt.Figure.
    """
    time = len(state)
    assert time == len(true_next_state), "state and next state don't have same length."
    assert time == len(pred_next_state_mu), "state and next state mu not eq length."
    assert time == len(pred_next_state_std), "state and next state std not eq length."

    fig, ax = plt.subplots()
    ax.plot(state, true_next_state, color=TRUE_COLOR, label='Ground Truth')
    ax.plot(state, pred_next_state_mu, color=PRED_COLOR, label='Predicted Function')
    ax.fill_between(state,
                    pred_next_state_mu - sigmas * pred_next_state_std,
                    pred_next_state_mu + sigmas * pred_next_state_std,
                    alpha=0.2, facecolor=PRED_COLOR)

    ax.set_xlabel('state')
    ax.set_ylabel('next state')
    ax.legend(loc='best')
    return fig

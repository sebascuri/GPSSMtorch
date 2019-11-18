"""Plotting functions for sequences of data."""

import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Sebastian Curi'
__all__ = ['plot_pred', 'plot_2d', 'plot_transition', 'plot_input_output']


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
        axes[idx].plot(pred_mean[idx], color='blue', label='Predicted Output')
        if pred_std is not None:
            axes[idx].fill_between(
                np.arange(time),
                (pred_mean - pred_std)[idx],
                (pred_mean + pred_std)[idx],
                alpha=0.2, facecolor='blue')

        if true_outputs is not None:
            axes[idx].plot(true_outputs[idx], color='red', label='True Output')

        axes[idx].set_ylabel('y_{}'.format(idx + 1))
        if legend:
            axes[idx].legend(loc='best')

    axes[-1].set_xlabel('Time')
    return fig


def plot_2d(pred_mean: np.ndarray, pred_std: np.ndarray = None,
            true_outputs: np.ndarray = None, sigmas: float = 1.) -> plt.Figure:
    """Plot predictions made by the model in 2d.

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
    figure: plt.Figure.
    """
    legend = False

    dim_outputs, time = pred_mean.shape
    assert dim_outputs >= 2, """At least two outputs."""
    if pred_std is not None:
        assert pred_mean.shape == pred_std.shape, """
        Mean and standard deviation must have the same shape.
        """

    if true_outputs is not None:
        legend = True
        assert pred_mean.shape == true_outputs.shape, """
        Prediction and Target must have the same shape.
        """

    fig, ax = plt.subplots()
    ax.plot(pred_mean[0], pred_mean[1], color='blue', label='Predicted Output')
    if pred_std is not None:
        ax.fill_between(pred_mean[0],
                        (pred_mean - sigmas * pred_std)[1],
                        (pred_mean + sigmas * pred_std)[1],
                        alpha=0.2, facecolor='blue')
        ax.fill_betweenx(pred_mean[1],
                         (pred_mean - sigmas * pred_std)[0],
                         (pred_mean + sigmas * pred_std)[0],
                         alpha=0.2, facecolor='blue')
    if true_outputs is not None:
        ax.plot(true_outputs[0], true_outputs[1], color='red', label='True Output')

    if legend:
        ax.legend(loc='best')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return fig


def plot_transition(state: np.ndarray, true_next_state: np.array,
                    pred_next_state_mu: np.ndarray, pred_next_state_std: np.ndarray,
                    sigmas: float = 1.) -> plt.Figure:
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
    ax.plot(state, true_next_state, color='red', label='True Function')
    ax.plot(state, pred_next_state_mu, color='blue', label='Predicted Function')
    ax.fill_between(state,
                    pred_next_state_mu - sigmas * pred_next_state_std,
                    pred_next_state_mu + sigmas * pred_next_state_std,
                    alpha=0.2, facecolor='blue')

    ax.set_xlabel('state')
    ax.set_ylabel('next_state')
    return fig


def plot_input_output(output_sequence: np.ndarray, input_sequence: np.ndarray,
                      single_plot: bool = False) -> plt.Figure:
    """Plot input/output data.

    It will plot the output sequence first and the input sequence next.
    If single_plot is True, then it will plot everything in a single subplot, else each
    sequence in a different subplot.

    Parameters
    ----------
    output_sequence: np.ndarray.
        Output sequence of shape [dim_outputs, time]
    input_sequence: np.ndarray.
        Input sequence of shape [dim_inputs, time]
    single_plot: bool, optional. (default=False).
        Flag that indicates if the input/output data should be plot in a single supblot
        (when True) or in multiple-sublplots (when False).

    Returns
    -------
    figure: plt.Figure.
    """
    dim_outputs, time = output_sequence.shape
    dim_inputs, aux = input_sequence.shape
    assert time == aux, "input and output sequence must have the same length."

    if single_plot:
        fig, ax1 = plt.subplots()
        idx = 0
        for output_seq in output_sequence:
            ax1.plot(output_seq, '-', label='y_{}'.format(idx + 1))
            idx += 1

        if dim_outputs > 0:
            ax1.set_ylabel('outputs')
            ax1.legend(loc='upper left')
            if dim_inputs > 0:
                ax2 = ax1.twinx()
        else:
            ax2 = ax1

        for input_seq in input_sequence:
            ax2.plot(input_seq, '--', label='u_{}'.format(idx - dim_outputs + 1))
            idx += 1
        if dim_inputs > 0:
            ax2.set_ylabel('inputs')
            ax2.legend(loc='upper right')

    else:  # Multiple-subplots
        fig, axes = plt.subplots(dim_outputs + dim_inputs, sharex='all')
        idx = 0
        if dim_outputs + dim_inputs == 1:
            axes = [axes]
        for output_seq in output_sequence:
            axes[idx].plot(output_seq)
            axes[idx].set_ylabel('y_{}'.format(idx + 1))
            idx += 1

        for input_seq in input_sequence:
            axes[idx].plot(input_seq)
            axes[idx].set_ylabel('u_{}'.format(idx - dim_outputs + 1))
            idx += 1

        axes[-1].set_xlabel('Time')

    return fig

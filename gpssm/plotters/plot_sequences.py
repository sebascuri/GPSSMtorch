"""Plotting functions for sequences of data."""

# TODO: Add uncertainty plots.

import numpy as np
import matplotlib.pyplot as plt


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
    figure: plt.Figure

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


def plot_transition(state_sequence: np.ndarray, next_state_sequence: np.ndarray
                    ) -> plt.Figure:
    """Plot the predicted transition function from samples.

    Parameters
    ----------
    state_sequence: np.ndarray.
        State sequence of shape [time]
    next_state_sequence: np.ndarray.
        Next state sequence of shape [time]

    Returns
    -------
    fig: plt.Figure

    """
    time = state_sequence.shape
    aux = next_state_sequence.shape
    assert time == aux, "input and output sequence must have the same length."

    fig, ax = plt.subplots()
    ax.plot(state_sequence, next_state_sequence)
    ax.set_xlabel('state')
    ax.set_ylabel('next_state')
    return fig

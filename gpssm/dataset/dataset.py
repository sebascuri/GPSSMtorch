"""Classes to access data."""

import torch
from torch.utils import data
import numpy as np
import scipy.io as sio
import os
from .utilities import get_data_split, generate_batches, generate_trajectory, Normalizer

__author__ = 'Sebastian Curi'
__all__ = ['Actuator', 'BallBeam', 'Drive', 'Dryer', 'Flutter', 'GasFurnace', 'Tank',
           'Sarcos', 'NonLinearSpring', 'RoboMove', 'RoboMoveSimple', 'KinkFunction',
           'Dataset']

DATA_DIR = 'data/'


class Dataset(data.TensorDataset):
    """Dataset handler for time-series data.

    Parameters
    ----------
    outputs: np.ndarray.
        Array of shape [n_experiment, time, dim] with outputs of the time series.

    inputs: np.ndarray, optional (default: data/).
        Array of shape [n_experiment, time, dim] with inputs of the time series.

    states: bool, optional (default: True).
        Array of shape [n_experiment, time, dim] with hidden states of the time series.

    sequence_length: int, optional (default: full sequence).
        Integer that indicates how long the sequences should be.

    sequence_stride: int, optional (default: 1).
        Integer that indicates every how many time-steps the sequences start.

    """

    def __init__(self, outputs: np.ndarray,
                 inputs: np.ndarray = None, states: np.ndarray = None,
                 sequence_length: int = None, sequence_stride: int = 1) -> None:

        assert outputs.ndim == 3, 'Outputs shape is [n_experiment, time, dim]'
        self.num_experiments, self.experiment_length, self.dim_outputs = outputs.shape
        self._sequence_length = sequence_length
        self._sequence_stride = sequence_stride

        if inputs is not None:
            assert inputs.ndim == 3, 'Inputs shape is [n_experiment, time, dim]'
            assert inputs.shape[0] == outputs.shape[0], """
                Inputs and outputs must have the same number of experiments"""
            assert inputs.shape[1] == outputs.shape[1], """
                Inputs and outputs experiments should be equally long"""
        else:
            inputs = np.zeros((self.num_experiments, self.experiment_length, 0))

        self.dim_inputs = inputs.shape[2]

        if states is not None:
            assert states.ndim == 3, 'States shape is [n_experiment, time, dim]'
            assert states.shape[0] == outputs.shape[0], """
                States and outputs must have the same number of experiments"""
            assert states.shape[1] == outputs.shape[1], """
                States and outputs experiments should be equally long"""
        else:
            states = np.zeros((self.num_experiments, self.experiment_length, 0))

        self.dim_states = states.shape[2]

        # Store normalized inputs, outputs, states.
        self.input_normalizer = Normalizer(inputs)
        self.output_normalizer = Normalizer(outputs)
        self.state_normalizer = Normalizer(states)

        self.inputs = self.input_normalizer(inputs)
        self.outputs = self.output_normalizer(outputs)
        self.states = self.state_normalizer(states)

        super().__init__(*[torch.tensor(
            generate_batches(x, self._sequence_length, self._sequence_stride)).float()
                           for x in [self.inputs, self.outputs, self.states]])

    def __str__(self):
        """Return string with dataset statistics."""
        string = 'input dim: {} \noutput dim: {} \nstate dim: {} \n\n'.format(
            self.dim_inputs, self.dim_outputs, self.dim_states
        )
        string += 'sequence length: {} \n'.format(
            self.outputs.shape[1]
        )
        string += 'train_samples: {} \ntrain_sequences: {} \n'.format(
            self.experiment_length, self.outputs.shape[0]
        )
        return string

    @property
    def sequence_length(self) -> int:
        """Get sequence length."""
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, new_seq_length):
        """Set sequence length and reshape the tensors.

        Parameters
        ----------
        new_seq_length: int.
        """
        self._sequence_length = new_seq_length
        self.tensors = [torch.tensor(
            generate_batches(x, self._sequence_length, self._sequence_stride)).float()
                        for x in [self.inputs, self.outputs, self.states]]


class Actuator(Dataset):
    """Actuator dataset implementation.

    Train Split: first half of the data.
    Test Split: second half of the data.

    Parameters
    ----------
    data_dir: str, optional (default: data/).
        Directory where `actuator.mat' is located.

    train: bool, optional (default: True).
        Flag that indicates dataset split.

    sequence_length: int, optional (default: full sequence).
        Integer that indicates how long the sequences should be.

    sequence_stride: int, optional (default: 1).
        Integer that indicates every how many time-steps the sequences start.



    References
    ----------
    https://github.com/zhenwendai/RGP/tree/master/datasets/system_identification

    """

    def __init__(self, data_dir: str = DATA_DIR, train: bool = True,
                 sequence_length: int = 512, sequence_stride: int = 1) -> None:
        raw_data = sio.loadmat(os.path.join(data_dir, 'actuator.mat'))

        inputs = get_data_split(raw_data['u'][np.newaxis], train=train)
        outputs = get_data_split(raw_data['p'][np.newaxis], train=train)
        states = get_data_split(raw_data['x'][np.newaxis], train=train)

        super().__init__(inputs=inputs, outputs=outputs, states=states,
                         sequence_length=sequence_length,
                         sequence_stride=sequence_stride)


class BallBeam(Dataset):
    """BallBeam dataset implementation.

    Train Split: first half of the data.
    Test Split: second half of the data.

    Parameters
    ----------
    data_dir: str, optional (default: data/).
        Directory where `actuator.mat' is located.

    train: bool, optional (default: True).
        Flag that indicates dataset split.

    sequence_length: int, optional (default: full sequence).
        Integer that indicates how long the sequences should be.

    sequence_stride: int, optional (default: 1).
        Integer that indicates every how many time-steps the sequences start.



    References
    ----------
    http://homes.esat.kuleuven.be/~smc/daisy/daisydata.html
    [96-004] Data of the ball-and-beam setup in STADIUS

    """

    def __init__(self, data_dir: str = DATA_DIR, train: bool = True,
                 sequence_length: int = 500, sequence_stride: int = 1) -> None:
        raw_data = np.loadtxt(os.path.join(data_dir, 'ballbeam.dat'))

        inputs = get_data_split(raw_data[np.newaxis, :, 0, np.newaxis], train=train)
        outputs = get_data_split(raw_data[np.newaxis, :, 1, np.newaxis], train=train)

        super().__init__(inputs=inputs, outputs=outputs,
                         sequence_length=sequence_length,
                         sequence_stride=sequence_stride)


class Drive(Dataset):
    """Drive dataset implementation.

    Train Split: first half of the data.
    Test Split: second half of the data.

    Parameters
    ----------
    data_dir: str, optional (default: data/).
        Directory where `actuator.mat' is located.

    train: bool, optional (default: True).
        Flag that indicates dataset split.

    sequence_length: int, optional (default: full sequence).
        Integer that indicates how long the sequences should be.

    sequence_stride: int, optional (default: 1).
        Integer that indicates every how many time-steps the sequences start.



    References
    ----------
    http://homes.esat.kuleuven.be/~smc/daisy/daisydata.html

    """

    def __init__(self, data_dir: str = DATA_DIR, train: bool = True,
                 sequence_length: int = 250, sequence_stride: int = 1) -> None:
        raw_data = sio.loadmat(os.path.join(data_dir, 'drive.mat'))

        inputs = get_data_split(
            np.stack([raw_data['u{}'.format(i)] for i in range(1, 4)]), train=train)
        outputs = get_data_split(
            np.stack([raw_data['z{}'.format(i)] for i in range(1, 4)]), train=train)

        super().__init__(inputs=inputs, outputs=outputs,
                         sequence_length=sequence_length,
                         sequence_stride=sequence_stride)


class Dryer(Dataset):
    """Hair Dryer dataset implementation.

    Train Split: first half of the data.
    Test Split: second half of the data.

    Parameters
    ----------
    data_dir: str, optional (default: data/).
        Directory where `actuator.mat' is located.

    train: bool, optional (default: True).
        Flag that indicates dataset split.

    sequence_length: int, optional (default: full sequence).
        Integer that indicates how long the sequences should be.

    sequence_stride: int, optional (default: 1).
        Integer that indicates every how many time-steps the sequences start.



    References
    ----------
    http://homes.esat.kuleuven.be/~smc/daisy/daisydata.html
    [96-006] Data of a laboratory setup acting like a hair dryer

    """

    def __init__(self, data_dir: str = DATA_DIR, train: bool = True,
                 sequence_length: int = 500, sequence_stride: int = 1) -> None:
        raw_data = np.loadtxt(os.path.join(data_dir, 'dryer.dat'))

        inputs = get_data_split(raw_data[np.newaxis, :, 0, np.newaxis], train=train)
        outputs = get_data_split(raw_data[np.newaxis, :, 1, np.newaxis], train=train)

        super().__init__(inputs=inputs, outputs=outputs,
                         sequence_length=sequence_length,
                         sequence_stride=sequence_stride)


class Flutter(Dataset):
    """Flutter dataset implementation.

    Train Split: first half of the data.
    Test Split: second half of the data.

    Parameters
    ----------
    data_dir: str, optional (default: data/).
        Directory where `actuator.mat' is located.

    train: bool, optional (default: True).
        Flag that indicates dataset split.

    sequence_length: int, optional (default: full sequence).
        Integer that indicates how long the sequences should be.

    sequence_stride: int, optional (default: 1).
        Integer that indicates every how many time-steps the sequences start.



    References
    ----------
    http://homes.esat.kuleuven.be/~smc/daisy/daisydata.html
    [96-008] Wing flutter data

    """

    def __init__(self, data_dir: str = DATA_DIR, train: bool = True,
                 sequence_length: int = 512, sequence_stride: int = 1) -> None:
        raw_data = np.loadtxt(os.path.join(data_dir, 'flutter.dat'))

        inputs = get_data_split(raw_data[np.newaxis, :, 0, np.newaxis], train=train)
        outputs = get_data_split(raw_data[np.newaxis, :, 1, np.newaxis], train=train)

        super().__init__(inputs=inputs, outputs=outputs,
                         sequence_length=sequence_length,
                         sequence_stride=sequence_stride)


class GasFurnace(Dataset):
    """Gas Furnace dataset implementation.

    Train Split: first half of the data.
    Test Split: second half of the data.

    Parameters
    ----------
    data_dir: str, optional (default: data/).
        Directory where `actuator.mat' is located.

    train: bool, optional (default: True).
        Flag that indicates dataset split.

    sequence_length: int, optional (default: full sequence).
        Integer that indicates how long the sequences should be.

    sequence_stride: int, optional (default: 1).
        Integer that indicates every how many time-steps the sequences start.



    References
    ----------
    https://openmv.net/info/gas-furnace

    """

    def __init__(self, data_dir: str = DATA_DIR, train: bool = True,
                 sequence_length: int = 148, sequence_stride: int = 1) -> None:
        raw_data = np.loadtxt(os.path.join(data_dir, 'gas_furnace.csv'),
                              skiprows=1, delimiter=',')

        inputs = get_data_split(raw_data[np.newaxis, :, 0, np.newaxis], train=train)
        outputs = get_data_split(raw_data[np.newaxis, :, 1, np.newaxis], train=train)

        super().__init__(inputs=inputs, outputs=outputs,
                         sequence_length=sequence_length,
                         sequence_stride=sequence_stride)


class Tank(Dataset):
    """Tank dataset implementation.

    Train Split: first half of the data.
    Test Split: second half of the data.

    Parameters
    ----------
    data_dir: str, optional (default: data/).
        Directory where `actuator.mat' is located.

    train: bool, optional (default: True).
        Flag that indicates dataset split.

    sequence_length: int, optional (default: full sequence).
        Integer that indicates how long the sequences should be.

    sequence_stride: int, optional (default: 1).
        Integer that indicates every how many time-steps the sequences start.



    References
    ----------
    https://github.com/zhenwendai/RGP/tree/master/datasets/system_identification

    """

    def __init__(self, data_dir: str = DATA_DIR, train: bool = True,
                 sequence_length: int = 1250, sequence_stride: int = 1) -> None:
        raw_data = sio.loadmat(os.path.join(data_dir, 'tank.mat'))

        inputs = get_data_split(raw_data['u'].T[np.newaxis], train=train)
        outputs = get_data_split(raw_data['y'].T[np.newaxis], train=train)

        super().__init__(inputs=inputs, outputs=outputs,
                         sequence_length=sequence_length,
                         sequence_stride=sequence_stride)


class Sarcos(Dataset):
    """Sarcos dataset implementation.

    Train Split: first 60 subsequences of the data.
    Test Split: last 6 subsequences of the data.

    Parameters
    ----------
    data_dir: str, optional (default: data/).
        Directory where `actuator.mat' is located.

    train: bool, optional (default: True).
        Flag that indicates dataset split.

    sequence_length: int, optional (default: full sequence).
        Integer that indicates how long the sequences should be.

    sequence_stride: int, optional (default: 1).
        Integer that indicates every how many time-steps the sequences start.



    References
    ----------
    http://www.gaussianprocess.org/gpml/data/

    """

    def __init__(self, data_dir: str = DATA_DIR, train: bool = True,
                 sequence_length: int = None, sequence_stride: int = 1) -> None:
        raw_data = sio.loadmat(os.path.join(data_dir, 'sarcos_inv.mat'))
        raw_data = raw_data['sarcos_inv']

        split_idx = 60  # Note that SARCOS is divided inter-experiment.
        exp_len = 674
        downsample = 2
        subsampled_data = np.stack([raw_data[ind:ind + exp_len:downsample, :] for ind in
                                    range(0, raw_data.shape[0], exp_len)])

        if train:
            inputs = subsampled_data[:split_idx, :, 21:28]
            outputs = subsampled_data[:split_idx, :, 0:7]
        else:
            inputs = subsampled_data[split_idx:, :, 21:28]
            outputs = subsampled_data[split_idx:, :, 0:7]
        super().__init__(inputs=inputs, outputs=outputs,
                         sequence_length=sequence_length,
                         sequence_stride=sequence_stride)


class NonLinearSpring(Dataset):
    """Non-Linear Spring dataset implementation.

    Train Split: first half of the data.
    Test Split: second half of the data.

    Parameters
    ----------
    data_dir: str, optional (default: data/).
        Directory where `actuator.mat' is located.

    train: bool, optional (default: True).
        Flag that indicates dataset split.

    sequence_length: int, optional (default: full sequence).
        Integer that indicates how long the sequences should be.

    sequence_stride: int, optional (default: 1).
        Integer that indicates every how many time-steps the sequences start.



    """

    def __init__(self, data_dir: str = DATA_DIR, train: bool = True,
                 sequence_length: int = None, sequence_stride: int = 1) -> None:
        raw_data = sio.loadmat(os.path.join(data_dir, 'spring_nonlinear.mat'))

        inputs = get_data_split(raw_data['ds_u'][np.newaxis], train=train)
        outputs = get_data_split(raw_data['ds_y'][np.newaxis], train=train)
        states = get_data_split(raw_data['ds_x'][np.newaxis], train=train)

        super().__init__(inputs=inputs, outputs=outputs, states=states,
                         sequence_length=sequence_length,
                         sequence_stride=sequence_stride)


class RoboMove(Dataset):
    """RoboMove dataset implementation.

    Train Split: first 25000 data points.
    Test Split: last 5000 data points.

    Parameters
    ----------
    data_dir: str, optional (default: data/).
        Directory where `actuator.mat' is located.

    train: bool, optional (default: True).
        Flag that indicates dataset split.

    sequence_length: int, optional (default: full sequence).
        Integer that indicates how long the sequences should be.

    sequence_stride: int, optional (default: 1).
        Integer that indicates every how many time-steps the sequences start.



    """

    def __init__(self, data_dir: str = DATA_DIR, train: bool = True,
                 sequence_length: int = None, sequence_stride: int = 1) -> None:
        raw_data = sio.loadmat(os.path.join(data_dir, 'robomove.mat'))
        split_idx = 25000

        inputs = get_data_split(raw_data['ds_u'][np.newaxis], split_idx, train=train)
        outputs = get_data_split(raw_data['ds_y'][np.newaxis], split_idx, train=train)
        states = get_data_split(raw_data['ds_x'][np.newaxis], split_idx, train=train)

        super().__init__(inputs=inputs, outputs=outputs, states=states,
                         sequence_length=sequence_length,
                         sequence_stride=sequence_stride)


class RoboMoveSimple(Dataset):
    """RoboMove Simple dataset implementation.

    Train Split: first 25000 data points.
    Test Split: last 5000 data points.

    Parameters
    ----------
    data_dir: str, optional (default: data/).
        Directory where `actuator.mat' is located.

    train: bool, optional (default: True).
        Flag that indicates dataset split.

    sequence_length: int, optional (default: full sequence).
        Integer that indicates how long the sequences should be.

    sequence_stride: int, optional (default: 1).
        Integer that indicates every how many time-steps the sequences start.



    """

    def __init__(self, data_dir: str = DATA_DIR, train: bool = True,
                 sequence_length: int = None, sequence_stride: int = 1) -> None:
        raw_data = sio.loadmat(os.path.join(data_dir, 'robomove_simple.mat'))
        split_idx = 25000

        inputs = get_data_split(raw_data['ds_u'][np.newaxis], split_idx, train=train)
        outputs = get_data_split(raw_data['ds_y'][np.newaxis], split_idx, train=train)
        states = get_data_split(raw_data['ds_x'][np.newaxis], split_idx, train=train)

        super().__init__(inputs=inputs, outputs=outputs, states=states,
                         sequence_length=sequence_length,
                         sequence_stride=sequence_stride)


class KinkFunction(Dataset):
    """Kink Function dataset implementation.

    The Kink function is:
        f(x) = 0.8 + (x + 0.2) * (1 - 5 / (1 + exp(-2 * x)))

    The trajectory is generated as:
        x_{k+1} = f(x_k) + process_noise
        y_k = x_k + observation_noise

    Parameters
    ----------
    data_dir: str, optional (default: data/).
        Directory where `actuator.mat' is located.

    train: bool, optional (default: True).
        Flag that indicates dataset split.

    sequence_length: int, optional (default: full sequence).
        Integer that indicates how long the sequences should be.

    sequence_stride: int, optional (default: 1).
        Integer that indicates every how many time-steps the sequences start.



    trajectory_length: int, optional (default = 120).
            Length of trajectory.

    x0: float, optional (default = 0.5).
        Initial state.

    process_noise_sd: float, optional (default = 0.05).
        Standard deviation of the process noise.

    observation_noise_sd: float, optional (default = sqrt(0.8)).
        Standard deviation of observation noise.

    References
    ----------
    Ialongo, A. D., Van Der Wilk, M., Hensman, J., & Rasmussen, C. E. (2019, May).
    Overcoming Mean-Field Approximations in Recurrent Gaussian Process Models.
    In International Conference on Machine Learning (pp. 2931-2940).

    """

    def __init__(self, data_dir: str = DATA_DIR, train: bool = True,
                 sequence_length: int = 60, sequence_stride: int = 1,
                 trajectory_length: int = 120, x0: float = 0.5,
                 process_noise_sd: float = 0.05,
                 observation_noise_sd: float = np.sqrt(0.8)) -> None:

        file_name = os.path.join(data_dir, 'kink_function.mat')
        if not os.path.exists(file_name):
            def f(x: np.ndarray, _: np.ndarray) -> np.ndarray:
                """Kink transition function."""
                return 0.8 + (x + 0.2) * (1 - 5 / (1 + np.exp(- 2 * x)))

            def g(x: np.ndarray, _: np.ndarray) -> np.ndarray:
                """Kink observation function."""
                return x

            states, outputs = generate_trajectory(
                f, g, trajectory_length=trajectory_length, x0=np.array([x0]),
                process_noise_sd=np.array([process_noise_sd]),
                observation_noise_sd=np.array([observation_noise_sd]))

            sio.savemat(file_name, {
                'ds_x': states,
                'ds_y': outputs,
                'title': 'Kink Function'
            })
            states = states[np.newaxis]
            outputs = outputs[np.newaxis]

        else:
            raw_data = sio.loadmat(file_name)
            states = raw_data['ds_x'][np.newaxis]
            outputs = raw_data['ds_y'][np.newaxis]

        outputs = get_data_split(outputs, train=train)
        states = get_data_split(states, train=train)

        super().__init__(outputs=outputs, states=states,
                         sequence_length=sequence_length,
                         sequence_stride=sequence_stride)

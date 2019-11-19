from .utilities import is_leonhard
from .runners import AbstractRunner, LeonhardRunner, SingleRunner


def init_runner(num_threads: int = 1, use_gpu: bool = False,
                wall_time: int = None, memory: int = None) -> AbstractRunner:
    """Initialize the runner.

    Parameters
    ----------
    num_threads: int, optional
        Number of threads to use.
    use_gpu: bool, optional
        Flag to indicate GPU usage.
    wall_time: int, optional
        Required time, in minutes, to run the process.
    memory: int, optional
        Required memory, in MB, to run run the process.

    Returns
    -------
    runner: AbstractRunner

    """
    if is_leonhard():
        return LeonhardRunner(num_threads, use_gpu, wall_time, memory)
    else:
        return SingleRunner(num_threads, use_gpu)

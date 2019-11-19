"""Utilities for runner submodule."""

import socket
import multiprocessing
import os
import numpy as np
import torch


__author__ = 'Sebastian Curi'
__all__ = ['is_leonhard', 'start_process', 'get_free_gpu', 'get_gpu_count']


def is_leonhard():
    """Check if host is Leonhard."""
    hostname = socket.gethostname()
    return 'lo-login' in hostname


def start_process(target, args=None):
    """Start a process from with the multiprocessing framework.

    Parameters
    ----------
    target: callable
    args: tuple, optional

    Returns
    -------
    p: Process

    """
    if args:
        p = multiprocessing.Process(target=target, args=args)
    else:
        p = multiprocessing.Process(target=target)
    p.start()
    return p


def get_free_gpu():
    """Get the GPU with largest free memory.

    Returns
    -------
    gpu: int

    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_gpu')
    memory_available = [int(x.split()[2])
                        for x in open('tmp_gpu', 'r').readlines()]
    return np.argmax(memory_available)


def get_gpu_count():
    """Get number of GPUs.

    Returns
    -------
    number: int

    """
    return torch.cuda.device_count()

""" Random number generation and reproducibility. """
import random
from typing import Optional

import torch
from torch.backends import cudnn

import numpy as np


def set_seed(seed: Optional[int] = None) -> int:
    """Control CPU and GPU random number generation reproducibility by setting a consistent random seed throughout
    the project and forcing deterministic implementation of algorithms using the CUDNN backend.

    Parameters
    ----------
    seed: int
        RNG seed value between 0 and 2**32 - 1. If None is used, a random value will be picked and logged.

    Returns
    -------
    seed: int
        Returns the value of the seed in case a random value was picked and needs to be recorded.

    Note
    ----
    In future versions, verify that the seed is recorded in the results directory and/or always set via the YAML.

    Examples
    --------
    Set random seed across libraries:

        >>> set_seed(314)

    """
    if seed is None:
        seed = np.random.choice(2**32 - 1)

    if seed < 0:
        raise ValueError("Seed needs to be a positive number between 0 and 2**32 - 1.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    return seed

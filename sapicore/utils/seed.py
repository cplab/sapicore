""" Random number generation and reproducibility. """
import random
import numpy as np

import torch
from torch.backends import cudnn


def fix_random_seed(seed: int = None) -> int:
    """Controls random number generation reproducibility by setting a consistent seed throughout
    the project and forcing deterministic implementation of algorithms using the CUDNN backend.

    Parameters
    ----------
    seed: int, optional
        RNG seed value between 0 and 2**32 - 1. If `None`, a random value will be picked.

    Returns
    -------
    seed: int
        Returns the value of the seed in case a random value was picked and needs to be recorded.

    Examples
    --------
    Set random seed across libraries:

        >>> fix_random_seed(314)

    """
    if seed is None:
        seed = np.random.choice(2**32 - 1)

    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Seed needs to be a positive number between 0 and {2**32 - 1}.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    return seed

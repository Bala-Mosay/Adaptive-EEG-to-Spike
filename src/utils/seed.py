"""Deterministic seeding for NumPy, Python, and PyTorch."""
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed all RNGs used by the pipeline.

    Parameters
    ----------
    seed : int
        Master seed.
    deterministic : bool
        If True, request deterministic cuDNN kernels.  This can reduce
        throughput but makes repeated runs comparable.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

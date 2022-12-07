import random

import numpy as np
import torch
import torch.backends
import torch.backends.cudnn


def seed_all_randomness(
    seed: int, cuda_deterministic: bool, cudnn_deterministic: bool
) -> None:
    """Seed all randomness with the given seed.

    Args:
    ----
        seed (int) : Seed to use for the random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic

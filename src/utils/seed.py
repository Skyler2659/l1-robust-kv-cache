"""Random seed management for reproducibility."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set random seed for Python, NumPy, PyTorch (CPU + CUDA)."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

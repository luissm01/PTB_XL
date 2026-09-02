"""Small reproducibility and device-selection utilities."""

import random

import numpy as np
import torch


MAX_NUMPY_SEED = 2**32 - 1


def seed_random_generators(seed: int) -> None:
    """Reset Python, NumPy, PyTorch and CUDA random-number generators."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= MAX_NUMPY_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_NUMPY_SEED}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve an explicit CPU/CUDA request without silently changing it."""
    if not isinstance(requested, str):
        raise TypeError("requested device must be a string")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    raise ValueError("requested device must be 'auto', 'cpu', or 'cuda'")

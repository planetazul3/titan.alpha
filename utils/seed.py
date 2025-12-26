"""
Random seed utilities for reproducible experiments.

This module provides utilities to set random seeds across all major
libraries (Python, NumPy, PyTorch) to ensure reproducible results.

Note:
    Perfect reproducibility is not guaranteed across different hardware,
    OS versions, or when using multi-threaded data loading. This is a
    best-effort approach for deterministic behavior.

Example:
    >>> from utils.seed import set_global_seed
    >>> set_global_seed(42)  # Set seed for all libraries
"""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_global_seed(seed: int | None = None) -> None:
    """
    Set global random seed for reproducibility across all libraries.

    This function sets the random seed for:
    - Python's built-in random module
    - NumPy's random number generator
    - PyTorch (CPU and CUDA)
    - Python's hash seed (via PYTHONHASHSEED)
    - cuDNN (made deterministic, benchmark disabled)

    Args:
        seed: Integer seed value. If None, uses config DEFAULT_SEED (42).
              Should be non-negative for deterministic behavior.

    Raises:
        ValueError: If seed is negative
        TypeError: If seed is not an integer

    Note:
        - Reproducibility is not guaranteed across different hardware/OS
        - Multi-threaded data loading may introduce non-determinism
        - Disabling cuDNN benchmark may reduce performance
        - This should be called before any model initialization

    Example:
        >>> set_global_seed(42)
        >>> # All subsequent random operations will be deterministic
        >>> import numpy as np
        >>> np.random.rand(3)  # Will always produce same values
    """
    if seed is None:
        from config.constants import DEFAULT_SEED

        seed = DEFAULT_SEED
        logger.info(f"No seed provided, using default: {DEFAULT_SEED}")

    if not isinstance(seed, int):
        raise TypeError(f"Seed must be an integer, got {type(seed).__name__}")

    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    logger.info(f"Setting global seed to {seed}")

    # Set seeds for all libraries
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set CUDA seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        logger.debug("CUDA seed set for all devices")

    # Set Python hash seed for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Make cuDNN deterministic (may reduce performance)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug("cuDNN set to deterministic mode")

    logger.info(f"Global seed {seed} set successfully")


def get_random_state() -> dict:
    """
    Capture the current random state of all libraries.

    Returns:
        Dictionary containing random states for Python, NumPy, and PyTorch

    Example:
        >>> state = get_random_state()
        >>> # Do some random operations
        >>> restore_random_state(state)  # Restore to previous state
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_random_state(state: dict) -> None:
    """
    Restore random state from a previously captured state.

    Args:
        state: Dictionary returned by get_random_state()

    Example:
        >>> state = get_random_state()
        >>> # Do some random operations
        >>> restore_random_state(state)
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    logger.debug("Random state restored")

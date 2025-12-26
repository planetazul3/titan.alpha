"""
DataLoader creation utilities for PyTorch training.

This module provides factory functions to create properly configured
DataLoader instances for training and validation, with appropriate
multi-processing, memory pinning, and worker initialization.

Functions:
    - worker_init_fn: Initialize DataLoader workers with unique seeds
    - create_dataloaders: Create train/val DataLoaders from data paths

Example:
    >>> from data.loader import create_dataloaders
    >>> from pathlib import Path
    >>> train_loader, val_loader = create_dataloaders(
    ...     Path('data/train'), Path('data/val'), settings
    ... )
"""

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.settings import Settings
from data.dataset import DerivDataset

logger = logging.getLogger(__name__)


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize each DataLoader worker with deterministic but different seeds.

    This ensures that each worker has a unique random state while maintaining
    reproducibility across runs. Critical for multi-process data loading.

    Args:
        worker_id: Worker ID assigned by DataLoader

    Note:
        This is called automatically by PyTorch DataLoader for each worker.
        Do not call manually.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    logger.debug(f"Initialized worker {worker_id} with seed {worker_seed}")


def create_dataloaders(
    train_data: Path, val_data: Path, settings: Settings, num_workers: int | None = None
) -> tuple[DataLoader, DataLoader]:
    """
    Create configured DataLoaders for training and validation.

    Automatically configures:
    - Number of workers (auto-detect up to 4)
    - Pin memory (enabled if CUDA available)
    - Worker initialization for reproducibility
    - Shuffling (train only)
    - Batch dropping (train only)

    Args:
        train_data: Path to training data directory (with Parquet files)
        val_data: Path to validation data directory (with Parquet files)
        settings: Configuration settings
        num_workers: Number of data loading workers. If None, auto-detect (max 4)

    Returns:
        Tuple of (train_loader, val_loader)

    Raises:
        FileNotFoundError: If data directories don't exist or lack data files
        ValueError: If settings are invalid

    Example:
        >>> from pathlib import Path
        >>> train_loader, val_loader = create_dataloaders(
        ...     Path('data_cache/train'),
        ...     Path('data_cache/val'),
        ...     settings
        ... )
        >>> for batch in train_loader:
        ...     # Training loop
        ...     pass
    """
    # Validate inputs
    train_data = Path(train_data)
    val_data = Path(val_data)

    if not train_data.exists():
        raise FileNotFoundError(f"Training data directory not found: {train_data}")
    if not val_data.exists():
        raise FileNotFoundError(f"Validation data directory not found: {val_data}")

    # Auto-detect workers (avoid too many on small machines)
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = min(cpu_count, 4)
        logger.info(f"Auto-detected {num_workers} workers (CPU count: {cpu_count})")
    else:
        if num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {num_workers}")
        logger.info(f"Using {num_workers} workers (user-specified)")

    # Pin memory if CUDA available for faster transfers
    pin_memory = torch.cuda.is_available()
    if pin_memory:
        logger.info("Enabling pin_memory for CUDA")

    # Create datasets
    logger.info(f"Creating training dataset from {train_data}")
    train_ds = DerivDataset(train_data, settings, mode="train")

    logger.info(f"Creating validation dataset from {val_data}")
    val_ds = DerivDataset(val_data, settings, mode="eval")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=settings.hyperparams.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        drop_last=True,  # Drop incomplete batches for consistent batch size
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=settings.hyperparams.batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        drop_last=False,  # Use all validation data
        persistent_workers=num_workers > 0,
    )

    logger.info(
        f"Created DataLoaders: train={len(train_loader)} batches, val={len(val_loader)} batches"
    )

    return train_loader, val_loader

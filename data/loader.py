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

def calculate_minimum_temporal_gap(settings: Settings, lookahead_candles: int = 1) -> int:
    """
    Calculate the minimum gap required between training and validation sets 
    to prevent data leakage via sliding windows and lookahead labels.
    
    Formula: Gap = SequenceLength + Warmup + Lookahead + (SequenceLength * SafetyFactor)
    SafetyFactor of 1.0 (another seq_len) ensures absolutely no overlap of input windows.
    
    Args:
        settings: Settings object with data_shapes
        lookahead_candles: Number of candles the label looks ahead
        
    Returns:
        Minimum gap in candles
    """
    seq_len = settings.data_shapes.sequence_length_candles
    warmup = settings.data_shapes.warmup_steps
    
    # Gap = (Input Window) + (Warmup) + (Label Lookahead) + (Safety Margin)
    # Input Window: The features for time T depend on [T-seq_len, T]
    # If Val[0] is at T_val, it sees [T_val-seq_len, T_val]
    # Train[last] is at T_train.
    # We need T_val-seq_len > T_train + lookahead (since Train label looked at T_train+lookahead)
    # Actually simpler:
    # Max Train Time involved = T_train + lookahead (Target)
    # Min Val Time involved = T_val - seq_len - warmup (Features)
    # We need Min Val > Max Train
    # T_val - seq_len - warmup > T_train + lookahead
    # T_val > T_train + seq_len + warmup + lookahead
    # So Gap = seq_len + warmup + lookahead.
    
    # Adding an extra safety factor of 1.0 * seq_len just to be paranoid about window strides
    safety_margin = int(seq_len * 1.0)
    
    gap = seq_len + warmup + lookahead_candles + safety_margin
    return gap



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

    # Audit Fix (Issue 2): Validate Train/Val Split Consistency
    from utils.data_validation import validate_split_consistency
    try:
        # Enforce strict forward validation (Train < Val)
        validate_split_consistency(train_data, val_data, strict_forward=True)
    except ValueError as e:
        logger.error(f"Data Split Validation Failed: {e}")
        # We raise error to prevent training on leaked data
        raise

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

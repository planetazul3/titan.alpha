
"""
Data validation utilities for the Deriv trading system.

Focuses on preventing data leakage and ensuring dataset integrity.
"""

import logging
from pathlib import Path
from typing import NamedTuple, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logger = logging.getLogger(__name__)

class TimeRange(NamedTuple):
    min_epoch: float
    max_epoch: float

def get_parquet_time_range(path: Path) -> Optional[TimeRange]:
    """
    Get the (min, max) epoch from a parquet file or directory of parquet files.
    Returns None if no data found.
    """
    if not HAS_PANDAS:
        logger.warning("Pandas not available, skipping strict data validation.")
        return None

    if not path.exists():
        return None

    files = []
    if path.is_file():
        if path.suffix == '.parquet':
            files = [path]
    elif path.is_dir():
        # Recursive search for parquet files
        files = sorted(list(path.glob("**/*.parquet")))
    
    if not files:
        return None

    min_epoch = float('inf')
    max_epoch = float('-inf')

    # Optimization: If using pyarrow, we could read metadata only. 
    # With pandas read_parquet, we might need to be careful not to load everything.
    # However, 'epoch' column is small.
    
    found_data = False
    
    for f in files:
        try:
            # Attempt to read just the epoch column if supported by engine, 
            # otherwise read minimal data?
            # fastparquet/pyarrow usually supports columns kwarg.
            df = pd.read_parquet(f, columns=["epoch"])
            
            if df.empty:
                continue
                
            file_min = df["epoch"].min()
            file_max = df["epoch"].max()
            
            if file_min < min_epoch:
                min_epoch = file_min
            if file_max > max_epoch:
                max_epoch = file_max
            
            found_data = True
            
        except Exception as e:
            logger.warning(f"Failed to validate file {f}: {e}")
            continue

    if not found_data:
        return None
        
    return TimeRange(min_epoch, max_epoch)

def validate_split_consistency(train_path: Path, val_path: Path, strict_forward: bool = True) -> bool:
    """
    Validate that training and validation datasets are temporally consistent.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        strict_forward: If True, enforces train_max < val_min. 
                       If False, only enforces no overlap (disjoint).
                       
    Returns:
        True if valid.
        
    Raises:
        ValueError: If data leakage detected or splits invalid.
    """
    if not HAS_PANDAS:
        return True

    train_range = get_parquet_time_range(train_path)
    val_range = get_parquet_time_range(val_path)

    if not train_range:
        logger.warning(f"No training data found at {train_path} to validate.")
        return True # Can't validate empty
        
    if not val_range:
        logger.warning(f"No validation data found at {val_path} to validate.")
        return True # Can't validate empty

    logger.info(f"Data Split Check: Train[{train_range.min_epoch}..{train_range.max_epoch}] vs Val[{val_range.min_epoch}..{val_range.max_epoch}]")

    # Check 1: Strict Forward (Train comes before Val)
    if train_range.max_epoch >= val_range.min_epoch:
        msg = (
            f"Possible Data Leakage! Training data ends ({train_range.max_epoch}) "
            f"AFTER validation data starts ({val_range.min_epoch})."
        )
        
        if strict_forward:
            raise ValueError(f"Strict Forward Validation Failed: {msg}")
        else:
            # Check 2: Disjoint (Overlap detection)
            # Two ranges overlap if max1 >= min2 AND max2 >= min1
            overlap = (train_range.max_epoch >= val_range.min_epoch) and (val_range.max_epoch >= train_range.min_epoch)
            
            if overlap:
                raise ValueError(f"Disjoint Validation Failed: Datasets overlap in time! {msg}")
            else:
                logger.warning(f"Validation Warning: {msg} (Allowed because strict_forward=False, but effectively disjoint)")

    logger.info("Train/Val split validation passed.")
    return True

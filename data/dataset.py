"""
PyTorch Dataset for Deriv trading data.

Supports:
- Loading from Parquet files (for training)
- Lazy loading with windowing
- Label generation from price movements
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from config.settings import Settings
from data.features import FeatureBuilder

logger = logging.getLogger(__name__)

# Optional pandas for Parquet support
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class DerivDataset(Dataset):
    """
    CANONICAL DATA PATH - The ONLY way to load training data.

    This dataset enforces a single, validated data loading path to ensure:
    - Consistent feature engineering across all runs
    - Reproducible preprocessing guarantees
    - Validated shape contracts
    - No ambiguous data versions

    IMPORTANT: All model training MUST go through this class. Do NOT:
    - Load Parquet files directly in training scripts
    - Create ad-hoc data loading functions
    - Bypass the validation checks

    The _load_data() method will:
    - Reject if no data files found (explicit download required)
    - Reject ambiguous cache (multiple versions of same data type)
    - Validate required columns exist
    - Ensure deterministic file ordering

    Usage:
        dataset = DerivDataset(Path('data_cache'), settings, mode='train')
        sample = dataset[0]  # Returns dict with tensors

    Shape Contract:
        Returns dict with:
        - 'ticks': (seq_len_ticks,) float32
        - 'candles': (seq_len_candles, 10) float32
        - 'vol_metrics': (4,) float32
        - 'targets': Dict with 'rise_fall', 'touch', 'range' labels
        - 'ticks_mask': (seq_len_ticks,) float32 (1=valid, 0=padding)
        - 'candles_mask': (seq_len_candles,) float32 (1=valid, 0=padding)
    """

    def __init__(
        self,
        data_source: Path,
        settings: Settings,
        mode: Literal["train", "eval"] = "train",
        lookahead_candles: int = 5,
    ):
        """
        Args:
            data_source: Path to directory with Parquet files
            settings: Configuration settings
            mode: 'train' for training, 'eval' for evaluation
            lookahead_candles: Number of candles ahead for label generation
        """
        super().__init__()

        self.settings = settings
        self.mode = mode
        self.lookahead = lookahead_candles

        self.tick_len = settings.data_shapes.sequence_length_ticks
        self.candle_len = settings.data_shapes.sequence_length_candles
        self.warmup_steps = settings.data_shapes.warmup_steps
        
        # Audit Fix: Dynamic Label Thresholds
        self.threshold_touch = settings.data_shapes.label_threshold_touch
        self.threshold_range = settings.data_shapes.label_threshold_range

        # CANONICAL feature pipeline - single source of truth
        self.feature_builder = FeatureBuilder(settings)

        # Load data
        self._load_data(data_source)

        # Calculate number of samples
        self._calculate_indices()

    def _get_cache_path(self, data_source: Path) -> Path:
        """Get or create cache directory."""
        cache_dir = data_source / ".cache"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir

    def _compute_hash(self, files: list[Path]) -> str:
        """Compute hash of file states (names + mtimes + sizes) + config."""
        hasher = hashlib.md5()
        
        # Audit Fix: Include configuration parameters in hash
        # If lookahead, seq_len, or thresholds change, cache must be rebuilt
        config_str = (
            f"lookahead={self.lookahead}:"
            f"tick_len={self.tick_len}:"
            f"candle_len={self.candle_len}:"
            f"warmup={self.warmup_steps}:"
            f"touch={self.threshold_touch}:"
            f"range={self.threshold_range}"
        )
        hasher.update(config_str.encode())
        
        for f in sorted(files):
            stat = f.stat()
            # Include name, size, time
            hasher.update(f"{f.name}:{stat.st_size}:{stat.st_mtime}".encode())
        return hasher.hexdigest()

    def _load_or_create_mmap(
        self, 
        name: str, 
        files: list[Path], 
        cache_dir: Path,
        loader_func: Any,
        dtype: Any
    ) -> np.ndarray:
        """
        Load data from memory-mapped file, creating it if needed.
        
        Args:
            name: Identifier for the data (e.g., 'ticks', 'candles')
            files: List of source files
            cache_dir: Directory to store cache
            loader_func: Function that returns numpy array if cache miss
            dtype: Expected data type
            
        Returns:
            Memory-mapped numpy array
        """
        # compute hash of source files
        source_hash = self._compute_hash(files)
        cache_file = cache_dir / f"{name}_{source_hash}.npy"
        
        if cache_file.exists():
            logger.info(f"Loading cached {name} from {cache_file}")
            return np.load(cache_file, mmap_mode="r")
            
        logger.info(f"Cache miss for {name}. Loading from source and creating cache...")
        data = loader_func(files)
        
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype)
        else:
            data = data.astype(dtype)
            
        # Atomic write
        temp_file = cache_file.with_suffix(".tmp.npy")
        np.save(temp_file, data)
        temp_file.rename(cache_file)
        
        logger.info(f"Created cache {cache_file} ({data.nbytes / 1e6:.1f} MB)")
        
        # Reload as mmap
        return np.load(cache_file, mmap_mode="r")

    def _load_data(self, data_source: Path):
        """
        Load tick and candle data using memory mapping.

        CANONICAL DATA PATH ENFORCEMENT:
        This is the ONLY way to load data for model consumption.
        """
        if not HAS_PANDAS:
            raise RuntimeError("pandas required for Parquet loading")

        data_source = Path(data_source)
        cache_dir = self._get_cache_path(data_source)

        # Try partitioned format first (new)
        tick_files, candle_files = self._find_parquet_files(data_source)

        # VALIDATION: Reject if no files found
        if not tick_files:
            raise FileNotFoundError(
                f"No tick files found in {data_source}. "
                f"Run 'python scripts/download_data.py --months N' to download data."
            )
        if not candle_files:
            raise FileNotFoundError(
                f"No candle files found in {data_source}. "
                f"Run 'python scripts/download_data.py --months N' to download data."
            )

        # --- Load Ticks ---
        def load_ticks(files):
            logger.info(f"Loading ticks from {len(files)} file(s)")
            tick_dfs = [pd.read_parquet(f) for f in files]
            # Must sort by epoch
            df = pd.concat(tick_dfs, ignore_index=True).sort_values("epoch").reset_index(drop=True)
            if "quote" not in df.columns:
                raise ValueError("Tick files missing 'quote' column")
            return df["quote"].values

        self.ticks = self._load_or_create_mmap(
            "ticks", tick_files, cache_dir, load_ticks, np.float32
        )

        # --- Load Tick Epochs ---
        def load_tick_epochs(files):
            tick_dfs = [pd.read_parquet(f) for f in files]
            df = pd.concat(tick_dfs, ignore_index=True).sort_values("epoch").reset_index(drop=True)
            return df["epoch"].values

        self.tick_epochs = self._load_or_create_mmap(
            "tick_epochs", tick_files, cache_dir, load_tick_epochs, np.float64
        )

        # --- Load Candles ---
        def load_candles(files):
            logger.info(f"Loading candles from {len(files)} file(s)")
            candle_dfs = [pd.read_parquet(f) for f in files]
            df = pd.concat(candle_dfs, ignore_index=True).sort_values("epoch").reset_index(drop=True)
            
            required_candle_cols = ["open", "high", "low", "close", "epoch"]
            missing = set(required_candle_cols) - set(df.columns)
            if missing:
                raise ValueError(f"Missing candle cols: {missing}")
                
            data = df[required_candle_cols].values.astype(np.float32)
            
            # Add volume (zeros) and reorder to [O,H,L,C,V,T]
            # Current: [O,H,L,C,E] (5 cols)
            # Need: [O,H,L,C,V,E] (6 cols)
            zeros = np.zeros((len(data), 1), dtype=np.float32)
            return np.hstack([data[:, :4], zeros, data[:, 4:5]])

        self.candles = self._load_or_create_mmap(
            "candles", candle_files, cache_dir, load_candles, np.float32
        )

        logger.info(
            f"Dataset mapped: {len(self.ticks)} ticks, {len(self.candles)} candles"
        )

    def _find_parquet_files(self, data_source: Path):
        """
        Find all tick and candle Parquet files, supporting both formats.

        Returns:
            Tuple of (tick_files, candle_files) lists, sorted for determinism
        """
        tick_files = []
        candle_files = []

        # Try partitioned format: {symbol}/ticks/*.parquet
        for symbol_dir in data_source.iterdir():
            if symbol_dir.is_dir():
                tick_dir = symbol_dir / "ticks"
                if tick_dir.exists():
                    tick_files.extend(sorted(tick_dir.glob("*.parquet")))

                # Find candle directories (candles_60, candles_300, etc.)
                for candle_dir in symbol_dir.glob("candles_*"):
                    if candle_dir.is_dir():
                        candle_files.extend(sorted(candle_dir.glob("*.parquet")))

        # Fallback to legacy format if no partitions found
        if not tick_files:
            tick_files = sorted(data_source.glob("*_ticks_*.parquet"))
        if not candle_files:
            candle_files = sorted(data_source.glob("*_candles_*.parquet"))

        return tick_files, candle_files

    def _calculate_indices(self):
        """Calculate valid sample indices based on sequence lengths."""
        # Each candle roughly corresponds to 60 seconds of ticks
        # We need enough history + lookahead for labels

        # We need enough history (warmup + seq) + lookahead for labels

        min_candle_idx = self.candle_len + self.warmup_steps
        max_candle_idx = len(self.candles) - self.lookahead

        if max_candle_idx <= min_candle_idx:
            raise ValueError(
                f"Not enough data: need {min_candle_idx + self.lookahead} candles, "
                f"have {len(self.candles)}"
            )

        self.valid_indices = list(range(min_candle_idx, max_candle_idx))
        logger.info(f"Dataset has {len(self.valid_indices)} valid samples")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get single sample.

        Returns:
            Dict with:
            - 'ticks': (seq_len_ticks,) normalized tick sequence
            - 'candles': (seq_len_candles, 10) normalized candle features
            - 'vol_metrics': (4,) volatility metrics
            - 'targets': Dict with 'rise_fall', 'touch', 'range' labels
        """
        candle_idx = self.valid_indices[idx]

        # Get candle slice with warmup
        # Fetch (warmup + sequence) length
        total_candle_len = self.candle_len + self.warmup_steps
        candle_start = candle_idx - total_candle_len
        candle_end = candle_idx
        # This slice will be (warmup + sequence) length
        candle_slice = self.candles[candle_start:candle_end]

        # Get corresponding tick slice using EXACT timestamp alignment
        # CRITICAL FIX: Use searchsorted to find ticks corresponding to this candle's time
        # Candle format: [open, high, low, close, volume, epoch]
        candle_timestamp = self.candles[candle_end - 1, 5]

        if self.tick_epochs is None:
            # Should be guaranteed by _load_data validation, but satisfying static analysis
            raise ValueError("Tick epochs missing")

        # Find the index in ticks where timestamp <= candle_timestamp
        # searchsorted returns the index where the value should be inserted to maintain order
        # side='right' means tick_end will be the index AFTER the last tick <= timestamp
        tick_end = int(np.searchsorted(self.tick_epochs, candle_timestamp, side="right"))

        # Get appropriate window length (warmup + sequence)
        # Use warmup_steps scaling for ticks (approx 1 tick per sec? No, just use raw steps for now)
        # Actually, using same warmup count for ticks is a reasonable heuristic for z-score stability
        total_tick_len = self.tick_len + self.warmup_steps
        tick_start = max(0, tick_end - total_tick_len)
        tick_slice = self.ticks[tick_start:tick_end]

        # Pad ticks if needed (edge padding with first available value)
        if len(tick_slice) < total_tick_len:
            pad_len = total_tick_len - len(tick_slice)
            # Use edge padding (first available price) to avoid log(0) crash
            # in normalizers.log_returns() which requires positive prices
            if len(tick_slice) > 0:
                padding = np.full(pad_len, tick_slice[0], dtype=np.float32)
            else:
                # Fallback to safe non-zero default (log returns will be 0 anyway)
                padding = np.full(pad_len, 100.0, dtype=np.float32)
            tick_slice = np.concatenate([padding, tick_slice])
            
            # Create tick mask (0 for padding, 1 for real data)
            valid_len = total_tick_len - pad_len
            tick_mask = np.concatenate([np.zeros(pad_len, dtype=np.float32), np.ones(valid_len, dtype=np.float32)])
        else:
            tick_mask = np.ones(len(tick_slice), dtype=np.float32)

        # Candle mask (always full valid due to index logic, but consistent)
        candle_mask = np.ones(len(candle_slice), dtype=np.float32)

        # CANONICAL feature processing via FeatureBuilder
        features = self.feature_builder.build_numpy(
            ticks=tick_slice,
            candles=candle_slice,
            validate=False,  # Allow partial data during dataset iteration
        )
        
        # Trim masks to match output feature length (remove warmup)
        tick_mask = tick_mask[-self.tick_len:]
        candle_mask = candle_mask[-self.candle_len:]

        # Generate labels from future movement
        targets = self._generate_labels(candle_idx)

        return {
            "ticks": torch.from_numpy(features["ticks"]),
            "candles": torch.from_numpy(features["candles"]),
            "vol_metrics": torch.from_numpy(features["vol_metrics"]),
            "targets": targets,
            "ticks_mask": torch.from_numpy(tick_mask),
            "candles_mask": torch.from_numpy(candle_mask),
        }

    def _generate_labels(self, candle_idx: int) -> dict[str, torch.Tensor]:
        """
        Generate labels from future price movement.

        Labels:
        - rise_fall: 1 if close[future] > close[current], else 0
        - touch: 1 if max(high[future...]) > threshold, else 0
        - range: 1 if price stays within range, else 0
        """
        current_close = self.candles[candle_idx - 1, 3]  # Last candle's close
        future_candles = self.candles[candle_idx : candle_idx + self.lookahead]

        if len(future_candles) == 0:
            return {
                "rise_fall": torch.tensor(0, dtype=torch.float32),
                "touch": torch.tensor(0, dtype=torch.float32),
                "range": torch.tensor(0, dtype=torch.float32),
            }

        # Rise/Fall: Direction of price after lookahead period
        future_close = future_candles[-1, 3]
        rise_fall = 1.0 if future_close > current_close else 0.0

        # Guard against division by zero in edge cases (e.g., synthetic test data)
        if current_close <= 0:
            current_close = 1e-10

        # Touch: Did price touch a barrier (simplified: moved > 0.5%)
        future_high = np.max(future_candles[:, 1])
        future_low = np.min(future_candles[:, 2])
        price_range = (future_high - future_low) / current_close
        touch = 1.0 if price_range > self.threshold_touch else 0.0


        # Range: Did price stay within 0.3% band?
        max_deviation = (
            max(abs(future_high - current_close), abs(future_low - current_close)) / current_close
        )
        stays_in_range = 1.0 if max_deviation < self.threshold_range else 0.0

        return {
            "rise_fall": torch.tensor(rise_fall, dtype=torch.float32),
            "touch": torch.tensor(touch, dtype=torch.float32),
            "range": torch.tensor(stays_in_range, dtype=torch.float32),
        }

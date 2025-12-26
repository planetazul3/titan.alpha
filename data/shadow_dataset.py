
import json
import logging
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config.settings import Settings
from data.features import FeatureBuilder

logger = logging.getLogger(__name__)

class ShadowTradeDataset(Dataset):
    """
    Dataset for training on discrete Shadow Trade episodes.
    
    Unlike DerivDataset which uses continuous time-series, this dataset
    loads independent 'episodes' (ticks + candles context) captured during live trading.
    
    This enables "Experience Replay" - retraining on specific real-world scenarios
    including those where the model was uncertain or wrong.
    """
    
    def __init__(
        self, 
        parquet_path: Path, 
        settings: Settings,
        mode: Literal["train", "eval"] = "train",
        only_resolved: bool = True
    ):
        """
        Args:
            parquet_path: Path to exported shadow trades parquet file
            settings: Configuration settings
            mode: 'train' or 'eval'
            only_resolved: If True, only load trades with a determined outcome
        """
        self.settings = settings
        self.feature_builder = FeatureBuilder(settings)
        
        if not Path(parquet_path).exists():
            raise FileNotFoundError(f"Shadow data not found: {parquet_path}")
            
        logger.info(f"Loading shadow data from {parquet_path}")
        self.df = pd.read_parquet(parquet_path)
        
        if only_resolved:
            # outcome is 1 (True), 0 (False), or NaN/None
            # We want rows where outcome is not null
            initial_len = len(self.df)
            self.df = self.df[self.df["outcome"].notna()].reset_index(drop=True)
            logger.info(f"Filtered to {len(self.df)} resolved trades (from {initial_len})")
            
        # Parse tick/candle windows from JSON strings if necessary
        # The export_parquet in SQLiteStore saves them as strings?
        # Let's check the schema. Yes, json.dumps in store.
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        
        # 1. Reconstruct Context
        # Windows are stored as JSON strings in the parquet export
        # We need to parse them back to lists/arrays
        
        try:
            tick_window = np.array(json.loads(row["tick_window"]), dtype=np.float32)
            candle_window = np.array(json.loads(row["candle_window"]), dtype=np.float32)
        except (TypeError, json.JSONDecodeError):
            # Safe fallback if already parsed or format varies (e.g. if pandas auto-converted)
            # But likely they are strings.
            tick_window = np.zeros(self.settings.data_shapes.sequence_length_ticks, dtype=np.float32)
            candle_window = np.zeros((self.settings.data_shapes.sequence_length_candles, 6), dtype=np.float32)
            logger.warning(f"Failed to parse windows for trade {row.get('trade_id')}")

        # 2. Feature Engineering
        # We use the feature builder to re-normalize and extract metrics just like in live inference
        features = self.feature_builder.build_numpy(
            ticks=tick_window,
            candles=candle_window,
            validate=False
        )
        
        # 3. Targets
        # Shadow trades have a specific 'outcome' (Win/Loss) for a specific contract type
        # We need to map this back to model targets (Rise/Fall, Touch, etc.)
        
        # Default all to 0
        targets = {
            "rise_fall": 0.0,
            "touch": 0.0,
            "range": 0.0
        }
        
        contract_type = row["contract_type"]
        direction = row["direction"]
        outcome = bool(row["outcome"])
        
        # Reverse-engineer the ground truth label
        # If 'CALL' won (outcome=True), then Rise/Fall target should be 1.
        # If 'CALL' lost (outcome=False), then Rise/Fall target should be 0.
        # If 'PUT' won (outcome=True), then Rise/Fall target should be 0 (Price fell).
        # If 'PUT' lost (outcome=False), then Rise/Fall target should be 1 (Price rose).
        
        rise_fall_label = 0.0
        if "RISE_FALL" in contract_type:
            if direction == "CALL":
                rise_fall_label = 1.0 if outcome else 0.0
            elif direction == "PUT":
                rise_fall_label = 0.0 if outcome else 1.0
            targets["rise_fall"] = rise_fall_label
            
        # Logic for Touch/Range contracts
        if "TOUCH" in contract_type:
            # TOUCH/NO_TOUCH: Check if barriers were hit
            # We need high/low prices from candle window to determine this
            if len(features["candles"]) > 0:
                high_prices = features["candles"][:, 1]  # Column 1 = High
                low_prices = features["candles"][:, 2]   # Column 2 = Low
                entry_price = float(row.get("entry_price", 0.0))
                
                # Use same barriers as in ShadowResolution (0.5% default)
                barrier_pct = 0.005
                upper_barrier = entry_price * (1 + barrier_pct)
                lower_barrier = entry_price * (1 - barrier_pct)
                
                touched = bool(np.any(high_prices >= upper_barrier) or 
                               np.any(low_prices <= lower_barrier))
                
                # Check ground truth vs touched status
                if direction == "TOUCH":
                    # If outcome=True, it means it DID touch.
                    # If outcome=False, it means it did NOT touch.
                    # Target is "Will it touch?" -> 1 if yes, 0 if no.
                    targets["touch"] = 1.0 if outcome else 0.0
                elif direction == "NO_TOUCH":
                    # If outcome=True, it means it did NOT touch.
                    # If outcome=False, it means it DID touch.
                    # Target is "Will it touch?" -> 0 if it didn't touch (NT won), 1 if it did touch (NT lost)
                    targets["touch"] = 0.0 if outcome else 1.0
                    
        elif "STAYS_BETWEEN" in contract_type:
             # STAYS_BETWEEN: Check if range was held
             if len(features["candles"]) > 0:
                # STAYS_BETWEEN target: 1 = Stays inside, 0 = Goes outside
                targets["range"] = 1.0 if outcome else 0.0
        
        # 4. Return Tensors
        return {
            "ticks": torch.from_numpy(features["ticks"]),
            "candles": torch.from_numpy(features["candles"]),
            "vol_metrics": torch.from_numpy(features["vol_metrics"]),
            # Dataset Protocol expects 'targets' to be a Tensor or Dict[str, Tensor]
            # Here we follow the Dict[str, Tensor] pattern
            "targets": cast(dict[str, torch.Tensor], {k: torch.tensor(v, dtype=torch.float32) for k, v in targets.items()})
        }

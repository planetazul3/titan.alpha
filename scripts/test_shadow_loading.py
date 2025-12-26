
import sys
from pathlib import Path
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import load_settings
from data.shadow_dataset import ShadowTradeDataset

def test_loading():
    settings = load_settings()
    parquet_path = Path("data_cache/shadow_replay.parquet")
    
    if not parquet_path.exists():
        print("Skipping test: parquet file not found")
        return

    print("Inspect Parquet Schema...")
    df = pd.read_parquet(parquet_path)
    print("Columns:", df.columns.tolist())
    print("First row:", df.iloc[0].to_dict())

    print("\nInitializing ShadowTradeDataset...")
    dataset = ShadowTradeDataset(parquet_path, settings, only_resolved=False) # Load unresolved too for test if needed
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print("Sample keys:", sample.keys())
        print("Ticks shape:", sample['ticks'].shape)
        print("Candles shape:", sample['candles'].shape)
        print("Targets:", sample['targets'])
        print("✅ SUCCESS: Shadow data loaded into tensors!")
    else:
        print("⚠️ Dataset empty (no trades yet?)")

if __name__ == "__main__":
    test_loading()

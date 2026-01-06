
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def generate_dummy_data(output_path: str = "data_cache/dummy_backtest.parquet", n_candles: int = 1000):
    start_time = datetime.now() - timedelta(minutes=n_candles)
    dates = [start_time + timedelta(minutes=i) for i in range(n_candles)]
    
    # Generate random walk price
    price = 100.0
    prices: list[float] = []
    
    data = []
    for d in dates:
        change = np.random.normal(0, 0.1)
        price += change
        
        # OHLC
        open_p = price
        close_p = price + np.random.normal(0, 0.05)
        high_p = max(open_p, close_p) + abs(np.random.normal(0, 0.02))
        low_p = min(open_p, close_p) - abs(np.random.normal(0, 0.02))
        
        data.append({
            "timestamp": d, # pd.Timestamp
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": abs(np.random.normal(100, 50)),
            "epoch": int(d.timestamp())
        })
        
    df = pd.DataFrame(data)
    
    # Convert timestamp to UTC aware if needed, but parquet handles it
    # Ensure directory exists
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    
    df.to_parquet(output_path)
    print(f"Generated {len(df)} candles to {output_path}")

if __name__ == "__main__":
    generate_dummy_data()

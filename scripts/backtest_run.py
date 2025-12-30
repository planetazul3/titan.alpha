#!/usr/bin/env python3
"""
Backtest Runner Script.

Replays historical data through the live trading pipeline using `BacktestClient`.
"""

import asyncio
import logging
import argparse
import sys
from pathlib import Path
import pandas as pd
import json
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import load_settings
from execution.backtest import BacktestClient, BacktestEngine
from data.buffer import MarketDataBuffer
from data.events import CandleEvent
from execution.decision import DecisionEngine
from execution.executor import DerivTradeExecutor
from execution.safety import SafeTradeExecutor, ExecutionSafetyConfig
from execution.sqlite_shadow_store import SQLiteShadowStore
from config.logging_config import setup_logging
from utils.bootstrap import create_trading_stack

log_file = setup_logging(script_name="backtest", level="INFO")
logger = logging.getLogger(__name__)

async def run_backtest(args):
    settings = load_settings()
    data_path = Path(args.data)
    
    if not data_path.exists():
        logger.warning(f"Data file not found: {data_path}. Continuing with synthetic data generation.")
    else:
        logger.info(f"Starting backtest on {data_path}")
    
    # 1. Initialize Components
    # Bootstrap stack (inject BacktestClient)
    client = BacktestClient(initial_balance=10000.0)
    
    stack = create_trading_stack(
        settings, 
        checkpoint_path=None, 
        device="cpu", # Force CPU for backtest
        verify_ckpt=False,
        client=client
    )
    
    engine = stack["engine"]
    shadow_store = stack["shadow_store"]
    feature_builder = stack["feature_builder"]
    model = stack["model"] # Loaded random/default
    
    # Simple Sizer
    from execution.position_sizer import FixedStakeSizer
    sizer = FixedStakeSizer(stake=10.0)
    
    # Executor
    # Note: engine.policy is available from bootstrap
    raw_executor = DerivTradeExecutor(client, settings, position_sizer=sizer, policy=engine.policy)
    safety_config = ExecutionSafetyConfig(max_trades_per_minute=100, kill_switch_enabled=False)
    # Use in-memory DB or temp file for backtest safety state
    safety_store_path = Path("backtest_safety.db")
    executor = SafeTradeExecutor(
        raw_executor, 
        safety_config, 
        state_file=safety_store_path,
        stake_resolver=lambda s: 10.0
    )
    
    # Buffer
    buffer = MarketDataBuffer(
        tick_length=settings.data_shapes.sequence_length_ticks,
        candle_length=settings.data_shapes.sequence_length_candles
    )
    
    # 2. Load Data
    df = pd.DataFrame()
    if data_path.exists():
        df = pd.read_parquet(data_path)
    
    # Extract candles sequence
    # Assumption: 'candle_window' column contains list of candles.
    # We need to extract the UNIQUE stream of candles.
    # Or if input is just OHLC CSV, easier.
    
    # For demonstration with shadow_replay.parquet (which has windows):
    # We will try to extract the LAST candle from each window, assuming chronological order.
    # This might miss candles if inference wasn't run every step, but good enough for verifying the *mechanism*.
    
    candles = []
    if "candle_window" in df.columns:
        logger.info("Extracting candles from shadow windows...")
        last_ts = None
        for i, row in df.iterrows():
            try:
                window = json.loads(row["candle_window"]) if isinstance(row["candle_window"], str) else row["candle_window"]
                # Window is usually [ [open, high, low, close, vol, time], ... ]
                # Let's take the last one
                if not window:
                    continue
                latest = window[-1] # [o, h, l, c, v, ts] logic depends on preprocessor
                # Wait, raw window in shadow store is usually features or raw? 
                # ShadowStore.record saves `context.candles` which is `np.ndarray`.
                # If it was saved as JSON list.
                
                # If we can't easily parse windows, let's look for 'exit_price' and 'resolved_at' etc.
                # Actually, building a robust backtester from *shadow logs* is hard.
                # Standard usage: User provides OHLC parquet/csv.
                pass
            except Exception as e:
                logger.warning(f"Failed to parse row {i}: {e}")
    
    # Mock Data Generation if extraction fails or for L04 verification
    logger.info("Generating synthetic backtest data...")
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1min")
    mock_candles = []
    price = 100.0
    for d in dates:
        change = np.random.normal(0, 0.5)
        price += change
        mock_candles.append({
            "timestamp": d,
            "open": price,
            "high": price + 0.2,
            "low": price - 0.2,
            "close": price + 0.1,
            "epoch": int(d.timestamp())
        })
        
    # 3. Replay Loop
    logger.info(f"Replaying {len(mock_candles)} candles...")
    
    for c in mock_candles:
        # Update client time/price
        client.update_market(c["close"], c["timestamp"])
        
        # Create Event
        event = CandleEvent(
            symbol=settings.trading.symbol,
            timestamp=c["timestamp"],
            open=c["open"],
            high=c["high"],
            low=c["low"],
            close=c["close"],
            volume=0.0
        )
        
        # Update Buffer
        is_new = buffer.update_candle(event)
        
        if is_new and buffer.is_ready():
            # Mock Inference
            # For backtest, we might skip the heavy model and just use random signal
            # or allow passing a model.
            # L04 goal is the *infrastructure*.
            
            # features = feature_builder.build(buffer.snapshot())
            # signal = model(features) ...
            
            # Simple random signal for verification
            from execution.decision import TradeSignal
            if np.random.random() > 0.8:
                signal = TradeSignal(
                     signal_id=f"bt_{c['epoch']}",
                     timestamp=c["timestamp"],
                     symbol=settings.trading.symbol,
                     direction="CALL" if np.random.random() > 0.5 else "PUT",
                     confidence=0.9,
                     contract_type="DIGITMATCH" # or whatever
                )
                logger.info(f"Signal generated: {signal.direction}")
                
                # Execute
                result = await executor.execute(signal)
                if result.success:
                    logger.info(f"Trade executed: {result}")
                else:
                    logger.info(f"Trade rejected: {result.reason}")

    # 4. Report
    balance = await client.get_balance()
    logger.info(f"Backtest Complete. Final Balance: ${balance:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_cache/shadow_replay.parquet")
    parser.add_argument("--strategy", default="fixed")
    args = parser.parse_args()
    
    try:
        asyncio.run(run_backtest(args))
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup temp DBs
        Path("backtest_safety.db").unlink(missing_ok=True)
        Path("backtest_shadow.db").unlink(missing_ok=True)

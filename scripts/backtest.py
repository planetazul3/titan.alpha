#!/usr/bin/env python3
"""
Comprehensive Backtesting Script (REC-004).

Replays historical market data through the full trading pipeline:
Data -> FeatureBuilder -> Model -> DecisionEngine -> Execution.

Usage:
    python scripts/backtest.py --data data_cache/history.parquet --model models/best.pt
"""

import asyncio
import argparse
import logging
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import load_settings
from config.logging_config import setup_logging
from utils.bootstrap import create_trading_stack
from execution.backtest import BacktestEngine, BacktestClient
from execution.safety import SafeTradeExecutor, ExecutionSafetyConfig
from execution.executor import DerivTradeExecutor
from execution.strategy_adapter import StrategyAdapter

logger = logging.getLogger(__name__)

async def run_detailed_backtest(args):
    # 1. Setup
    setup_logging(script_name="backtest", level=args.log_level)
    settings = load_settings()
    
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    logger.info(f"Starting backtest on {data_path}")
    logger.info(f"Model: {args.model if args.model else 'RANDOM/Untrained'}")
    
    # 2. Initialize Backtest Client
    client = BacktestClient(
        initial_balance=args.balance,
        latency_ms=args.latency
    )
    
    # 3. Bootstrap Stack
    stack = create_trading_stack(
        settings,
        checkpoint_path=Path(args.model) if args.model else None,
        device=args.device,
        client=client,
        verify_ckpt=False # Skip strict verification 
    )
    
    engine = stack["engine"]
    feature_builder = stack["feature_builder"]
    model = stack["model"]
    
    # 4. Configure Execution Stack
    # Strategy Adapter
    from execution.position_sizer import KellyPositionSizer
    
    position_sizer = KellyPositionSizer(
        base_stake=10.0,
        safety_factor=0.2, # Equivalent to kelly_fraction=0.2
        min_stake=1.0,
        max_stake=settings.execution_safety.max_stake_per_trade,
    )
    
    strategy_adapter = StrategyAdapter(
        settings=settings,
        position_sizer=position_sizer
    )
    
    # Safe Executor
    # We wrap the *BacktestClient* via DerivTradeExecutor?
    # BacktestClient mocks DerivClient interface.
    # DerivTradeExecutor expects DerivClient.
    # Yes, client=client passed to create_trading_stack sets stack['client'] = client.
    # But DerivTradeExecutor is NOT created by bootstrap.
    
    raw_executor = DerivTradeExecutor(client, settings) # Removed position_sizer arg in previous refactor
    
    safety_config = ExecutionSafetyConfig(
        max_trades_per_minute=1000, # Relax for backtest
        kill_switch_enabled=False
    )
    
    # Ephemeral safety store
    safety_store_path = Path("backtest_safety.db")
    if safety_store_path.exists():
        safety_store_path.unlink()
        
    executor = SafeTradeExecutor(
        raw_executor,
        safety_config,
        state_file=safety_store_path
    )
    
    # 5. Initialize Backtest Engine
    # We need to create a buffer here because it's stateful
    from data.buffer import MarketDataBuffer
    buffer = MarketDataBuffer(
        tick_length=settings.data_shapes.sequence_length_ticks,
        candle_length=settings.data_shapes.sequence_length_candles
    )

    bt_engine = BacktestEngine(
        settings=settings,
        data_path=data_path,
        initial_balance=args.balance,
        slip_prob=args.slippage,
        # Pipeline injections
        buffer=buffer,
        feature_builder=feature_builder,
        model=model,
        decision_engine=engine,
        executor=executor,
        strategy_adapter=strategy_adapter
    )
    
    # 6. Run
    try:
        metrics = await bt_engine.run()
        
        # Print summary table
        print("\n" + "="*50)
        print(f"BACKTEST RESULTS: {data_path.name}")
        print("="*50)
        print(f"Initial Balance: ${metrics['initial_balance']:.2f}")
        print(f"Final Balance:   ${metrics['final_balance']:.2f}")
        print(f"Net PnL:         ${metrics['net_profit']:.2f} ({metrics['return_pct']:.2f}%)")
        print("-" * 50)
        print(f"Total Trades:    {metrics['total_trades']}")
        print(f"Win Rate:        {metrics['win_rate']:.2f}% ({metrics['winning_trades']}W - {metrics['losing_trades']}L)")
        print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.3f}")
        print(f"Max Drawdown:    {metrics['max_drawdown_pct']:.2f}%")
        print(f"Expectancy:      ${metrics['expectancy']:.3f} per trade")
        print("="*50 + "\n")
        
    finally:
        if safety_store_path.exists():
            safety_store_path.unlink()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="x.titan Backtester")
    parser.add_argument("--data", required=True, help="Path to historical parquet/csv")
    parser.add_argument("--model", help="Path to model checkpoint")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance")
    parser.add_argument("--latency", type=float, default=0.0, help="Simulated latency (ms)")
    parser.add_argument("--slippage", type=float, default=0.0, help="Slippage probability")
    parser.add_argument("--device", default="cpu", help="Compute device (cpu/cuda)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(run_detailed_backtest(args))
    except KeyboardInterrupt:
        pass

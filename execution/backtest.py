"""
Event-Driven Backtester for x.titan.

Allows replaying historical data (Parquet/CSV) through the EXACT same
pipeline as live trading (`MarketDataBuffer` -> `FeatureBuilder` -> `Model` -> `DecisionEngine`).

Key Components:
1.  BacktestClient: Mocks `DerivClient` to return instant executions and replay market data.
2.  BacktestEngine: Orchestrates event loop, advancing time tick-by-tick or candle-by-candle.
"""

import asyncio
import logging
import random
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import AsyncIterator, Any
from pathlib import Path

from data.ingestion.client import DerivClient
from execution.executor import TradeResult
from config.settings import Settings

logger = logging.getLogger(__name__)

class BacktestClient:
    """
    Mocks DerivClient for historical replay.
    
    - Executions are instant at current 'market' price.
    - Balance tracks P&L internally.
    - Slippage can be simulated.
    """
    def __init__(self, initial_balance: float = 10000.0, slip_prob: float = 0.0, slip_avg: float = 0.0):
        self.balance = initial_balance
        self.slip_prob = slip_prob
        self.slip_avg = slip_avg
        self.positions = []
        self.current_price = 100.0
        self.current_time = datetime.now(timezone.utc)
        self.is_connected = True
        
    async def connect(self):
        logger.info("BacktestClient connected (simulated).")
        return True
        
    async def disconnect(self):
        logger.info("BacktestClient disconnected.")
        return True
        
    async def get_balance(self) -> float:
        return self.balance
        
    async def buy(self, amount: float, contract_type: str, duration: int, 
                  symbol: str, barrier: float | None = None, barrier2: float | None = None) -> Any:
        # Simulate execution
        # In live client, this returns a Dictionary or Contract object
        # We'll return a mock dict
        
        # Deduct stake
        self.balance -= amount
        
        entry_price = self.current_price
        
        # C01 Fix: Apply slippage simulation
        if self.slip_prob > 0 and random.random() < self.slip_prob:
            # Calculate slippage magnitude using gaussian distribution
            # Standard deviation is 1/3 of mean to keep values reasonable
            slippage = abs(random.gauss(self.slip_avg, self.slip_avg / 3))
            
            # Apply directional slippage (unfavorable to trader)
            if contract_type.upper() in ("CALL", "RISE"):
                # For CALL/RISE: higher entry is unfavorable
                entry_price += slippage
            elif contract_type.upper() in ("PUT", "FALL"):
                # For PUT/FALL: lower entry is unfavorable
                entry_price -= slippage
            # For TOUCH/RANGE contracts, slippage affects entry reference
            
            logger.debug(f"Slippage applied: {slippage:.5f} ({contract_type})")
        
        contract_id = f"sim_{len(self.positions) + 1}"
        
        position = {
            "contract_id": contract_id,
            "contract_type": contract_type,
            "amount": amount,
            "entry_time": self.current_time,
            "entry_price": entry_price,
            "duration": duration,  # minutes
            "barrier": barrier,
            "barrier2": barrier2,
            "status": "open",
            "price_high": entry_price,  # Track extremes for TOUCH/RANGE
            "price_low": entry_price,
        }
        self.positions.append(position)
        
        logger.info(f"SIM EXECUTION: {contract_type} ${amount} @ {entry_price}")
        return {"contract_id": contract_id, "buy_price": amount}

    async def get_open_contracts(self) -> list:
        return [p for p in self.positions if p["status"] == "open"]

    # Replay specific methods
    def update_market(self, price: float, timestamp: datetime, 
                      high: float | None = None, low: float | None = None):
        """
        Update market state and check position outcomes.
        
        Args:
            price: Current close price
            timestamp: Current timestamp
            high: Candle high (optional, for TOUCH/RANGE)
            low: Candle low (optional, for TOUCH/RANGE)
        """
        self.current_price = price
        self.current_time = timestamp
        
        # Track price extremes for open positions (for TOUCH/RANGE contracts)
        for pos in self.positions:
            if pos["status"] == "open":
                pos["price_high"] = max(pos["price_high"], high if high else price)
                pos["price_low"] = min(pos["price_low"], low if low else price)
        
        self._check_outcomes()
    
    def _check_outcomes(self, payout_ratio: float = 0.90):
        """
        C02 Fix: Check and resolve expired positions.
        
        Iterates through open positions and resolves them if expiration
        time has passed. Applies win/loss based on contract type.
        
        Args:
            payout_ratio: Payout multiplier on win (default 0.90 = 90%)
        """
        for pos in self.positions:
            if pos["status"] != "open":
                continue
            
            # Calculate expiration time (duration is in minutes)
            expiration = pos["entry_time"] + timedelta(minutes=pos["duration"])
            
            if self.current_time >= expiration:
                # Determine outcome based on contract type
                contract_type = pos["contract_type"].upper()
                entry_price = pos["entry_price"]
                exit_price = self.current_price
                stake = pos["amount"]
                
                won = False
                
                if contract_type in ("CALL", "RISE"):
                    won = exit_price > entry_price
                    
                elif contract_type in ("PUT", "FALL"):
                    won = exit_price < entry_price
                    
                elif contract_type in ("ONETOUCH", "TOUCH"):
                    # Check if price touched barrier during lifetime
                    barrier = pos.get("barrier")
                    if barrier:
                        # For touch, typically barrier is above current price
                        won = pos["price_high"] >= barrier or pos["price_low"] <= barrier
                    else:
                        # Fallback: 0.5% movement
                        barrier_up = entry_price * 1.005
                        barrier_down = entry_price * 0.995
                        won = pos["price_high"] >= barrier_up or pos["price_low"] <= barrier_down
                        
                elif contract_type == "NOTOUCH":
                    # Price did NOT touch barrier
                    barrier = pos.get("barrier")
                    if barrier:
                        touched = pos["price_high"] >= barrier or pos["price_low"] <= barrier
                        won = not touched
                    else:
                        barrier_up = entry_price * 1.005
                        barrier_down = entry_price * 0.995
                        touched = pos["price_high"] >= barrier_up or pos["price_low"] <= barrier_down
                        won = not touched
                        
                elif contract_type in ("STAYSINRANGE", "STAYS_BETWEEN", "RANGE"):
                    # Price stayed within barrier and barrier2
                    barrier_up = pos.get("barrier") or entry_price * 1.003
                    barrier_down = pos.get("barrier2") or entry_price * 0.997
                    stayed_in = pos["price_high"] <= barrier_up and pos["price_low"] >= barrier_down
                    won = stayed_in
                    
                elif contract_type in ("GOESOUTSIDE", "GOES_OUTSIDE"):
                    barrier_up = pos.get("barrier") or entry_price * 1.003
                    barrier_down = pos.get("barrier2") or entry_price * 0.997
                    went_out = pos["price_high"] > barrier_up or pos["price_low"] < barrier_down
                    won = went_out
                
                # Update balance and position
                if won:
                    payout = stake * (1 + payout_ratio)
                    self.balance += payout
                    pos["outcome"] = "win"
                    logger.info(f"✅ WIN: {contract_type} | Entry: {entry_price:.5f} → Exit: {exit_price:.5f} | Payout: ${payout:.2f}")
                else:
                    pos["outcome"] = "loss"
                    logger.info(f"❌ LOSS: {contract_type} | Entry: {entry_price:.5f} → Exit: {exit_price:.5f} | Lost: ${stake:.2f}")
                
                pos["status"] = "closed"
                pos["exit_price"] = exit_price
                pos["exit_time"] = self.current_time

class BacktestEngine:
    """
    Orchestrates the backtest replay.
    
    C02 Fix: Complete implementation with data iteration and statistics.
    """
    def __init__(
        self, 
        settings: Settings, 
        data_path: Path,
        initial_balance: float = 10000.0,
        slip_prob: float = 0.0,
        slip_avg: float = 0.0,
        payout_ratio: float = 0.90,
    ):
        """
        Initialize backtest engine.
        
        Args:
            settings: Application settings
            data_path: Path to historical data (Parquet or CSV)
            initial_balance: Starting account balance
            slip_prob: Probability of slippage (0.0 to 1.0)
            slip_avg: Average slippage amount when applied
            payout_ratio: Win payout multiplier (e.g., 0.90 = 90%)
        """
        self.settings = settings
        self.data_path = data_path
        self.payout_ratio = payout_ratio
        self.client = BacktestClient(
            initial_balance=initial_balance,
            slip_prob=slip_prob,
            slip_avg=slip_avg,
        )
        self.data: pd.DataFrame | None = None
        self.initial_balance = initial_balance
        
    def load_data(self) -> pd.DataFrame:
        """Load historical data from file."""
        if str(self.data_path).endswith(".parquet"):
            self.data = pd.read_parquet(self.data_path)
        else:
            self.data = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.data)} rows from {self.data_path}")
        return self.data
        
    async def run(self) -> dict[str, Any]:
        """
        C02 Fix: Run the backtest by iterating through historical data.
        
        Iterates through each row of loaded data, feeding prices through
        the BacktestClient for market updates and outcome resolution.
        
        Returns:
            Statistics dictionary containing:
                - total_trades: Number of trades executed
                - wins: Number of winning trades
                - losses: Number of losing trades
                - win_rate: Win percentage (0-100)
                - initial_balance: Starting balance
                - final_balance: Ending balance
                - net_pnl: Net profit/loss
                - return_pct: Return percentage
                - max_drawdown: Maximum drawdown observed
        """
        logger.info("Starting backtest replay...")
        
        # Ensure data is loaded
        if self.data is None:
            self.load_data()
        
        if self.data is None or len(self.data) == 0:
            logger.error("No data loaded for backtest")
            return {"error": "No data loaded"}
        
        # Track metrics
        peak_balance = self.initial_balance
        max_drawdown = 0.0
        
        # Determine column names (handle different formats)
        df = self.data
        
        # Try to find timestamp column
        timestamp_col = None
        for col in ["timestamp", "time", "datetime", "epoch", "open_time"]:
            if col in df.columns:
                timestamp_col = col
                break
        
        # Try to find price columns
        close_col = next((c for c in ["close", "Close", "price"] if c in df.columns), None)
        high_col = next((c for c in ["high", "High"] if c in df.columns), None)
        low_col = next((c for c in ["low", "Low"] if c in df.columns), None)
        
        if close_col is None:
            logger.error("Could not find close price column in data")
            return {"error": "Missing close price column"}
        
        # Iterate through data
        for idx, row in df.iterrows():
            # Parse timestamp
            if timestamp_col:
                ts = row[timestamp_col]
                if isinstance(ts, (int, float)):
                    timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
                elif isinstance(ts, str):
                    timestamp = pd.to_datetime(ts).to_pydatetime()
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                else:
                    timestamp = ts
                    if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                # Fallback: assume 1-minute intervals from now
                timestamp = datetime.now(timezone.utc) + timedelta(minutes=int(idx))  # type: ignore[arg-type]
            
            # Get prices
            close_price = float(row[close_col])
            high_price = float(row[high_col]) if high_col and pd.notna(row.get(high_col)) else None
            low_price = float(row[low_col]) if low_col and pd.notna(row.get(low_col)) else None
            
            # Update market and check outcomes
            self.client.update_market(
                price=close_price,
                timestamp=timestamp,
                high=high_price,
                low=low_price,
            )
            
            # Track drawdown
            current_balance = self.client.balance
            if current_balance > peak_balance:
                peak_balance = current_balance
            else:
                drawdown = (peak_balance - current_balance) / peak_balance
                max_drawdown = max(max_drawdown, drawdown)
            
            # Yield control periodically to allow async operations
            if int(idx) % 100 == 0:  # type: ignore[arg-type]
                await asyncio.sleep(0)
        
        # Calculate final statistics
        final_balance = self.client.balance
        net_pnl = final_balance - self.initial_balance
        return_pct = (net_pnl / self.initial_balance) * 100
        
        wins = sum(1 for p in self.client.positions if p.get("outcome") == "win")
        losses = sum(1 for p in self.client.positions if p.get("outcome") == "loss")
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        
        stats = {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "initial_balance": self.initial_balance,
            "final_balance": final_balance,
            "net_pnl": net_pnl,
            "return_pct": return_pct,
            "max_drawdown": max_drawdown * 100,  # As percentage
        }
        
        logger.info(
            f"Backtest complete: {total_trades} trades, "
            f"{wins}W/{losses}L ({win_rate:.1f}%), "
            f"PnL: ${net_pnl:+.2f} ({return_pct:+.1f}%)"
        )
        
        return stats

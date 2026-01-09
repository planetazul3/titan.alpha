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
import torch
from datetime import datetime, timezone, timedelta
from typing import AsyncIterator, Any
from pathlib import Path

from data.events import CandleEvent
from data.buffer import MarketDataBuffer

from data.ingestion.client import DerivClient
from execution.executor import TradeResult
from config.settings import Settings
from execution.metrics import TradingMetrics

logger = logging.getLogger(__name__)

class BacktestClient:
    """
    Mocks DerivClient for historical replay.
    
    - Executions are instant at current 'market' price.
    - Balance tracks P&L internally.
    - Slippage can be simulated.
    """
    def __init__(self, initial_balance: float = 10000.0, slip_prob: float = 0.0, slip_avg: float = 0.0, latency_ms: float = 0.0):
        self.balance = initial_balance
        self.slip_prob = slip_prob
        self.slip_avg = slip_avg
        self.latency_ms = latency_ms
        self.positions: list[dict[str, Any]] = []
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
        # Simulate network latency
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)

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
        # Pipeline Components (Optional for simple replay, required for full simulation)
        buffer: Any = None,
        feature_builder: Any = None,
        model: Any = None,
        decision_engine: Any = None,
        executor: Any = None,
        strategy_adapter: Any = None,
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
            buffer: MarketDataBuffer (optional)
            feature_builder: FeatureBuilder (optional)
            model: PyTorch model (optional)
            decision_engine: DecisionEngine (optional)
            executor: SafeTradeExecutor (optional)
            strategy_adapter: StrategyAdapter (optional)
        """
        self.settings = settings
        self.data_path = data_path
        self.payout_ratio = payout_ratio
        
        self.buffer = buffer
        self.feature_builder = feature_builder
        self.model = model
        self.decision_engine = decision_engine
        self.executor = executor
        self.strategy_adapter = strategy_adapter
        
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
            
            # C02: Execute Pipeline if components are present
            if self.buffer and self.feature_builder and self.model and self.decision_engine and self.executor:
                # 1. Create Candle Event
                event = CandleEvent(
                    symbol=self.settings.trading.symbol,
                    timestamp=timestamp,
                    open=float(row.get("open", close_price)),
                    high=high_price if high_price else close_price,
                    low=low_price if low_price else close_price,
                    close=close_price,
                    volume=float(row.get("volume", 0.0))
                )
                
                # 2. Update Buffer
                is_new = self.buffer.update_candle(event)
                
                # 3. Run Inference on new candle
                if is_new and self.buffer.is_ready():
                    try:
                        # Build Features
                        features = self.feature_builder.build_numpy(
                            self.buffer.snapshot(),
                            timestamp=timestamp.timestamp()
                        )
                        
                        # Prepare Tensor
                        tick_window, candle_window = features
                        
                        # Convert to tensor
                        device = self.settings.get_device()
                        tick_tensor = torch.from_numpy(tick_window).float().unsqueeze(0).to(device)
                        candle_tensor = torch.from_numpy(candle_window).float().unsqueeze(0).to(device)
                        
                        # Inference
                        self.model.eval()
                        with torch.inference_mode():  # I4: More memory-efficient
                            outputs = self.model(tick_tensor, candle_tensor)
                            
                        # Parse outputs (assuming model returns dict or tuple)
                        # Standard x.titan model return: (probs_dict, reconstruction_error, etc)
                        # Or if it's the newer model: {"probs": ..., "reconstruction_error": ...}
                        
                        probs = {}
                        reconstruction_error = 0.0
                        
                        if isinstance(outputs, dict):
                            probs = outputs.get("probs", {})
                            reconstruction_error = outputs.get("reconstruction_error", 0.0)
                        elif isinstance(outputs, tuple):
                             # Compat with older models
                             probs = outputs[0]
                             if len(outputs) > 1:
                                 reconstruction_error = outputs[1]

                        # 4. Decision Engine
                        signals = await self.decision_engine.process_model_output(
                            probs=probs,
                            reconstruction_error=reconstruction_error,
                            timestamp=timestamp,
                            market_data={"close": close_price}
                        )
                        
                        # 5. Execution
                        from execution.executor import ExecutionRequest
                        
                        for signal in signals:
                            if self.strategy_adapter:
                                req = self.strategy_adapter.convert_signal(
                                    signal, 
                                    account_balance=self.client.balance
                                )
                            else:
                                # Fallback: direct execution request with simple sizing.
                                from execution.common.contract_mapping import map_signal_to_contract_type
                                req = ExecutionRequest(
                                    signal_id=signal.signal_id,
                                    symbol=signal.metadata.get("symbol", self.settings.trading.symbol),
                                    contract_type=map_signal_to_contract_type(signal),
                                    duration=1, # Default
                                    duration_unit="m",
                                    stake=10.0, # Default
                                    barrier=None,
                                    barrier2=None
                                )
                            
                            logger.info(f"Signal generated: {signal.direction} @ {timestamp}")
                            await self.executor.execute(req)
                            
                    except Exception as e:
                        logger.error(f"Pipeline error at {timestamp}: {e}", exc_info=True)

            
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
        
        # Prepare trades list for metrics calculator
        trade_history = []
        for pos in self.client.positions:
            if pos["status"] == "closed":
                outcome = pos.get("outcome", "unknown")
                # Profit/Loss = Payout - Stake (if win), -Stake (if loss)
                # But our client implementation adds payout to balance on win.
                # Logic:
                # Win: Payout = Stake * (1 + PayoutRatio)
                #      Profit = Payout - Stake = Stake * PayoutRatio
                # Loss: Profit = -Stake
                
                # However, backtest client logic:
                # buy: balance -= stake
                # win: balance += stake * (1+payout)
                # net: stake * payout
                
                # Let's derive PnL from outcome
                stake = pos["amount"]
                if outcome == "win":
                    pnl = stake * self.payout_ratio
                else:
                    pnl = -stake
                    
                trade_history.append({
                    "outcome": outcome,
                    "profit_loss": pnl,
                    "exit_time": pos.get("exit_time")
                })

        metrics = TradingMetrics.calculate(trade_history, initial_balance=self.initial_balance)
        
        logger.info(
            f"Backtest complete: {metrics.total_trades} trades, "
            f"{metrics.winning_trades}W/{metrics.losing_trades}L ({metrics.win_rate:.1f}%), "
            f"PnL: ${metrics.net_profit:+.2f} ({metrics.return_pct:+.1f}%)"
            f"Sharpe: {metrics.sharpe_ratio:.2f}, DD: {metrics.max_drawdown_pct:.2f}%"
        )
        
        return metrics.__dict__

"""
Unit tests for the backtest module.
"""

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from execution.backtest import BacktestClient, BacktestEngine

class TestBacktestClient:
    """Tests for BacktestClient."""

    @pytest.mark.asyncio
    async def test_buy_reduces_balance(self):
        client = BacktestClient(initial_balance=1000.0)
        client.update_market(price=1.12345, timestamp=datetime.now(timezone.utc))
        
        await client.buy(
            contract_type="CALL",
            amount=100.0,
            duration=1,  # 1 minute
            symbol="R_100"
        )
        
        assert client.balance == 900.0
        assert len(client.positions) == 1
        assert client.positions[0]["amount"] == 100.0

    @pytest.mark.asyncio
    async def test_slippage_application(self):
        # Setting slip_prob to 1.0 to guarantee slippage
        client = BacktestClient(initial_balance=1000.0, slip_prob=1.0, slip_avg=0.01)
        client.update_market(price=1.0000, timestamp=datetime.now(timezone.utc))
        
        # Test CALL slippage (entry price should be HIGHER)
        await client.buy(contract_type="CALL", amount=10.0, duration=1, symbol="R_100")
        call_pos = client.positions[0]
        assert call_pos["entry_price"] > 1.0000
        
        # Test PUT slippage (entry price should be LOWER)
        client.update_market(price=1.0000, timestamp=datetime.now(timezone.utc))
        await client.buy(contract_type="PUT", amount=10.0, duration=1, symbol="R_100")
        put_pos = client.positions[1]
        assert put_pos["entry_price"] < 1.0000

    @pytest.mark.asyncio
    async def test_outcome_resolution_win(self):
        client = BacktestClient(initial_balance=1000.0)
        entry_time = datetime.now(timezone.utc)
        client.update_market(price=1.0000, timestamp=entry_time)
        
        # Buy CALL
        await client.buy(contract_type="CALL", amount=100.0, duration=1, symbol="R_100")
        
        # Expire with higher price
        exit_time = entry_time + timedelta(minutes=2)
        client.update_market(price=1.0050, timestamp=exit_time)
        
        assert client.balance == 1090.0
        assert client.positions[0]["status"] == "closed"
        assert client.positions[0]["outcome"] == "win"

    @pytest.mark.asyncio
    async def test_outcome_resolution_loss(self):
        client = BacktestClient(initial_balance=1000.0)
        entry_time = datetime.now(timezone.utc)
        client.update_market(price=1.0000, timestamp=entry_time)
        
        # Buy CALL
        await client.buy(contract_type="CALL", amount=100.0, duration=1, symbol="R_100")
        
        # Expire with lower price
        exit_time = entry_time + timedelta(minutes=2)
        client.update_market(price=0.9950, timestamp=exit_time)
        
        # Balance should remain 900.0
        assert client.balance == 900.0
        assert client.positions[0]["status"] == "closed"
        assert client.positions[0]["outcome"] == "loss"

class TestBacktestEngine:
    """Tests for BacktestEngine."""

    @pytest.mark.asyncio
    async def test_engine_run(self, tmp_path):
        # Create dummy data file
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            "epoch": [1000, 1001, 1002, 1003, 1004, 1005],
            "close": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        })
        df.to_csv(csv_path, index=False)
        
        from unittest.mock import MagicMock
        from config.settings import Settings
        settings = MagicMock(spec=Settings)
        
        engine = BacktestEngine(
            settings=settings,
            data_path=csv_path,
            initial_balance=1000.0
        )
        
        # Run engine
        stats = await engine.run()
            
        assert stats["final_balance"] == 1000.0 # No trades made
        assert stats["total_trades"] == 0

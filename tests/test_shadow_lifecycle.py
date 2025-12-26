
import unittest
import asyncio
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.shadow_resolution import ShadowTradeResolver
from execution.sqlite_shadow_store import SQLiteShadowStore
from execution.decision import DecisionEngine
from execution.signals import TradeSignal
from config.constants import SIGNAL_TYPES, CONTRACT_TYPES

@dataclass
class MockTrade:
    contract_type: str
    direction: str
    entry_price: float
    probability: float
    timestamp: datetime




class TestShadowLifecycle(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Use temp file for testing to ensure connection sharing across threads/loops
        import tempfile
        self.tmp_dir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmp_dir.name, "test_shadow.db")
        self.store = SQLiteShadowStore(db_path)
        self.resolver = ShadowTradeResolver(self.store)

    def tearDown(self):
        self.store.close()
        self.tmp_dir.cleanup()
        
    async def test_lifecycle_flow(self):
        """Verify the complete lifecycle: Store -> Resolve -> Win/Loss"""
        now = datetime.now(timezone.utc)
        
        # 1. Create a shadow trade
        trade = MockTrade(
            contract_type=CONTRACT_TYPES.RISE_FALL,
            direction="CALL",
            entry_price=100.0,
            probability=0.75,
            timestamp=now
        )
        
        # 2. Store via proper API
        from execution.shadow_store import ShadowTradeRecord
        from data.features import FEATURE_SCHEMA_VERSION
        
        record = ShadowTradeRecord.create(
            contract_type=trade.contract_type,
            direction=trade.direction,
            probability=trade.probability,
            entry_price=trade.entry_price,
            reconstruction_error=0.1,
            regime_state="trusted",
            tick_window=[],
            candle_window=[],
            model_version="test_v1",
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            metadata={"contract_duration": 1}
        )
        trade_id = record.trade_id
        self.store.append(record)
        
        self.assertIsNotNone(trade_id)
        
        # 2. Verify trade is "OPEN"
        open_trades = self.store.query(unresolved_only=True)
        self.assertEqual(len(open_trades), 1)
        self.assertEqual(open_trades[0].trade_id, trade_id)
        
        # 3. Simulate candle CLOSING (Price moved UP -> WIN)
        close_time = now + timedelta(minutes=1, seconds=1)
        # Note: Store doesn't track candles, we pass price to resolver
        
        # 4. Run Resolver
        # We pass the closing price of the candle directly
        resolved_count = await self.resolver.resolve_trades(current_price=100.5, current_time=close_time)
        self.assertEqual(resolved_count, 1)
        
        # 5. Verify Outcome
        # Status should be CLOSED
        open_trades_after = self.store.query(unresolved_only=True)
        self.assertEqual(len(open_trades_after), 0)
        
        # Check metrics/history
        stats = self.store.get_statistics()
        self.assertEqual(stats["total_records"], 1)
        self.assertEqual(stats["wins"], 1) # CALL + price went up = WIN
        self.assertAlmostEqual(stats["win_rate"], 1.0)
        
        print(f"\n✅ Lifecycle Test Passed: Trade {trade_id} resolved as WIN")

    async def test_loss_scenario(self):
        """Verify LOSS resolution logic"""
        now = datetime.now(timezone.utc)
        
        # Create PUT trade
        from execution.shadow_store import ShadowTradeRecord
        from data.features import FEATURE_SCHEMA_VERSION

        record = ShadowTradeRecord.create(
            contract_type=CONTRACT_TYPES.RISE_FALL,
            direction="PUT",
            probability=0.65,
            entry_price=100.0,
            reconstruction_error=0.1,
            regime_state="trusted",
            tick_window=[],
            candle_window=[],
            model_version="test_v1",
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            metadata={"contract_duration": 1}
        )
        trade_id = record.trade_id
        self.store.append(record)
        
        # Simulate price going UP (Loss for PUT)
        close_time = now + timedelta(minutes=1, seconds=5)
        
        # Pass failing price (101.0 > 100.0, bad for PUT)
        await self.resolver.resolve_trades(current_price=101.0, current_time=close_time)
        
        stats = self.store.get_statistics()
        self.assertEqual(stats["wins"], 0)
        self.assertEqual(stats["losses"], 1)
        self.assertEqual(stats["win_rate"], 0.0)
        print(f"\n✅ Loss Scenario Passed: Trade resolved as LOSS")

if __name__ == "__main__":
    unittest.main()

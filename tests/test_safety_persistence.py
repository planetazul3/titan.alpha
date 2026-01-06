
import pytest
import unittest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock

from execution.common.types import ExecutionRequest
from execution.safety import SafeTradeExecutor, ExecutionSafetyConfig
from execution.safety_store import SQLiteSafetyStateStore
from config.constants import CONTRACT_TYPES, SIGNAL_TYPES

class TestSafetyPersistence(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "safety_test.db"
        self.config = ExecutionSafetyConfig(
            max_trades_per_minute=2, # Small limit for testing
            max_trades_per_minute_per_symbol=2,
            max_daily_loss=100.0,
            max_stake_per_trade=10.0,
            kill_switch_enabled=False
        )
        self.mock_inner = AsyncMock()
        self.mock_inner.execute.return_value = MagicMock(success=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_signal(self, symbol="R_100"):
        return ExecutionRequest(
            signal_id="SIG_PERSIST",
            symbol=symbol,
            contract_type="RISE_FALL",
            stake=5.0,
            duration=1,
            duration_unit="m"
        )

    def test_persistence_across_restarts(self):
        """Test that rate limits persist across executor restarts."""
        async def run_test():
            # 1. Start Executor A
            executor_a = SafeTradeExecutor(
                self.mock_inner, self.config, self.db_path
            )
            
            # Execute 2 trades (hitting the limit)
            sig = self._create_signal()
            await executor_a.execute(sig)
            await executor_a.execute(sig)
            
            # 2. Start Executor B (simulating restart) pointing to SAME DB
            executor_b = SafeTradeExecutor(
                self.mock_inner, self.config, self.db_path
            )
            
            # 3. Try to execute 3rd trade -> Should fail immediately due to persistence
            result = await executor_b.execute(sig)
            
            self.assertFalse(result.success)
            self.assertIn("Rate limit exceeded", result.error)
            
            # Verify DB state
            store = SQLiteSafetyStateStore(self.db_path)
            count = store.get_trades_in_window(None, 60)
            self.assertEqual(count, 2)
            
        asyncio.run(run_test())

    def test_concurrency_locking(self):
        """Test that concurrent executions don't race to bypass limits."""
        async def run_test():
            # Config allows 2 trades. We try to launch 5 concurrently.
            executor = SafeTradeExecutor(
                self.mock_inner, self.config, self.db_path
            )
            
            sig = self._create_signal()
            tasks = [executor.execute(sig) for _ in range(5)]
            
            results = await asyncio.gather(*tasks)
            
            success_count = sum(1 for r in results if r.success)
            failure_count = sum(1 for r in results if not r.success)
            
            # Should strictly allow only 2
            self.assertEqual(success_count, 2)
            self.assertEqual(failure_count, 3)
            
        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()

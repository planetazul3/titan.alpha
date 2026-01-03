
import unittest
import tempfile
import shutil
from pathlib import Path
from execution.adaptive_risk import AdaptiveRiskManager
from execution.safety_store import SQLiteSafetyStateStore

class TestRiskPersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = Path(self.test_dir) / "test_safety.db"
        self.store = SQLiteSafetyStateStore(self.db_path)
        
    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.test_dir)
        
    def test_state_persistence(self):
        """Test that risk state including returns history is persisted and restored."""
        # 1. Initialize manager with store
        manager1 = AdaptiveRiskManager(state_store=self.store)
        
        # 2. Record some trades to generate history and state
        # Win (+10), Loss (-5), Win (+15)
        manager1.record_trade(10.0, current_equity=1010.0)
        manager1.record_trade(-5.0, current_equity=1005.0)
        manager1.record_trade(15.0, current_equity=1020.0)
        
        # Add buffer trades to hit minimum length for Sharpe (10)
        for _ in range(10):
            manager1.record_trade(1.0, current_equity=1021.0)
        
        # Verify in-memory state of manager1
        self.assertEqual(len(manager1.performance._returns), 13)
        self.assertEqual(manager1.performance._peak_equity, 1021.0)
        
        # Calculate metrics
        sharpe1 = manager1.performance.get_sharpe_ratio()
        self.assertGreater(sharpe1, 0.0)
        
        # 3. Create NEW manager instance using SAME store (simulating restart)
        # Force a fresh load from DB
        manager2 = AdaptiveRiskManager(state_store=self.store)
        
        # 4. Verify restored state
        self.assertEqual(len(manager2.performance._returns), 13)
        # Check first 3 and last 1
        restored = list(manager2.performance._returns)
        self.assertEqual(restored[:3], [10.0, -5.0, 15.0])
        self.assertEqual(restored[-1], 1.0)
        self.assertEqual(manager2.performance._peak_equity, 1021.0)
        
        # 5. Verify metrics match
        sharpe2 = manager2.performance.get_sharpe_ratio()
        self.assertAlmostEqual(sharpe1, sharpe2, places=5)
        
    def test_empty_store_handling(self):
        """Test that manager handles empty/new store gracefully."""
        # New manager, empty DB
        manager = AdaptiveRiskManager(state_store=self.store)
        
        self.assertEqual(len(manager.performance._returns), 0)
        self.assertEqual(manager.performance._current_drawdown, 0.0)
        
    def test_partial_state_recovery(self):
        """Test recovery when only some fields exist (backward compatibility)."""
        # Manually write partial state (mimicking old schema)
        import time
        ts = time.time()
        with self.store._transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, ?, ?)",
                ("risk_peak_equity", "1000.0", ts)
            )
            
        manager = AdaptiveRiskManager(state_store=self.store)
        
        self.assertEqual(manager.performance._peak_equity, 1000.0)
        self.assertEqual(len(manager.performance._returns), 0)  # Should default to empty

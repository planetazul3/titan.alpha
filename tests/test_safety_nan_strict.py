
import unittest
import shutil
import tempfile
import math
from pathlib import Path
from execution.safety_store import SQLiteSafetyStateStore

class TestSafetyStoreNaN(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = Path(self.test_dir) / "test_safety.db"
        self.store = SQLiteSafetyStateStore(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_update_pnl_valid(self):
        """Valid PnL should be accepted."""
        self.store.update_daily_pnl(10.5)
        count, pnl = self.store.get_daily_stats()
        self.assertAlmostEqual(pnl, 10.5)

    def test_update_pnl_nan_raises(self):
        """NaN PnL should raise ValueError."""
        with self.assertRaises(ValueError) as cm:
            self.store.update_daily_pnl(float('nan'))
        self.assertIn("CRITICAL: Attempted to persist NaN PnL", str(cm.exception))
        
    def test_update_pnl_inf_raises(self):
        """Infinity PnL should raise ValueError."""
        with self.assertRaises(ValueError):
            self.store.update_daily_pnl(float('inf'))

if __name__ == "__main__":
    unittest.main()

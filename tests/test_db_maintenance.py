
import unittest
import sqlite3
import os
from pathlib import Path
from execution.sqlite_shadow_store import SQLiteShadowStore
from execution.shadow_store import ShadowTradeRecord
from datetime import datetime, timedelta, timezone

class TestDBMaintenance(unittest.TestCase):
    def setUp(self):
        self.db_path = Path("test_maintenance.db")
        if self.db_path.exists():
            os.remove(self.db_path)
        self.store = SQLiteShadowStore(self.db_path)
        
    def tearDown(self):
        self.store.close()
        if self.db_path.exists():
            os.remove(self.db_path)
            
    def test_prune_reclaims_space(self):
        """Test that prune deletes old records and runs vacuum."""
        # Insert old records
        old_time = datetime.now(timezone.utc) - timedelta(days=60)
        for i in range(10):
            record = ShadowTradeRecord(
                trade_id=f"old_{i}",
                timestamp=old_time,
                contract_type="CALL",
                direction="UP",
                probability=0.8,
                entry_price=100.0,
                reconstruction_error=0.1,
                regime_state="BULL",
                model_version="1.0",
                feature_schema_version="1.0",
                outcome=True,
                exit_price=101.0,
                resolved_at=old_time
            )
            record._created_at = old_time.isoformat()
            self.store.append(record)
            
        # Insert new records
        new_time = datetime.now(timezone.utc)
        for i in range(5):
            record = ShadowTradeRecord(
                trade_id=f"new_{i}",
                timestamp=new_time,
                contract_type="CALL",
                direction="UP",
                probability=0.8,
                entry_price=100.0,
                reconstruction_error=0.1,
                regime_state="BULL",
                model_version="1.0",
                feature_schema_version="1.0",
                outcome=True,
                exit_price=101.0,
                resolved_at=new_time
            )
            self.store.append(record)
            
        # Verify total count
        stats = self.store.get_statistics()
        self.assertEqual(stats["total_records"], 15)
        
        # Run Prune (retention 30 days)
        # Verify it works without error (including VACUUM/Checkpoint)
        deleted = self.store.prune(retention_days=30)
        
        # Verify counts
        stats = self.store.get_statistics()
        self.assertEqual(deleted, 10)
        self.assertEqual(stats["total_records"], 5)
        
        # Verify old records gone
        self.assertIsNone(self.store.get_by_id("old_1"))
        self.assertIsNotNone(self.store.get_by_id("new_1"))

if __name__ == '__main__':
    unittest.main()

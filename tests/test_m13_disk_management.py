
import pytest
import time
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, timezone
from utils.logging_setup import cleanup_logs
from execution.sqlite_shadow_store import SQLiteShadowStore
from execution.shadow_store import ShadowTradeRecord

class TestDiskManagement:
    
    def test_log_cleanup(self, tmp_path):
        """Verify log cleanup deletes old files."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        
        # Create recent file
        (log_dir / "recent.log").touch()
        
        # Create old file (10 days old)
        old_file = log_dir / "old.log"
        old_file.touch()
        # Set mtime to 10 days ago
        old_time = time.time() - (10 * 86400)
        import os
        os.utime(old_file, (old_time, old_time))
        
        # Run cleanup (keep 7 days)
        deleted = cleanup_logs(log_dir, retention_days=7)
        
        # Verify
        assert deleted == 1
        assert not old_file.exists()
        assert (log_dir / "recent.log").exists()

    def test_db_pruning(self, tmp_path):
        """Verify DB pruning deletes old records."""
        db_path = tmp_path / "test.db"
        store = SQLiteShadowStore(db_path)
        
        # Helper to insert
        def insert_record(trade_id: str, days_ago: int):
            ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
            record = ShadowTradeRecord(
                trade_id=trade_id,
                timestamp=ts,
                contract_type="CALL",
                direction="UP",
                probability=0.8,
                entry_price=100.0,
                reconstruction_error=0.1,
                regime_state="BULL",
                model_version="v1",
                feature_schema_version="1.0"
            )
            store.append(record)
            # Hack to force timestamp back because append uses record.timestamp but let's be sure
            # Actually append uses record.timestamp so it's fine.
            
        # Insert recent record (1 day old)
        insert_record("recent", 1)
        
        # Insert old record (40 days old)
        insert_record("old", 40)
        
        # Verify both exist
        assert store.get_by_id("recent") is not None
        assert store.get_by_id("old") is not None
        
        # Prune (keep 30 days)
        deleted = store.prune(retention_days=30)
        
        # Verify
        assert deleted == 1
        assert store.get_by_id("recent") is not None
        assert store.get_by_id("old") is None
        
        store.close()

if __name__ == "__main__":
    pytest.main([__file__])

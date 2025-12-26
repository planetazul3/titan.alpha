"""
Unit tests for SQLite shadow store.

Tests include:
- Basic CRUD operations
- Atomic outcome updates
- Concurrent access safety
- Query filtering
- Migration from NDJSON
"""

import json
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from execution.shadow_store import ShadowTradeRecord
from execution.sqlite_shadow_store import SQLiteShadowStore


class TestSQLiteShadowStore:
    """Tests for SQLiteShadowStore basic operations."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_shadow.db"

    @pytest.fixture
    def sample_record(self):
        """Create sample shadow trade record."""
        return ShadowTradeRecord.create(
            contract_type="RISE_FALL",
            direction="CALL",
            probability=0.75,
            entry_price=100.0,
            reconstruction_error=0.05,
            regime_state="TRUSTED",
            tick_window=[99.0, 99.5, 100.0],
            candle_window=[[99.0, 100.5, 98.5, 100.0, 1000.0]],
            model_version="v1.0.0",
            feature_schema_version="1.0",
            metadata={"symbol": "R_100"},
        )

    def test_init_creates_database(self, temp_db):
        """Database file should be created on initialization."""
        store = SQLiteShadowStore(temp_db)
        assert temp_db.exists()
        store.close()

    def test_append_and_query(self, temp_db, sample_record):
        """Should append and query records."""
        store = SQLiteShadowStore(temp_db)

        store.append(sample_record)

        records = store.query()
        assert len(records) == 1
        assert records[0].trade_id == sample_record.trade_id
        assert records[0].probability == 0.75
        store.close()

    def test_append_preserves_data(self, temp_db, sample_record):
        """All record fields should be preserved."""
        store = SQLiteShadowStore(temp_db)
        store.append(sample_record)

        records = store.query()
        r = records[0]

        assert r.contract_type == "RISE_FALL"
        assert r.direction == "CALL"
        assert r.probability == 0.75
        assert r.entry_price == 100.0
        assert r.reconstruction_error == 0.05
        assert r.regime_state == "TRUSTED"
        assert r.tick_window == [99.0, 99.5, 100.0]
        assert r.candle_window == [[99.0, 100.5, 98.5, 100.0, 1000.0]]
        assert r.model_version == "v1.0.0"
        assert r.metadata == {"symbol": "R_100"}
        store.close()

    def test_get_by_id(self, temp_db, sample_record):
        """Should fetch single record by ID."""
        store = SQLiteShadowStore(temp_db)
        store.append(sample_record)

        record = store.get_by_id(sample_record.trade_id)

        assert record is not None
        assert record.trade_id == sample_record.trade_id

        # Non-existent ID
        assert store.get_by_id("non-existent") is None
        store.close()

    def test_update_outcome_atomic(self, temp_db, sample_record):
        """Outcome updates should be atomic (no file rewrite)."""
        store = SQLiteShadowStore(temp_db)
        store.append(sample_record)

        # Verify unresolved
        record = store.get_by_id(sample_record.trade_id)
        assert record.outcome is None
        assert record.exit_price is None

        # Update outcome
        success = store.update_outcome(record, True, 101.5)

        assert success is True

        # Verify updated
        record = store.get_by_id(sample_record.trade_id)
        assert record.outcome is True
        assert record.exit_price == 101.5
        assert record.resolved_at is not None
        store.close()

    def test_update_outcome_nonexistent(self, temp_db):
        """Test updating a non-existent trade."""
        store = SQLiteShadowStore(temp_db)
        from dataclasses import dataclass
        @dataclass
        class DummyRecord:
            trade_id: str
        
        dummy = DummyRecord(trade_id="nonexistent_id")
        success = store.update_outcome(dummy, True, 100.0)

        assert success is False
        store.close()


class TestSQLiteShadowStoreQueries:
    """Tests for query filtering."""

    @pytest.fixture
    def populated_store(self, tmp_path):
        """Create store with multiple records."""
        db_path = tmp_path / "test.db"
        store = SQLiteShadowStore(db_path)

        now = datetime.now(timezone.utc)

        # Create records at different times
        for i in range(5):
            record = ShadowTradeRecord(
                trade_id=f"trade_{i}",
                timestamp=now - timedelta(hours=5 - i),
                contract_type="RISE_FALL",
                direction="CALL" if i % 2 == 0 else "PUT",
                probability=0.7 + i * 0.05,
                entry_price=100.0 + i,
                reconstruction_error=0.05,
                regime_state="TRUSTED",
                tick_window=[],
                candle_window=[],
                outcome=True if i < 2 else (False if i == 2 else None),
                exit_price=101.0 if i < 3 else None,
                resolved_at=now if i < 3 else None,
            )
            store.append(record)

        yield store
        store.close()

    def test_query_all(self, populated_store):
        """Should return all records."""
        records = populated_store.query()
        assert len(records) == 5

    def test_query_resolved_only(self, populated_store):
        """Should filter to resolved records."""
        records = populated_store.query(resolved_only=True)
        assert len(records) == 3
        assert all(r.outcome is not None for r in records)

    def test_query_unresolved_only(self, populated_store):
        """Should filter to unresolved records."""
        records = populated_store.query(unresolved_only=True)
        assert len(records) == 2
        assert all(r.outcome is None for r in records)

    def test_query_time_range(self, populated_store):
        """Should filter by time range."""
        now = datetime.now(timezone.utc)

        # Last 3 hours only
        records = populated_store.query(start=now - timedelta(hours=3))
        assert len(records) >= 2


class TestSQLiteShadowStoreStatistics:
    """Tests for statistics calculation."""

    def test_statistics_empty(self, tmp_path):
        """Should handle empty store."""
        store = SQLiteShadowStore(tmp_path / "empty.db")

        stats = store.get_statistics()

        assert stats["total_records"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["storage_backend"] == "sqlite"
        store.close()

    def test_statistics_with_data(self, tmp_path):
        """Should compute correct statistics."""
        store = SQLiteShadowStore(tmp_path / "test.db")

        # Add 4 records: 2 wins, 1 loss, 1 unresolved
        for i, (outcome, resolved) in enumerate(
            [(True, True), (True, True), (False, True), (None, False)]
        ):
            record = ShadowTradeRecord(
                trade_id=f"trade_{i}",
                timestamp=datetime.now(timezone.utc),
                contract_type="RISE_FALL",
                direction="CALL",
                probability=0.75,
                entry_price=100.0,
                reconstruction_error=0.05,
                regime_state="TRUSTED",
                outcome=outcome,
                exit_price=101.0 if resolved else None,
                resolved_at=datetime.now(timezone.utc) if resolved else None,
            )
            store.append(record)

        stats = store.get_statistics()

        assert stats["total_records"] == 4
        assert stats["resolved_records"] == 3
        assert stats["unresolved_records"] == 1
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert stats["win_rate"] == pytest.approx(2 / 3)
        store.close()


class TestSQLiteShadowStoreConcurrency:
    """Tests for concurrent access safety."""

    def test_concurrent_writes(self, tmp_path):
        """Multiple threads should be able to write safely."""
        db_path = tmp_path / "concurrent.db"
        store = SQLiteShadowStore(db_path)

        errors = []

        def writer(thread_id):
            try:
                for i in range(10):
                    record = ShadowTradeRecord.create(
                        contract_type="RISE_FALL",
                        direction="CALL",
                        probability=0.75,
                        entry_price=100.0,
                        reconstruction_error=0.05,
                        regime_state="TRUSTED",
                        tick_window=[],
                        candle_window=[],
                        metadata={"thread": thread_id, "index": i},
                    )
                    store.append(record)
            except Exception as e:
                errors.append(e)

        # Start 5 writer threads
        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0

        # Should have all records
        records = store.query()
        assert len(records) == 50
        store.close()

    def test_concurrent_reads_writes(self, tmp_path):
        """Should handle concurrent reads and writes."""
        db_path = tmp_path / "concurrent_rw.db"
        store = SQLiteShadowStore(db_path)

        # Pre-populate
        for _i in range(10):
            record = ShadowTradeRecord.create(
                contract_type="RISE_FALL",
                direction="CALL",
                probability=0.75,
                entry_price=100.0,
                reconstruction_error=0.05,
                regime_state="TRUSTED",
                tick_window=[],
                candle_window=[],
            )
            store.append(record)

        errors = []
        read_counts = []

        def reader():
            try:
                for _ in range(10):
                    records = store.query()
                    read_counts.append(len(records))
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for _i in range(5):
                    record = ShadowTradeRecord.create(
                        contract_type="RISE_FALL",
                        direction="CALL",
                        probability=0.75,
                        entry_price=100.0,
                        reconstruction_error=0.05,
                        regime_state="TRUSTED",
                        tick_window=[],
                        candle_window=[],
                    )
                    store.append(record)
                    time.sleep(0.02)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Final count should be 15 (10 initial + 5 from writer)
        assert len(store.query()) == 15
        store.close()


class TestSQLiteShadowStoreMigration:
    """Tests for NDJSON migration."""

    def test_migrate_from_ndjson(self, tmp_path):
        """Should import records from NDJSON file."""
        # Create NDJSON file
        ndjson_path = tmp_path / "shadow_trades.ndjson"
        records = []

        for i in range(5):
            record = ShadowTradeRecord.create(
                contract_type="RISE_FALL",
                direction="CALL",
                probability=0.75,
                entry_price=100.0,
                reconstruction_error=0.05,
                regime_state="TRUSTED",
                tick_window=[99.0, 100.0],
                candle_window=[[99.0, 100.5, 98.5, 100.0]],
                metadata={"index": i},
            )
            records.append(record)

        with open(ndjson_path, "w") as f:
            for r in records:
                f.write(json.dumps(r.to_dict()) + "\n")

        # Migrate
        db_path = tmp_path / "migrated.db"
        store = SQLiteShadowStore.from_ndjson(ndjson_path, db_path)

        # Verify
        migrated = store.query()
        assert len(migrated) == 5

        # Check data integrity
        for i, r in enumerate(migrated):
            assert r.contract_type == "RISE_FALL"
            assert r.probability == 0.75

        store.close()

    def test_migrate_handles_malformed(self, tmp_path):
        """Should skip malformed lines during migration."""
        ndjson_path = tmp_path / "messy.ndjson"

        with open(ndjson_path, "w") as f:
            # Valid record
            record = ShadowTradeRecord.create(
                contract_type="RISE_FALL",
                direction="CALL",
                probability=0.75,
                entry_price=100.0,
                reconstruction_error=0.05,
                regime_state="TRUSTED",
                tick_window=[],
                candle_window=[],
            )
            f.write(json.dumps(record.to_dict()) + "\n")

            # Malformed lines
            f.write("not valid json\n")
            f.write('{"incomplete": true}\n')
            f.write("\n")  # Empty line

            # Another valid record
            record2 = ShadowTradeRecord.create(
                contract_type="TOUCH",
                direction="TOUCH",
                probability=0.80,
                entry_price=105.0,
                reconstruction_error=0.08,
                regime_state="CAUTION",
                tick_window=[],
                candle_window=[],
            )
            f.write(json.dumps(record2.to_dict()) + "\n")

        db_path = tmp_path / "cleaned.db"
        store = SQLiteShadowStore.from_ndjson(ndjson_path, db_path)

        # Should have imported 2 valid records
        records = store.query()
        assert len(records) == 2
        store.close()

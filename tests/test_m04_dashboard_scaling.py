
import pytest
import sqlite3
import time
from unittest.mock import MagicMock, patch
from api.dashboard_server import get_shadow_trade_stats, get_shadow_trades, ShadowTradeStats, STATS_CACHE_TTL
from concurrent.futures import ThreadPoolExecutor

class TestDashboardScaling:
    
    @pytest.fixture
    def mock_db_connection(self):
        """Mock sqlite connection to avoid file I/O."""
        with patch("api.dashboard_server.readonly_connection") as mock_conn:
            mock_cursor = MagicMock()
            mock_conn.return_value.__enter__.return_value.cursor.return_value = mock_cursor
            yield mock_cursor

    def test_stats_caching(self, mock_db_connection):
        """Verify that get_shadow_trade_stats caches results."""
        # Reset cache
        import api.dashboard_server
        api.dashboard_server._stats_cache = {"data": None, "timestamp": 0.0}
        
        # Setup mock return values
        # Total, Resolved, Wins
        mock_db_connection.execute.return_value.fetchone.side_effect = [(100,), (80,), (40,)]
        
        # First call: Should query DB
        stats1 = get_shadow_trade_stats()
        assert stats1.total == 100
        assert mock_db_connection.execute.call_count == 3
        
        # Second call immediately: Should return cached result (no new DB calls)
        stats2 = get_shadow_trade_stats()
        assert stats2.total == 100
        # Call count should remain 3 (no new execution)
        assert mock_db_connection.execute.call_count == 3
        
        # Simulate cache expiration
        import time
        api.dashboard_server._stats_cache["timestamp"] = time.time() - (STATS_CACHE_TTL + 1)
        
        # Setup mock for second query (different values to prove it's a new query)
        mock_db_connection.execute.return_value.fetchone.side_effect = [(102,), (80,), (40,)]
        
        # Third call: Should query DB again
        stats3 = get_shadow_trade_stats()
        assert stats3.total == 102
        assert mock_db_connection.execute.call_count == 6  # 3 initial + 3 new

    @pytest.mark.asyncio
    async def test_keyset_pagination(self, mock_db_connection):
        """Verify get_shadow_trades uses correct SQL for keyset pagination."""
        # We need to mock the context manager properly for async function
        with patch("api.dashboard_server.readonly_connection") as mock_conn_ctx:
             mock_conn = MagicMock()
             mock_cursor = MagicMock()
             mock_conn_ctx.return_value.__enter__.return_value = mock_conn
             mock_conn.cursor.return_value = mock_cursor
             
             # Case 1: Legacy OFFSET
             await get_shadow_trades(limit=10, offset=5)
             
             # Check SQL info
             call_args = mock_cursor.execute.call_args
             sql_query = call_args[0][0]
             assert "LIMIT ? OFFSET ?" in sql_query
             assert "WHERE timestamp < ?" not in sql_query
             
             # Case 2: Keyset 'before'
             await get_shadow_trades(limit=10, before="2025-01-01T00:00:00")
             
             call_args = mock_cursor.execute.call_args
             sql_query = call_args[0][0]
             assert "WHERE timestamp < ?" in sql_query
             assert "OFFSET" not in sql_query

if __name__ == "__main__":
    import asyncio
    asyncio.run(TestDashboardScaling().test_keyset_pagination(None))

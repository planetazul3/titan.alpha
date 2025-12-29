"""
Tests for SafeTradeExecutor.reconcile_with_api() functionality.

This module verifies the startup reconciliation feature that compares
API open contracts with the pending trade store to detect discrepancies.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
import tempfile

from execution.safety import SafeTradeExecutor, ExecutionSafetyConfig


class TestSafetyReconciliation:
    """Tests for reconcile_with_api safety feature."""

    @pytest.fixture
    def temp_state_file(self):
        """Provide isolated state file for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_reconcile.db"

    @pytest.fixture
    def mock_executor(self):
        """Create mock underlying executor."""
        executor = MagicMock()
        return executor

    @pytest.fixture
    def config(self):
        """Create standard safety config."""
        return ExecutionSafetyConfig()

    @pytest.mark.asyncio
    async def test_reconcile_empty_state(
        self, mock_executor, config, temp_state_file
    ):
        """Empty API and store should produce no discrepancies."""
        # Setup
        mock_client = MagicMock()
        mock_client.get_open_contracts = AsyncMock(return_value={})
        
        mock_store = MagicMock()
        mock_store.get_all_pending = MagicMock(return_value=[])
        
        safe_executor = SafeTradeExecutor(mock_executor, config, state_file=temp_state_file)
        
        # Execute
        result = await safe_executor.reconcile_with_api(mock_client, mock_store)
        
        # Verify
        assert result["api_open_count"] == 0
        assert result["store_pending_count"] == 0
        assert result["matched"] == []
        assert result["api_only"] == []
        assert result["store_only"] == []
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_reconcile_matched_contracts(
        self, mock_executor, config, temp_state_file
    ):
        """API and store with same contracts should show as matched."""
        # Setup - API returns list format
        mock_client = MagicMock()
        mock_client.get_open_contracts = AsyncMock(return_value=[
            {"contract_id": "12345"},
            {"contract_id": "67890"}
        ])
        
        mock_store = MagicMock()
        mock_store.get_all_pending = MagicMock(return_value=[
            {"contract_id": "12345"},
            {"contract_id": "67890"}
        ])
        
        safe_executor = SafeTradeExecutor(mock_executor, config, state_file=temp_state_file)
        
        # Execute
        result = await safe_executor.reconcile_with_api(mock_client, mock_store)
        
        # Verify
        assert result["api_open_count"] == 2
        assert result["store_pending_count"] == 2
        assert set(result["matched"]) == {"12345", "67890"}
        assert result["api_only"] == []
        assert result["store_only"] == []

    @pytest.mark.asyncio
    async def test_reconcile_api_only_contracts(
        self, mock_executor, config, temp_state_file
    ):
        """Contracts on API but not in store should be flagged."""
        # Setup
        mock_client = MagicMock()
        mock_client.get_open_contracts = AsyncMock(return_value=[
            {"contract_id": "API_ONLY_123"}
        ])
        
        mock_store = MagicMock()
        mock_store.get_all_pending = MagicMock(return_value=[])
        
        safe_executor = SafeTradeExecutor(mock_executor, config, state_file=temp_state_file)
        
        # Execute
        result = await safe_executor.reconcile_with_api(mock_client, mock_store)
        
        # Verify
        assert result["api_open_count"] == 1
        assert result["store_pending_count"] == 0
        assert result["api_only"] == ["API_ONLY_123"]
        assert result["store_only"] == []

    @pytest.mark.asyncio
    async def test_reconcile_store_only_contracts(
        self, mock_executor, config, temp_state_file
    ):
        """Contracts in store but not on API are already settled."""
        # Setup
        mock_client = MagicMock()
        mock_client.get_open_contracts = AsyncMock(return_value=[])
        
        mock_store = MagicMock()
        mock_store.get_all_pending = MagicMock(return_value=[
            {"contract_id": "ALREADY_SETTLED_456"}
        ])
        
        safe_executor = SafeTradeExecutor(mock_executor, config, state_file=temp_state_file)
        
        # Execute
        result = await safe_executor.reconcile_with_api(mock_client, mock_store)
        
        # Verify
        assert result["api_open_count"] == 0
        assert result["store_pending_count"] == 1
        assert result["api_only"] == []
        assert result["store_only"] == ["ALREADY_SETTLED_456"]

    @pytest.mark.asyncio
    async def test_reconcile_without_pending_store(
        self, mock_executor, config, temp_state_file
    ):
        """Reconciliation should work without a pending store."""
        # Setup
        mock_client = MagicMock()
        mock_client.get_open_contracts = AsyncMock(return_value=[
            {"contract_id": "12345"}
        ])
        
        safe_executor = SafeTradeExecutor(mock_executor, config, state_file=temp_state_file)
        
        # Execute without pending_store
        result = await safe_executor.reconcile_with_api(mock_client, pending_store=None)
        
        # Verify
        assert result["api_open_count"] == 1
        assert result["store_pending_count"] == 0
        assert result["api_only"] == ["12345"]

    @pytest.mark.asyncio
    async def test_reconcile_handles_api_error(
        self, mock_executor, config, temp_state_file
    ):
        """Reconciliation should handle API errors gracefully."""
        # Setup
        mock_client = MagicMock()
        mock_client.get_open_contracts = AsyncMock(side_effect=Exception("API unavailable"))
        
        safe_executor = SafeTradeExecutor(mock_executor, config, state_file=temp_state_file)
        
        # Execute
        result = await safe_executor.reconcile_with_api(mock_client)
        
        # Verify - should not raise, returns error in result
        assert result["error"] is not None
        assert "API unavailable" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

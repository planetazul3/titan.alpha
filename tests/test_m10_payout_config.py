
import pytest
from unittest.mock import MagicMock
from observability.shadow_metrics import ShadowTradeMetrics

class TestPayoutConfig:
    def test_shadow_metrics_respects_ratio(self):
        """Verify ShadowTradeMetrics calculates PnP using passed ratio."""
        metrics = ShadowTradeMetrics()
        
        # Mock shadow store
        store = MagicMock()
        # 10 wins, 10 losses.
        # If payout is 1.0 (100%), PnL = 10*1 - 10*1 = 0
        # If payout is 0.9 (90%), PnL = 10*0.9 - 10 = -1
        store.get_statistics.return_value = {
            "total_records": 20,
            "resolved_records": 20,
            "unresolved_records": 0,
            "wins": 10,
            "losses": 10,
            "win_rate": 0.5
        }
        # Mock query return empty to avoid detailed logic
        store.query.return_value = []
        
        # Test case 1: Payout 1.0
        metrics.update_from_store(store, payout_ratio=1.0)
        assert metrics.simulated_pnl == 0.0
        
        # Test case 2: Payout 0.9
        metrics.update_from_store(store, payout_ratio=0.9)
        assert metrics.simulated_pnl == pytest.approx(-1.0)
        
        # Test case 3: Payout 0.5 (very bad broker)
        metrics.update_from_store(store, payout_ratio=0.5)
        assert metrics.simulated_pnl == pytest.approx(-5.0)

if __name__ == "__main__":
    pytest.main([__file__])

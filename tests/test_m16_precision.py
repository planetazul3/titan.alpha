
import pytest
from decimal import Decimal
from execution.real_trade_tracker import RealTradeTracker

class TestPrecision:
    
    def test_accumulation_accuracy(self):
        """Verify Decimal accumulation prevents drift."""
        tracker = RealTradeTracker()
        
        # Add 0.1 ten times
        # In float: 0.1 * 10 often becomes 0.99999999 or 1.0000001
        # In Decimal: 1.0 exactly
        
        profit = 0.1
        for _ in range(10):
            # We simulate what _handle_outcome does: convert to Decimal(str(profit))
            tracker._total_pnl += Decimal(str(profit))
            
        assert tracker._total_pnl == Decimal("1.0")
        
        # Verify stats output conversion
        stats = tracker.get_statistics()
        assert stats["total_pnl"] == 1.0
        assert isinstance(stats["total_pnl"], float)

if __name__ == "__main__":
    pytest.main([__file__])

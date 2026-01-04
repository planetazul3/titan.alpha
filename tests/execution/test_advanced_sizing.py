
import pytest
from unittest.mock import MagicMock
from execution.position_sizer import KellyPositionSizer, TargetVolatilitySizer, PositionSizeResult
from execution.strategy_adapter import StrategyAdapter
from execution.signals import TradeSignal
from config.settings import Settings

class TestAdvancedSizing:
    def test_kelly_volatility_scaling(self):
        """Test that KellyPositionSizer reduces stake when numeric volatility is high."""
        sizer = KellyPositionSizer(base_stake=100, high_volatility_reduction=0.5, max_stake=100.0)
        
        # Win probability 0.6, payout 0.9 -> Kelly = (0.6*0.9 - 0.4)/0.9 = 0.1555
        # Adjusted = 0.1555 * 0.5 (safety) = 0.0777
        # Stake = 0.0777 * 100 * 10 (scale) approx 77
        
        # Test 1: Normal Volatility (0.2)
        res_normal = sizer.compute_stake(
            probability=0.6,
            volatility=0.2
        )
        
        # Test 2: High Volatility (0.9) - should be halved
        res_high = sizer.compute_stake(
            probability=0.6,
            volatility=0.9
        )
        
        assert res_high.stake < res_normal.stake
        # Should be roughly half, depending on rounding and min stakes
        assert res_high.stake < (res_normal.stake * 0.6) 

    def test_target_volatility_sizer(self):
        """Test TargetVolatilitySizer scales inversely to volatility."""
        sizer = TargetVolatilitySizer(base_stake=10.0, target_volatility=0.20)
        
        # Case 1: Volatility == Target -> Base Stake
        res_target = sizer.compute_stake(volatility=0.20)
        assert res_target.stake == 10.0
        
        # Case 2: Volatility == 2x Target (0.40) -> Half Stake
        res_high = sizer.compute_stake(volatility=0.40)
        assert res_high.stake == 5.0
        
        # Case 3: Volatility == 0.5x Target (0.10) -> Double Stake
        res_low = sizer.compute_stake(volatility=0.10)
        assert res_low.stake == 20.0
        
        # Case 4: Bounds Check
        sizer_bounded = TargetVolatilitySizer(base_stake=10.0, target_volatility=0.2, max_stake=15.0)
        res_bounded = sizer_bounded.compute_stake(volatility=0.01) # Massive multiplier
        assert res_bounded.stake == 15.0

    def test_strategy_adapter_volatility_passing(self):
        """Verify StrategyAdapter pulls volatility from signal metadata."""
        settings = Settings()
        mock_sizer = MagicMock()
        mock_sizer.suggest_stake_for_signal.return_value = 10.0
        
        adapter = StrategyAdapter(settings=settings, position_sizer=mock_sizer)
        
        signal = TradeSignal(
            signal_id="test",
            timestamp=12345,
            symbol="TEST",
            signal_type="REAL",
            contract_type="CALL",
            direction="CALL",
            probability=0.8,
            metadata={"volatility": 0.45}
        )
        
        adapter.convert_signal(signal)
        
        # Verify call arguments
        mock_sizer.suggest_stake_for_signal.assert_called_once()
        args, kwargs = mock_sizer.suggest_stake_for_signal.call_args
        assert kwargs.get("volatility") == 0.45


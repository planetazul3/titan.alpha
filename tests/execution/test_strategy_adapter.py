
import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone
import asyncio

from config.settings import Settings
from config.constants import CONTRACT_TYPES, SIGNAL_TYPES
from execution.signals import TradeSignal
from execution.common.types import ExecutionRequest
from execution.strategy_adapter import StrategyAdapter
from execution.position_sizer import FixedStakeSizer

class TestExecutionRequest:
    def test_validation(self):
        """Test ExecutionRequest validation logic."""
        # Valid request
        req = ExecutionRequest(
            signal_id="sig_1",
            symbol="R_100",
            contract_type="CALL",
            stake=10.0,
            duration=1,
            duration_unit="m"
        )
        assert req.stake == 10.0
        
        # Invalid stake
        with pytest.raises(ValueError, match="stake must be positive"):
            ExecutionRequest(
                signal_id="sig_1",
                symbol="R_100",
                contract_type="CALL",
                stake=-5.0,
                duration=1,
                duration_unit="m"
            )
            
        # Invalid duration
        with pytest.raises(ValueError, match="duration must be positive"):
            ExecutionRequest(
                signal_id="sig_1",
                symbol="R_100",
                contract_type="CALL",
                stake=10.0,
                duration=0,
                duration_unit="m"
            )

class TestStrategyAdapter:
    def setup_method(self):
        self.settings = MagicMock(spec=Settings)
        # Mock nested settings
        self.settings.trading = MagicMock()
        self.settings.trading.symbol = "R_100"
        self.settings.trading.stake_amount = 10.0
        self.settings.trading.timeframe = "1m"
        
        self.settings.contracts = MagicMock()
        self.settings.contracts.default_duration = 1
        self.settings.contracts.default_duration_unit = "m"
        
        # Mock position sizer
        self.sizer = FixedStakeSizer(stake=20.0)
        
        # Mock duration resolver
        self.duration_resolver = MagicMock()
        self.duration_resolver.resolve_duration.return_value = (1, "m")
        
        self.adapter = StrategyAdapter(
            settings=self.settings,
            position_sizer=self.sizer,
            duration_resolver=self.duration_resolver
        )

    def test_convert_signal_basic(self):
        """Test conversion of a standard signal."""
        signal = TradeSignal(
            signal_id="sig_1",
            signal_type=SIGNAL_TYPES.REAL_TRADE,
            timestamp=datetime.now(timezone.utc),
            direction="CALL",
            probability=0.8,
            contract_type=CONTRACT_TYPES.RISE_FALL,
            metadata={"symbol": "R_100"}
        )
        
        request = self.adapter.convert_signal(signal)
        
        assert isinstance(request, ExecutionRequest)
        assert request.signal_id == "sig_1"
        assert request.symbol == "R_100"
        assert request.contract_type == "CALL" # Mapped from RISE_FALL + CALL
        assert request.stake == 20.0 # From FixedStakeSizer
        assert request.duration == 1 # Default resolution for 1m
        assert request.duration_unit == "m"

    def test_convert_signal_put(self):
        """Test conversion of a PUT signal."""
        signal = TradeSignal(
            signal_id="sig_2",
            signal_type=SIGNAL_TYPES.REAL_TRADE,
            timestamp=datetime.now(timezone.utc),
            direction="PUT",
            probability=0.8,
            contract_type=CONTRACT_TYPES.RISE_FALL,
            metadata={"symbol": "R_100"}
        )
        
        request = self.adapter.convert_signal(signal)
        assert request.contract_type == "PUT"

    def test_convert_signal_touch(self):
        """Test conversion of a TOUCH signal."""
        signal = TradeSignal(
            signal_id="sig_3",
            signal_type=SIGNAL_TYPES.REAL_TRADE,
            timestamp=datetime.now(timezone.utc),
            direction="TOUCH",
            probability=0.8,
            contract_type=CONTRACT_TYPES.TOUCH_NO_TOUCH,
            metadata={"barrier": "+1.5", "symbol": "R_100"}
        )
        
        request = self.adapter.convert_signal(signal)
        assert request.contract_type == "ONETOUCH"
        assert request.barrier == "+1.5"
    
    def test_stake_override_from_metadata(self):
        """Test that metadata stake overrides position sizer."""
        signal = TradeSignal(
            signal_id="sig_4",
            signal_type=SIGNAL_TYPES.REAL_TRADE,
            timestamp=datetime.now(timezone.utc),
            direction="CALL",
            probability=0.8,
            contract_type=CONTRACT_TYPES.RISE_FALL,
            metadata={"stake": 50.0, "symbol": "R_100"}
        )
        
        request = self.adapter.convert_signal(signal)
        assert request.stake == 50.0 

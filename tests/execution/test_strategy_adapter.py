
import pytest
from unittest.mock import MagicMock, patch
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

@pytest.mark.asyncio
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
        
        # Mock param_service (Since StrategyAdapter instantiates it in __init__, we need to patch class or use Dependency Injection)
        # For this test, let's just let it be created or patch execution.strategy_adapter.ContractParameterService
        
        # We need to start the patcher here and stop it in teardown, or use a context manager that persists?
        # Standard pattern for setup_method:
        self.patcher = patch('execution.strategy_adapter.ContractParameterService')
        MockService = self.patcher.start()
        instance = MockService.return_value
        instance.resolve_duration.return_value = (1, "m")
        instance.resolve_barriers.return_value = (None, None)
        
        self.adapter = StrategyAdapter(
            settings=self.settings,
            position_sizer=self.sizer
        )

    def teardown_method(self):
        self.patcher.stop()

    async def test_convert_signal_basic(self):
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
        
        # IMPORTANT-001: Await async conversion
        request = await self.adapter.convert_signal(signal)
        
        assert isinstance(request, ExecutionRequest)
        assert request.signal_id == "sig_1"
        assert request.symbol == "R_100"
        assert request.contract_type == "CALL" # Mapped from RISE_FALL + CALL
        assert request.stake == 20.0 # From FixedStakeSizer
        assert request.duration == 1 # Default resolution for 1m
        assert request.duration_unit == "m"

    async def test_convert_signal_put(self):
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
        
        request = await self.adapter.convert_signal(signal)
        assert request.contract_type == "PUT"

    async def test_convert_signal_touch(self):
        """Test conversion of a TOUCH signal."""
        # Setup specific return for touch
        self.adapter.param_service.resolve_barriers.return_value = ("+1.5", None)
        
        signal = TradeSignal(
            signal_id="sig_3",
            signal_type=SIGNAL_TYPES.REAL_TRADE,
            timestamp=datetime.now(timezone.utc),
            direction="TOUCH",
            probability=0.8,
            contract_type=CONTRACT_TYPES.TOUCH_NO_TOUCH,
            metadata={"barrier": "+1.5", "symbol": "R_100"}
        )
        
        request = await self.adapter.convert_signal(signal)
        assert request.contract_type == "ONETOUCH"
        # Since we mocked the service in setup (technically), we need to ensure the instance we are using is the one we configured
        # But wait, StrategyAdapter creates NEW instance.
        # So we better rely on correct init.
        # Actually in test_convert_signal_basic, I patched the class but didn't assign the mock to `self.adapter.param_service`.
        # StrategyAdapter.__init__ does: self.param_service = ContractParameterService(settings)
        # So if we want to control behaviors per test, we should overwrite it.
        
        # Manual Override for this test
        self.adapter.param_service.resolve_barriers = MagicMock(return_value=("+1.5", None))
        request = await self.adapter.convert_signal(signal)
        
        assert request.barrier == "+1.5"
    
    async def test_stake_override_from_metadata(self):
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
        
        request = await self.adapter.convert_signal(signal)
        assert request.stake == 50.0 

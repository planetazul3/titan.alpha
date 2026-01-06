import logging
from typing import Protocol, Optional, Any

from config.settings import Settings
from config.constants import CONTRACT_TYPES
from execution.common.types import ExecutionRequest
from execution.signals import TradeSignal
from execution.contract_params import ContractParameterService
from execution.position_sizer import PositionSizer, FixedStakeSizer
from execution.common.contract_mapping import map_signal_to_contract_type

logger = logging.getLogger(__name__)

class StrategyAdapter:
    """
    Bridge between Decision Engine (Signals) and Execution (Requests).
    
    Responsibilities:
    1.  Apply Position Sizing (Risk Management).
    2.  Resolve Contract Durations (Market Microstructure).
    3.  Map Signal Types to Contract Types (Protocol Adaptation).
    4.  Enforce Pre-Execution Constraints (Safety).
    """
    
    def __init__(
        self,
        settings: Settings,
        position_sizer: Optional[PositionSizer] = None,
        # duration_resolver no longer used, kept for signature compat if needed, but we prefer param_service
    ):
        self.settings = settings
        self.position_sizer = position_sizer or FixedStakeSizer(stake=settings.trading.stake_amount)
        self.param_service = ContractParameterService(settings)
        
    async def convert_signal(
        self, 
        signal: TradeSignal, 
        account_balance: Optional[float] = None,
        reconstruction_error: Optional[float] = None,
        regime_state: Optional[str] = None
    ) -> ExecutionRequest:
        """
        Convert a raw TradeSignal into a validated ExecutionRequest.
        """
        # 1. Determine Stake
        # Allow explicit override from signal metadata (e.g. from RL agent)
        stake = signal.metadata.get("stake")
        if stake is None:
            # R02: Pass volatility from metadata if available
            volatility = signal.metadata.get("volatility", 0.0)
            
            # IMPORTANT-001: Async Sizing Support
            if hasattr(self.position_sizer, "suggest_stake_for_signal_async"):
                 stake = await self.position_sizer.suggest_stake_for_signal_async(
                    signal, 
                    account_balance=account_balance,
                    volatility=volatility
                 )
            else:
                 stake = self.position_sizer.suggest_stake_for_signal(
                    signal, 
                    account_balance=account_balance,
                    volatility=volatility
                 )
            
        # 2. Resolve Duration and Barriers (CRITICAL-003)
        duration, duration_unit = self.param_service.resolve_duration(signal.contract_type)
        barrier, barrier2 = self.param_service.resolve_barriers(signal.contract_type)
        
        # 3. Map Contract Type (Signal -> API)
        deriv_contract_type = map_signal_to_contract_type(signal)
        
        # 4. Extract Barriers Override (if any in metadata)
        if signal.metadata.get("barrier"):
             barrier = signal.metadata.get("barrier")
        if signal.metadata.get("barrier2"):
             barrier2 = signal.metadata.get("barrier2")
        
        return ExecutionRequest(
            signal_id=signal.signal_id,
            symbol=signal.metadata.get("symbol", self.settings.trading.symbol),
            contract_type=deriv_contract_type,
            stake=float(stake),
            duration=duration,
            duration_unit=duration_unit,
            barrier=barrier,
            barrier2=barrier2,
            # CRITICAL-004: Regime Context
            regime_state=regime_state,
            reconstruction_error=reconstruction_error
        )
        


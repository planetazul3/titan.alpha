
import logging
from typing import Protocol, Optional, Any

from config.settings import Settings
from config.constants import CONTRACT_TYPES
from execution.common.types import ExecutionRequest
from execution.signals import TradeSignal
from execution.contract_params import ContractDurationResolver
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
        duration_resolver: Optional[ContractDurationResolver] = None
    ):
        self.settings = settings
        self.position_sizer = position_sizer or FixedStakeSizer(stake=settings.trading.stake_amount)
        self.duration_resolver = duration_resolver or ContractDurationResolver(settings)
        
    def convert_signal(
        self, 
        signal: TradeSignal, 
        account_balance: Optional[float] = None
    ) -> ExecutionRequest:
        """
        Convert a raw TradeSignal into a validated ExecutionRequest.
        """
        # 1. Determine Stake
        # Allow explicit override from signal metadata (e.g. from RL agent)
        stake = signal.metadata.get("stake")
        stake = signal.metadata.get("stake")
        if stake is None:
            # R02: Pass volatility from metadata if available
            volatility = signal.metadata.get("volatility", 0.0)
            stake = self.position_sizer.suggest_stake_for_signal(
                signal, 
                account_balance=account_balance,
                volatility=volatility
            )
            
        # 2. Resolve Duration
        duration, duration_unit = self.duration_resolver.resolve_duration(signal.contract_type)
        
        # 3. Map Contract Type (Signal -> API)
        # TODO: Move this mapping to a shared utility? 
        # Ideally Executor handles the API-specific string mapping, but Request should be explicit.
        # Let's keep the high-level types in Request and let Executor map to Deriv strings if they differ?
        # Actually, Executor is "DerivTradeExecutor", so it expects Deriv strings or our constants?
        # Current Executor maps:
        # types.CONTRACT_TYPES.RISE_FALL -> "CALL"/"PUT"
        # Let's do the rigorous mapping HERE so the Request is unambiguous.
        
        deriv_contract_type = map_signal_to_contract_type(signal)
        
        # 4. Extract Barriers
        barrier = signal.metadata.get("barrier")
        barrier2 = signal.metadata.get("barrier2")
        
        return ExecutionRequest(
            signal_id=signal.signal_id,
            symbol=signal.metadata.get("symbol", self.settings.trading.symbol),
            contract_type=deriv_contract_type,
            stake=float(stake),
            duration=duration,
            duration_unit=duration_unit,
            barrier=barrier,
            barrier2=barrier2
        )
        


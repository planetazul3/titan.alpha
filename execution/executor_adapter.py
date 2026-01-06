from config.settings import Settings
from execution.common.types import ExecutionRequest
from execution.contract_params import ContractParameterService
from execution.signals import TradeSignal

class SignalAdapter:
    """
    Adapter to convert TradeSignals into executable ExecutionRequests.
    
    This bridges the gap between the Decision Engine (signals) and 
    the Execution Engine (requests), applying:
    1. Position Sizing (mapping probability to stake)
    2. Duration Resolution (looking up correct duration for contract type)
    3. Barrier Logic (applying offsets for Touch/Range contracts)
    """
    
    def __init__(self, settings: Settings, position_sizer=None):
        self.settings = settings
        self.position_sizer = position_sizer
        self.param_service = ContractParameterService(settings)
        
    def to_execution_request(self, signal: TradeSignal) -> ExecutionRequest:
        """
        Convert a high-level TradeSignal into a precise ExecutionRequest.
        
        Args:
            signal: The trade signal to execute
            
        Returns:
            ExecutionRequest populated with resolved parameters
        """
        # 1. Resolve Contract Details
        # NOTE: signal.contract_type is 'RISE_FALL', signal.direction is 'CALL'.
        # The Executor passes `request.contract_type` to `client.buy`.
        
        contract_type = signal.direction
        if not contract_type:
             raise ValueError(f"Signal {signal.signal_id} has no direction/contract_type")

        # 2. Resolve Duration and Barriers (CRITICAL-003)
        duration, unit = self.param_service.resolve_duration(signal.contract_type)
        barrier, barrier2 = self.param_service.resolve_barriers(signal.contract_type)
        
        # 3. Resolve Stake
        if self.position_sizer:
            try:
                # Use the protocol method
                stake = self.position_sizer.suggest_stake_for_signal(signal)
            except AttributeError:
                # Fallback if position_sizer doesn't follow protocol exactly
                if hasattr(self.position_sizer, "compute_stake"):
                     res = self.position_sizer.compute_stake(probability=signal.probability)
                     stake = res.stake
                else:
                    stake = self.settings.trading.stake_amount
        else:
            stake = self.settings.trading.stake_amount

        # 4. Build Request
        return ExecutionRequest(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            contract_type=contract_type,
            stake=stake,
            duration=duration,
            duration_unit=unit,
            barrier=barrier,
            barrier2=barrier2
        )

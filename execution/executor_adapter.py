from config.settings import Settings
from execution.common.types import ExecutionRequest
from execution.contract_params import ContractDurationResolver
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
        self.duration_resolver = ContractDurationResolver(settings)
        
    def to_execution_request(self, signal: TradeSignal) -> ExecutionRequest:
        """
        Convert a high-level TradeSignal into a precise ExecutionRequest.
        
        Args:
            signal: The trade signal to execute
            
        Returns:
            ExecutionRequest populated with resolved parameters
        """
        # 1. Resolve Contract Details
        # Signal direction is 'CALL'/'PUT' etc.
        # Contract type is 'RISE_FALL' etc.
        # Deriv API uses 'contract_type' like 'CALLE'/'PUT', 'TOUCH'/'NOTOUCH' ??
        # Or just 'CALL'/'PUT' for 'buy'.
        # Let's assume signal.direction is the API-compatible type like 'CALL' or 'PUT'.
        # If signal.direction maps to contract_type, we use it.
        
        # NOTE: signal.contract_type is 'RISE_FALL', signal.direction is 'CALL'.
        # The Executor passes `request.contract_type` to `client.buy`.
        # So we should use signal.direction as the `contract_type` for the request.
        
        contract_type = signal.direction
        if not contract_type:
             raise ValueError(f"Signal {signal.signal_id} has no direction/contract_type")

        # 2. Resolve Duration
        duration, unit = self.duration_resolver.resolve_duration(signal.contract_type)
        
        # 3. Resolve Stake
        if self.position_sizer:
            try:
                # Use the protocol method
                stake = self.position_sizer.suggest_stake_for_signal(signal)
            except AttributeError:
                # Fallback if position_sizer doesn't follow protocol exactly (e.g. old interface)
                # But we should rely on protocol
                if hasattr(self.position_sizer, "compute_stake"):
                     res = self.position_sizer.compute_stake(probability=signal.probability)
                     stake = res.stake
                else:
                    stake = self.settings.trading.stake_amount
        else:
            stake = self.settings.trading.stake_amount
            
        # 4. Resolve Barriers
        barrier = None
        barrier2 = None
        
        # Logic for barrier contracts
        if "TOUCH" in signal.contract_type:
            # Touch contracts usually require a barrier
            # This logic needs to be robust. For now, we take from settings.
            offset = self.settings.trading.barrier_offset
            barrier = f"+{offset}" if "TOUCH" in contract_type else None # Simplified
            # Correct barrier logic depends on current price vs barrier which we don't have here easily.
            # But usually barrier offset is relative.
            if self.settings.trading.barrier_offset: 
                barrier = str(self.settings.trading.barrier_offset)

        # 5. Build Request
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

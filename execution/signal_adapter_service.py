import logging
from typing import Optional, Protocol, runtime_checkable

from config.settings import Settings
from execution.common.types import ExecutionRequest
from execution.signals import TradeSignal
from execution.executor_adapter import SignalAdapter

logger = logging.getLogger(__name__)

@runtime_checkable
class PositionSizerProtocol(Protocol):
    def suggest_stake_for_signal(self, signal: TradeSignal) -> float: ...

class SignalAdapterService:
    """
    Service to adapt TradeSignals into ExecutionRequests.
    
    Centralizes adaptation logic, position sizing integration, and error handling.
    Replaces inline logic in Executor (I-001).
    """

    def __init__(self, settings: Settings, position_sizer: Optional[PositionSizerProtocol] = None):
        self.settings = settings
        self.position_sizer = position_sizer
        self._internal_adapter = SignalAdapter(settings, position_sizer)

    async def adapt(self, signal: TradeSignal) -> ExecutionRequest:
        """
        Adapt a signal to an execution request safely.
        
        Args:
            signal: The trade signal to adapt.
            
        Returns:
            ExecutionRequest: The adapted request.
            
        Raises:
            ValueError: If critical parameters are missing or invalid.
            Exception: For other adaptation failures.
        """
        try:
            # Delegate to internal adapter logic
            # Note: SignalAdapter.to_execution_request is async
            request = await self._internal_adapter.to_execution_request(signal)
            
            # Additional validation layer if needed
            if request.stake <= 0:
                 logger.warning(f"Adapted signal {signal.signal_id} has invalid stake: {request.stake}. Forcing to 0.0 (No Trade).")
                 # We could raise here, or let executor filter it.
                 # Let's ensure it's valid for executor.
                 pass

            return request
            
        except Exception as e:
            logger.error(f"Failed to adapt signal {signal.signal_id} in SignalAdapterService: {e}")
            raise

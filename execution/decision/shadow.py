import asyncio
import logging
from typing import Any, Set
from datetime import datetime

import numpy as np

from config.settings import Settings
from execution.contract_params import ContractDurationResolver
from execution.shadow_ops import fire_shadow_trade_task
from execution.shadow_store import ShadowTradeStore
from execution.sqlite_shadow_store import SQLiteShadowStore
from execution.signals import TradeSignal
from execution.regime import RegimeAssessmentProtocol

logger = logging.getLogger(__name__)

class ShadowDispatcher:
    """
    Handles asynchronous dispatch of shadow trades to the storage backend.
    
    Extracts shadow logging logic from DecisionEngine to improve modularity.
    """
    
    def __init__(
        self,
        settings: Settings,
        shadow_store: ShadowTradeStore | SQLiteShadowStore | None = None,
        model_version: str = "unknown",
        execution_mode: str = "REAL",
    ):
        self.settings = settings
        self.shadow_store = shadow_store
        self.model_version = model_version
        self.execution_mode = execution_mode
        
        # Helper for efficient duration resolution
        self.duration_resolver = ContractDurationResolver(settings)
        
        # Track pending background tasks
        self._pending_tasks: Set[asyncio.Task] = set()

    def dispatch(
        self,
        signals: list[TradeSignal],
        reconstruction_error: float,
        regime_assessment: RegimeAssessmentProtocol,
        entry_price: float,
        market_data: dict[str, Any] | None = None,
        tick_window: np.ndarray | None = None,
        candle_window: np.ndarray | None = None,
    ) -> None:
        """Fire and forget shadow trade tasks."""
        if not self.shadow_store or not signals:
            return

        regime_state = self._get_regime_state_string(regime_assessment)
        is_vetoed = regime_assessment.is_vetoed()
        
        for sig in signals:
            fire_shadow_trade_task(
                pending_tasks=self._pending_tasks,
                store=self.shadow_store,
                settings=self.settings,
                resolver=self.duration_resolver,
                model_version=self.model_version,
                execution_mode=self.execution_mode,
                signal=sig,
                reconstruction_error=reconstruction_error,
                regime_state=regime_state,
                entry_price=entry_price,
                regime_vetoed=is_vetoed,
                metadata=market_data,
                tick_window=tick_window,
                candle_window=candle_window,
            )

    def _get_regime_state_string(self, assessment: RegimeAssessmentProtocol) -> str:
        """Extract regime state string."""
        if hasattr(assessment, "trust_state") and hasattr(assessment.trust_state, "value"):
            return str(assessment.trust_state.value)
        if assessment.is_vetoed():
            return "veto"
        elif assessment.requires_caution():
            return "caution"
        else:
            return "trusted"

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Gracefully shutdown and flush pending tasks."""
        if not self._pending_tasks:
            return

        logger.info(f"ShadowDispatcher: Waiting for {len(self._pending_tasks)} shadow tasks to flush...")
        done, pending = await asyncio.wait(self._pending_tasks, timeout=timeout)
        
        if pending:
            logger.warning(f"ShadowDispatcher shutdown timed out with {len(pending)} tasks remaining.")
            for task in pending:
                task.cancel()
        else:
            logger.info("ShadowDispatcher: All shadow tasks flushed successfully.")

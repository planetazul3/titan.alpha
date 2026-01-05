"""
Shadow Trade Operations.

Handles the asynchronous persistence of shadow trades (Fire-and-Forget).
"""

import asyncio
import logging
from typing import Any

import numpy as np

from config.settings import Settings
from data.features import FEATURE_SCHEMA_VERSION
from execution.contract_params import ContractDurationResolver
from execution.shadow_store import ShadowTradeRecord, ShadowTradeStore
from execution.signals import TradeSignal
from observability.shadow_logging import shadow_trade_logger

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from execution.sqlite_shadow_store import SQLiteShadowStore


def extract_barrier_value(metadata: dict[str, Any], key: str) -> float | None:
    """Safely extract and parse barrier value from metadata."""
    if key not in metadata:
        return None
    
    value = metadata[key]
    if value is None:
        return None
        
    try:
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            clean_value = value.strip().replace("+", "")
            if not clean_value:
                return None
            return float(clean_value)
            
        return None
    except (ValueError, TypeError):
         logger.warning(f"Failed to parse barrier value for {key}: {value}")
         return None


async def do_store_shadow_trade(
    store: "ShadowTradeStore | SQLiteShadowStore",
    settings: Settings,
    resolver: ContractDurationResolver,
    model_version: str,
    execution_mode: str,
    signal: TradeSignal,
    reconstruction_error: float,
    regime_state: str,
    entry_price: float,
    tick_window: np.ndarray | None = None,
    candle_window: np.ndarray | None = None,
    regime_vetoed: bool = False,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Internal implementation of shadow trade storage."""
    
    # I01 Fix: Only persist shadow trades exceeding minimum learning threshold
    min_prob = settings.shadow_trade.min_probability_track
    if signal.probability < min_prob:
        return

    duration_minutes, _ = resolver.resolve_duration(signal.contract_type)

    record = ShadowTradeRecord.create(
        contract_type=signal.contract_type,
        direction=signal.direction or "",
        probability=signal.probability,
        entry_price=entry_price,
        reconstruction_error=reconstruction_error,
        regime_state=regime_state,
        tick_window=tick_window,
        candle_window=candle_window,
        model_version=model_version,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        barrier_level=extract_barrier_value(signal.metadata, "barrier"),
        barrier2_level=extract_barrier_value(signal.metadata, "barrier2"),
        duration_minutes=duration_minutes,
        metadata={
            "signal_type": signal.signal_type,
            "regime_vetoed": regime_vetoed,
            "execution_mode": execution_mode,
            **(metadata or {}),
            **signal.metadata,
        },
    )

    await store.append_async(record)
    shadow_trade_logger.log_stored(record.trade_id)
    
    shadow_trade_logger.log_created(
        trade_id=record.trade_id,
        contract_type=record.contract_type,
        direction=record.direction,
        probability=record.probability,
        metadata={"duration_minutes": duration_minutes}
    )


def fire_shadow_trade_task(
    pending_tasks: set,
    store: "ShadowTradeStore | SQLiteShadowStore | None",
    settings: Settings,
    resolver: ContractDurationResolver,
    model_version: str,
    execution_mode: str,
    signal: TradeSignal,
    reconstruction_error: float,
    regime_state: str,
    entry_price: float,
    tick_window: np.ndarray | None = None,
    candle_window: np.ndarray | None = None,
    regime_vetoed: bool = False,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Store a shadow trade in the background (Fire-and-Forget).
    """
    if not store:
        return

    task = asyncio.create_task(
        do_store_shadow_trade(
            store=store,
            settings=settings,
            resolver=resolver,
            model_version=model_version,
            execution_mode=execution_mode,
            signal=signal,
            reconstruction_error=reconstruction_error,
            regime_state=regime_state,
            entry_price=entry_price,
            tick_window=tick_window,
            candle_window=candle_window,
            regime_vetoed=regime_vetoed,
            metadata=metadata,
        )
    )
    pending_tasks.add(task)
    task.add_done_callback(pending_tasks.discard)

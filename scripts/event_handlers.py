"""
Market event handlers for tick and candle processing.

Implements production patterns:
- Circuit breaker integration for fault tolerance
- H6 staleness veto for data freshness
- Explicit state management via context
- Inference cooldown to prevent overtrading

Reference: docs/plans/live_script_refactoring.md Section 4.3
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts.context import LiveTradingContext
    from data.types import CandleEvent, TickEvent

logger = logging.getLogger(__name__)

# Inference cooldown in seconds
INFERENCE_COOLDOWN_SECONDS = 30.0


class MarketEventHandler:
    """
    Handles market data events with safety checks.
    
    Design patterns:
    - Circuit breaker pattern for fault tolerance
    - H6 staleness veto for data freshness
    - Explicit state management via context
    
    Safety Requirements:
    - H6: Staleness Veto - reject stale data
    - C01: Circuit breaker prevents processing when open
    """
    
    def __init__(self, context: LiveTradingContext):
        """
        Initialize event handler.
        
        Args:
            context: LiveTradingContext with dependencies
        """
        self.context = context
        self._first_tick = True
        self._first_candle = True
        self._last_inference_time = 0.0
        self._last_tick_log_count = 0
    
    async def handle_tick(self, tick_event: TickEvent) -> bool:
        """
        Process tick event with circuit breaker check.
        
        Args:
            tick_event: Normalized tick event
            
        Returns:
            True if tick was processed, False if skipped
        """
        from scripts.console_utils import console_log
        
        # Update liveness immediately (H8)
        self.context.last_tick_time = datetime.now(timezone.utc)
        
        # Check shutdown
        if self.context.shutdown_event.is_set():
            return False
        
        # Check circuit breaker
        if self.context.client and hasattr(self.context.client, 'circuit_breaker'):
            if self.context.client.circuit_breaker.state.value == "open":
                logger.warning("Circuit breaker open, skipping tick processing")
                return False
        
        try:
            logger.debug(f"Tick received: {tick_event}")
            
            # Add to buffer
            if self.context.buffer:
                self.context.buffer.append_tick(tick_event.price)
            
            tick_count = self.context.increment_tick_count()
            
            # Log first tick and then every 100
            if self._first_tick:
                console_log(f"First LIVE tick received: {tick_event.price:.2f}", "SUCCESS")
                self._first_tick = False
            elif tick_count - self._last_tick_log_count >= 100:
                console_log(f"Received {tick_count} ticks (latest: {tick_event.price:.2f})", "DATA")
                self._last_tick_log_count = tick_count
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing tick: {e}", exc_info=True)
            return False
    
    async def handle_candle(self, candle_event: CandleEvent) -> bool:
        """
        Process candle event with staleness veto (H6).
        
        Safety checks:
        - H6: Staleness veto - reject stale data
        - Circuit breaker: prevent cascading failures
        - Shutdown signal: graceful termination
        
        Args:
            candle_event: Normalized candle event
            
        Returns:
            True if candle processed successfully
        """
        from scripts.console_utils import console_log
        
        if not candle_event:
            return False
        
        # Check shutdown
        if self.context.shutdown_event.is_set():
            logger.info("Shutdown signal received, skipping candle processing")
            return False
        
        # Check circuit breaker
        if self.context.client and hasattr(self.context.client, 'circuit_breaker'):
            if self.context.client.circuit_breaker.state.value == "open":
                logger.warning("Circuit breaker open, skipping candle processing")
                return False
        
        start_time = datetime.now()
        
        try:
            # ═══════════════════════════════════════════════════════════════════
            # H6 Staleness Veto - CRITICAL SAFETY CHECK
            # ═══════════════════════════════════════════════════════════════════
            now_utc = datetime.now(timezone.utc)
            candle_time = candle_event.timestamp
            if candle_time.tzinfo is None:
                candle_time = candle_time.replace(tzinfo=timezone.utc)
            
            latency = (now_utc - candle_time).total_seconds()
            stale_threshold = self.context.settings.heartbeat.stale_data_threshold_seconds
            
            if latency > stale_threshold:
                logger.warning(
                    f"[H6 VETO] Skipping stale candle (closed {latency:.1f}s ago). "
                    f"Threshold: {stale_threshold:.1f}s"
                )
                console_log(f"H6 VETO: Stale candle ({latency:.1f}s lag)", "WARN")
                return False
            
            # ═══════════════════════════════════════════════════════════════════
            # Buffer Update
            # ═══════════════════════════════════════════════════════════════════
            is_new_candle = False
            if self.context.buffer:
                is_new_candle = self.context.buffer.update_candle(candle_event)
            
            self.context.increment_candle_count()
            
            # First candle log
            if self._first_candle:
                console_log(
                    f"First LIVE candle (O:{candle_event.open:.2f} H:{candle_event.high:.2f} "
                    f"L:{candle_event.low:.2f} C:{candle_event.close:.2f})",
                    "SUCCESS"
                )
                self._first_candle = False
            
            process_time = (datetime.now() - start_time).total_seconds()
            
            # ═══════════════════════════════════════════════════════════════════
            # Shadow Resolution (every candle close)
            # ═══════════════════════════════════════════════════════════════════
            if is_new_candle and self.context.resolver:
                resolved = await self._resolve_shadow_trades(candle_event)
                if resolved > 0:
                    console_log(f"🎯 Resolved {resolved} shadow trade(s)", "SUCCESS")
            
            # ═══════════════════════════════════════════════════════════════════
            # Inference Trigger (with cooldown)
            # ═══════════════════════════════════════════════════════════════════
            if is_new_candle and self.context.buffer and self.context.buffer.is_ready():
                time_since_last = time.time() - self._last_inference_time
                
                if time_since_last < INFERENCE_COOLDOWN_SECONDS:
                    logger.debug(f"[COOLDOWN] Skipping inference ({time_since_last:.1f}s < 30s)")
                else:
                    console_log(
                        f"Candle closed @ {candle_event.close:.2f} - Running inference... "
                        f"(latency: {latency:.1f}s)",
                        "MODEL"
                    )
                    
                    inference_ok = await self._run_inference_cycle()
                    if inference_ok:
                        self._last_inference_time = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing candle: {e}", exc_info=True)
            return False
    
    async def _resolve_shadow_trades(self, candle_event: CandleEvent) -> int:
        """
        Resolve shadow trades for the candle.
        
        Returns:
            Number of resolved trades
        """
        if self.context.resolver is None:
            return 0
        
        try:
            resolved = await self.context.resolver.resolve_trades(
                current_price=candle_event.close,
                current_time=candle_event.timestamp,
                high_price=candle_event.high,
                low_price=candle_event.low,
            )
            
            if resolved > 0:
                logger.info(f"Resolved {resolved} shadow trades this candle")
            
            return resolved
            
        except Exception as e:
            logger.error(f"Shadow resolution failed: {e}")
            return 0
    
    async def _run_inference_cycle(self) -> bool:
        """
        Run inference and execute trades.
        
        Returns:
            True if inference completed successfully
        """
        if self.context.orchestrator is None or self.context.buffer is None:
            return False
        
        try:
            snapshot = self.context.buffer.get_snapshot()
            
            result = await self.context.orchestrator.run_cycle(
                market_snapshot=snapshot,
                challengers=[]  # Challengers handled separately
            )
            
            self.context.increment_inference_count()
            
            # Log high reconstruction error
            if result is not None and result > 0.3:
                from scripts.console_utils import console_log
                console_log(f"⚠️ High reconstruction error: {result:.3f}", "WARN")
            
            return True
            
        except asyncio.CancelledError:
            logger.info("Inference cycle cancelled")
            raise
        except Exception as e:
            logger.error(f"Inference cycle failed: {e}", exc_info=True)
            if self.context.metrics:
                self.context.metrics.record_error("inference_failure")
            return False

"""
Heartbeat task with hot-reload support.

Extracts heartbeat() from live.py per ADR-009.

Critical Safety Requirements:
- [M12] Atomic hot-reload: validate schema BEFORE swap (see _check_hot_reload L213)
- [C-06] Exponential backoff: 60s → 1800s on failures (see L257)
- [H6] Staleness detection: alerts when no ticks received (see L97)

Backoff Formula: min(BASE_BACKOFF * 2^(failures-1), MAX_BACKOFF)
- BASE_BACKOFF_SECONDS = 60
- MAX_BACKOFF_SECONDS = 1800 (30 minutes)

Implementation: 2026-01-07 (ADR-009)
Reference: docs/adr/009-live-script-modularization.md
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from scripts.context import LiveTradingContext

logger = logging.getLogger(__name__)

# Constants for backoff
BASE_BACKOFF_SECONDS = 60  # 60s, 120s, 240s... 
MAX_BACKOFF_SECONDS = 1800  # 30 minutes max


async def heartbeat_task(context: LiveTradingContext) -> None:
    """
    Periodic heartbeat with hot-reload and staleness monitoring.
    
    Critical safety requirements preserved:
    - M12: Atomic model swap with validation
    - C06: Exponential backoff for failures
    - H6: Staleness detection
    
    Args:
        context: LiveTradingContext with all dependencies
    """
    # Lazy imports for components that may not be in context yet
    from observability.shadow_metrics import ShadowTradeMetrics
    from scripts.console_utils import console_log
    
    settings = context.settings
    heartbeat_interval = settings.heartbeat.interval_seconds
    stale_threshold = settings.heartbeat.stale_data_threshold_seconds
    
    # Shadow metrics cache
    shadow_metrics_cache = ShadowTradeMetrics()
    last_metrics_update = 0.0
    metrics_update_interval = 300  # 5 minutes
    
    logger.info(f"Heartbeat task started (interval={heartbeat_interval}s)")
    
    while not context.shutdown_event.is_set():
        try:
            # Wait for interval or shutdown
            try:
                await asyncio.wait_for(
                    context.shutdown_event.wait(),
                    timeout=heartbeat_interval
                )
                logger.info("Heartbeat: Shutdown event received")
                break
            except asyncio.TimeoutError:
                pass  # Normal timeout, continue with heartbeat
            
            now = datetime.now(timezone.utc)
            now_ts = time.time()
            stale_seconds = (now - context.last_tick_time).total_seconds()
            
            # ═══════════════════════════════════════════════════════════════════
            # Update Shadow Metrics (Periodic)
            # ═══════════════════════════════════════════════════════════════════
            if now_ts - last_metrics_update > metrics_update_interval:
                try:
                    await _update_shadow_metrics(context, shadow_metrics_cache)
                    last_metrics_update = now_ts
                except Exception as e:
                    logger.error(f"Failed to update shadow metrics: {e}")
            
            # ═══════════════════════════════════════════════════════════════════
            # Hot-Reload Check (M12 Critical)
            # ═══════════════════════════════════════════════════════════════════
            if context.checkpoint_path and context.checkpoint_path.exists():
                await _check_hot_reload(context)
            
            # ═══════════════════════════════════════════════════════════════════
            # Staleness Check
            # ═══════════════════════════════════════════════════════════════════
            if stale_seconds > stale_threshold:
                console_log(f"WARNING: No ticks for {stale_seconds:.1f}s - possible connection issue", "WARN")
                logger.warning(f"[STALE] No ticks received for {stale_seconds:.1f}s")
            
            # ═══════════════════════════════════════════════════════════════════
            # Heartbeat Logging
            # ═══════════════════════════════════════════════════════════════════
            logger.info(
                f"[HEARTBEAT] ticks={context.tick_count}, candles={context.candle_count}, "
                f"inferences={context.inference_count}, stale_sec={stale_seconds:.1f}, "
                f"shadow_win_rate={shadow_metrics_cache.win_rate:.3f}"
            )
            
        except asyncio.CancelledError:
            logger.info("Heartbeat task cancelled")
            raise
        except Exception as e:
            logger.exception(f"Heartbeat error: {e}")
            await asyncio.sleep(5)
    
    logger.info("Heartbeat task exiting")


async def _update_shadow_metrics(
    context: LiveTradingContext, 
    metrics_cache: Any
) -> None:
    """Update shadow trade metrics in executor thread."""
    if context.shadow_store is None:
        return
    
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        metrics_cache.update_from_store,
        context.shadow_store
    )


async def _check_hot_reload(context: LiveTradingContext) -> None:
    """
    Check for new checkpoint and perform atomic hot-reload.
    
    M12 Safety: Atomic model swap with validation
    C06 Safety: Exponential backoff for failures
    """
    from scripts.console_utils import console_log
    from utils.bootstrap import validate_model_compatibility
    from observability.alerting import get_alert_manager, AlertLevel
    
    checkpoint_path = context.checkpoint_path
    if checkpoint_path is None:
        return
    
    hot_reload_state = context.hot_reload_state
    
    try:
        current_mtime = checkpoint_path.stat().st_mtime
        
        # Initialize mtime if first check
        if hot_reload_state.last_ckpt_mtime == 0:
            hot_reload_state.last_ckpt_mtime = current_mtime
            return
        
        # Check backoff period
        now = time.time()
        if now < hot_reload_state.backoff_until:
            return  # Still in backoff
        
        # Check if checkpoint updated
        if current_mtime <= hot_reload_state.last_ckpt_mtime:
            return  # No update
        
        console_log(f"Checkpoint update detected! Reloading... ({checkpoint_path.name})", "MODEL")
        logger.info(f"[HOT-RELOAD] Checkpoint modified at {datetime.fromtimestamp(current_mtime)}")
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 1: Load checkpoint to temporary (Blue-Green pattern)
        # ═══════════════════════════════════════════════════════════════════
        loop = asyncio.get_running_loop()
        new_checkpoint = await loop.run_in_executor(
            None,
            lambda: torch.load(checkpoint_path, map_location=context.device, weights_only=True)
        )
        
        # Find state dict key
        state_dict_key = None
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in new_checkpoint:
                state_dict_key = key
                break
        
        if state_dict_key is None:
            logger.error(f"[HOT-RELOAD] No valid state_dict key found. Available: {list(new_checkpoint.keys())}")
            console_log("Hot-reload aborted: no valid state_dict key", "WARN")
            return
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 2: Validate Architecture Match
        # ═══════════════════════════════════════════════════════════════════
        if context.model is None:
            logger.error("[HOT-RELOAD] No model in context")
            return
        
        current_keys = set(context.model.state_dict().keys())
        new_keys = set(new_checkpoint[state_dict_key].keys())
        
        missing = current_keys - new_keys
        unexpected = new_keys - current_keys
        
        if missing or unexpected:
            logger.warning(f"[HOT-RELOAD] Mismatch! Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            console_log("Hot-reload aborted: architecture mismatch", "WARN")
            return
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 3: Validate Schema Compatibility (M12 Critical)
        # ═══════════════════════════════════════════════════════════════════
        try:
            validate_model_compatibility(
                new_checkpoint,
                context.settings,
                feature_builder=context.feature_builder
            )
        except RuntimeError as schema_err:
            logger.error(f"[HOT-RELOAD] Schema validation failed: {schema_err}")
            console_log(f"Hot-reload aborted: {schema_err}", "ERROR")
            return
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 4: Atomic Swap (after all validations pass)
        # ═══════════════════════════════════════════════════════════════════
        context.model.load_state_dict(new_checkpoint[state_dict_key], strict=False)
        context.model.eval()
        
        # Update metadata
        new_ver = "unknown"
        if "manifest" in new_checkpoint:
            new_ver = new_checkpoint["manifest"].get("model_version", "unknown")
        if context.engine and hasattr(context.engine, 'model_version'):
            context.engine.model_version = new_ver
        
        console_log(f"Model reloaded: v{new_ver}", "SUCCESS")
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 5: Update State & Reset Monitors
        # ═══════════════════════════════════════════════════════════════════
        hot_reload_state.last_ckpt_mtime = current_mtime
        hot_reload_state.consecutive_failures = 0
        
        if context.calibration_monitor:
            context.calibration_monitor.reset()
        
        logger.info("[HOT-RELOAD] Success. Calibration monitor reset.")
        
    except Exception as e:
        # Exponential backoff
        hot_reload_state.consecutive_failures += 1
        failures = hot_reload_state.consecutive_failures
        
        backoff_seconds = min(BASE_BACKOFF_SECONDS * (2 ** (failures - 1)), MAX_BACKOFF_SECONDS)
        hot_reload_state.backoff_until = time.time() + backoff_seconds
        
        logger.error(f"[HOT-RELOAD] Failed (attempt {failures}): {e}. Next retry in {backoff_seconds}s")
        console_log(f"Hot-reload failed: {e}", "ERROR")
        
        # Alert after 3 consecutive failures
        if failures >= 3:
            alert_manager = get_alert_manager()
            alert_manager.trigger(
                "hot_reload_persistent_failure",
                f"Model hot-reload failed {failures} times for {checkpoint_path.name}",
                AlertLevel.WARNING
            )

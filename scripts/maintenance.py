"""
Maintenance task for log cleanup and database pruning.

Extracts maintenance_task() from live.py per ADR-009.

Production Patterns:
- Async file I/O via run_in_executor (non-blocking)
- Graceful shutdown support via context.shutdown_event
- Default 24-hour interval (configurable via settings)

Operations:
- Log cleanup: removes files older than settings.log_retention_days
- DB pruning: removes shadow records older than settings.db_retention_days

Implementation: 2026-01-07 (ADR-009)
Reference: docs/plans/live_script_refactoring.md Section 4.4
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.context import LiveTradingContext

logger = logging.getLogger(__name__)

# Default interval: 24 hours
DEFAULT_MAINTENANCE_INTERVAL = 86400


async def maintenance_task(context: LiveTradingContext) -> None:
    """
    Background maintenance for logs and database pruning.
    
    Runs every 24 hours (or settings.maintenance_interval if defined).
    Uses run_in_executor for blocking operations.
    
    Args:
        context: LiveTradingContext with settings and shadow_store
    """
    settings = context.settings
    interval = getattr(settings, 'maintenance_interval', DEFAULT_MAINTENANCE_INTERVAL)
    
    logger.info(f"Maintenance task started (interval={interval}s)")
    
    while not context.shutdown_event.is_set():
        try:
            # Wait for interval or shutdown
            try:
                await asyncio.wait_for(
                    context.shutdown_event.wait(),
                    timeout=interval
                )
                logger.info("Maintenance: Shutdown event received")
                break
            except asyncio.TimeoutError:
                pass  # Normal timeout, run maintenance
            
            logger.info("[MAINTENANCE] Starting background maintenance...")
            
            # ═══════════════════════════════════════════════════════════════════
            # 1. Log Cleanup
            # ═══════════════════════════════════════════════════════════════════
            await _cleanup_logs(context)
            
            # ═══════════════════════════════════════════════════════════════════
            # 2. Database Pruning
            # ═══════════════════════════════════════════════════════════════════
            await _prune_database(context)
            
            logger.info("[MAINTENANCE] Background maintenance completed.")
            
        except asyncio.CancelledError:
            logger.info("Maintenance task cancelled")
            raise
        except Exception as e:
            logger.exception(f"[MAINTENANCE] Unexpected error: {e}")
            await asyncio.sleep(60)  # Brief pause before next attempt
    
    logger.info("Maintenance task exiting")


async def _cleanup_logs(context: LiveTradingContext) -> int:
    """
    Clean up old log files.
    
    Returns:
        Number of deleted log files
    """
    try:
        from config.logging_config import cleanup_logs
        
        # Get log directory from settings or default
        log_dir = getattr(context.settings.system, 'log_dir', None)
        if log_dir is None:
            return 0
        
        log_dir = Path(log_dir)
        if not log_dir.exists():
            return 0
        
        retention_days = context.settings.system.log_retention_days
        
        # Run in executor to avoid blocking event loop
        loop = asyncio.get_running_loop()
        deleted_count = await loop.run_in_executor(
            None,
            lambda: cleanup_logs(log_dir, retention_days=retention_days)
        )
        
        if deleted_count > 0:
            logger.info(f"[MAINTENANCE] Deleted {deleted_count} old log files")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"[MAINTENANCE] Log cleanup failed: {e}")
        return 0


async def _prune_database(context: LiveTradingContext) -> int:
    """
    Prune old records from shadow database.
    
    Returns:
        Number of pruned records
    """
    if context.shadow_store is None:
        return 0
    
    try:
        retention_days = context.settings.system.db_retention_days
        
        # Run in executor as VACUUM locks DB
        loop = asyncio.get_running_loop()
        pruned_count = await loop.run_in_executor(
            None,
            lambda: context.shadow_store.prune(retention_days=retention_days)
        )
        
        if pruned_count > 0:
            logger.info(f"[MAINTENANCE] Pruned {pruned_count} old shadow records")
        
        return pruned_count
        
    except Exception as e:
        logger.error(f"[MAINTENANCE] DB pruning failed: {e}")
        return 0

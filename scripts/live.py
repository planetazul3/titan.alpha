#!/usr/bin/env python3
"""
Unified live trading script (x.titan).

Thin orchestration layer after ADR-009 modularization (974 → 599 lines).

Architecture:
1.  **Bootstrap**: Load settings, model, create trading stack
2.  **Context**: Initialize LiveTradingContext with all dependencies
3.  **Sync**: Fetch history, populate buffer, transition to LIVE mode
4.  **TaskGroup**: Launch 4 extracted modules concurrently:
    - tick_processor (event_handlers.py)
    - candle_processor (event_handlers.py)
    - heartbeat_task (heartbeat.py)
    - maintenance_task (maintenance.py)

Safety Requirements (see ARCHITECTURE_SSOT.md):
- [H6] Staleness veto in candle_processor
- [M12] Atomic hot-reload in heartbeat_task
- [C-01] Circuit breaker sync in event_handlers
- [C-06] Exponential backoff in heartbeat_task

Usage:
    python scripts/live.py                    # Run live trading
    python scripts/live.py --test             # Verify connection only
    python scripts/live.py --shadow-only      # Shadow mode (no real trades)
    python scripts/live.py --checkpoint best  # Load specific checkpoint

Implementation: 2026-01-07 (ADR-009)
"""

import argparse
import asyncio
from typing import Any
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from dotenv import load_dotenv

load_dotenv()


from config.settings import load_settings
from data.ingestion.deriv_adapter import DerivEventAdapter
from data.buffer import MarketDataBuffer
from observability.calibration import CalibrationMonitor
from execution.sqlite_shadow_store import SQLiteShadowStore  # Full context capture
from models.core import DerivOmniModel
from observability import TradingMetrics
from observability.dashboard import (
    SystemHealthMonitor,
    create_executor_health_checker,
    create_model_health_checker,
)
from observability.model_health import ModelHealthMonitor
from execution.shadow_resolution import ShadowTradeResolver
from execution.real_trade_tracker import RealTradeTracker
from utils.bootstrap import create_trading_stack, create_challenger_stack
# NEW IMPORTS
from execution.synchronizer import StartupSynchronizer
from execution.orchestrator import InferenceOrchestrator

# Phase 5 Integration: Extracted Modules
from scripts.context import LiveTradingContext, HotReloadState
from scripts.heartbeat import heartbeat_task
from scripts.maintenance import maintenance_task as maintenance_task_module
from scripts.event_handlers import MarketEventHandler, tick_processor, candle_processor

# I02: Checkpoint verification utility
from tools.verify_checkpoint import verify_checkpoint

# R01: OpenTelemetry tracing
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    TRACING_ENABLED = True
    tracer: Any = trace.get_tracer(__name__)
except ImportError:
    TRACING_ENABLED = False
    tracer = None


from config.logging_config import setup_logging

# Configure logging to both console and file
log_file = setup_logging(script_name="live_trading", level="INFO")
logger = logging.getLogger(__name__)
log_dir = log_file.parent if log_file else None

# Import shared console logging utilities
from scripts.console_utils import console_log


# IMPL: CalibrationMonitor moved to observability/calibration.py


async def run_live_trading(args):
    """Main live trading loop."""
    console_log("=" * 60, "INFO")
    console_log("LIVE TRADING SYSTEM STARTING", "SUCCESS")
    console_log("=" * 60, "INFO")

    settings = load_settings()
    device = settings.get_device()

    console_log(f"Symbol: {settings.trading.symbol} | Device: {device}", "INFO")
    logger.info(f"Starting live trading for {settings.trading.symbol}")
    logger.info(f"Device: {device}")
    
    # Initialize variables for reliable cleanup
    model = None
    client = None
    engine = None
    executor = None
    shadow_store = None

    # M13: Disk Usage Management - DEPRECATED: Moved to background task

    # ══════════════════════════════════════════════════════════════════════
    # STRUCTURED METRICS - for production observability
    # Records: inference latency, trade outcomes, regime assessments, P&L
    # ══════════════════════════════════════════════════════════════════════
    console_log("Initializing metrics collector...", "WAIT")
    metrics = TradingMetrics(enable_prometheus=settings.observability.enable_prometheus)
    
    # Initialize AlertManager
    from observability.alerting import get_alert_manager, AlertLevel, LogAlertChannel
    alert_manager = get_alert_manager()
    # Configure suppression based on settings
    # Configure suppression based on settings
    alert_manager.set_suppression_interval(settings.observability.alert_suppression_interval)
    
    console_log(f"Alert Manager active (suppression: {settings.observability.alert_suppression_interval}s)", "SUCCESS")
    console_log("Metrics collector ready", "SUCCESS")
    logger.info(
        f"Metrics collector initialized (prometheus={'enabled' if metrics.use_prometheus else 'disabled'})"
    )

    # Initialize System and Model Monitors
    system_monitor = SystemHealthMonitor()
    model_monitor = ModelHealthMonitor(accuracy_window=100)
    system_monitor.register_component("model", create_model_health_checker(model_monitor))

    # Determine checkpoint to load (same logic)
    checkpoint_name = args.checkpoint
    checkpoint_path = None
    if not checkpoint_name:
        default_ckpt = Path(args.checkpoint_dir) / "best_model.pt"
        if default_ckpt.exists():
            checkpoint_path = default_ckpt
            logger.info("No checkpoint specified, auto-selecting 'best_model'")
        else:
            if not args.test and not args.skip_checkpoint_verify:
                 logger.critical("FATAL: No checkpoint found and 'best_model.pt' missing.")
                 return 1
            logger.warning("No Checkpoint Mode: Model initialized with RANDOM weights.")
    else:
        checkpoint_path = Path(args.checkpoint_dir) / f"{checkpoint_name}.pt"
        
    # BOOTSTRAP: Centralized Initialization
    # Replaces manual creation of Model, Client, Engine, etc.
    try:
        stack = create_trading_stack(
            settings, 
            checkpoint_path=checkpoint_path, 
            device=None, # Auto-detect
            verify_ckpt=not getattr(args, 'skip_checkpoint_verify', False)
        )
    except Exception as e:
        logger.critical(f"Failed to bootstrap trading stack: {e}")
        console_log(f"BOOTSTRAP FAILED: {e}", "ERROR")
        return 1
        
    model = stack["model"]
    client = stack["client"]
    engine = stack["engine"]
    shadow_store = stack["shadow_store"]
    feature_builder = stack["feature_builder"]
    regime_veto = stack["regime_veto"]
    device = stack["device"]
    
    console_log(f"Stack initialized (Model v{engine.model_version})", "SUCCESS")
    console_log(f"Shadow Store: SQLite", "SUCCESS")
    logger.info(f"MarketDataBuffer initialized with warmup={settings.data_shapes.warmup_steps}")

    # Buffer abstraction - encapsulates tick/candle management and candle close detection
    # C04: Add warmup steps to buffer sizes to prevent feature flickering
    warmup = settings.data_shapes.warmup_steps
    tick_len = settings.data_shapes.sequence_length_ticks + warmup
    candle_len = settings.data_shapes.sequence_length_candles + warmup
    buffer = MarketDataBuffer(tick_length=tick_len, candle_length=candle_len)
    logger.info(f"MarketDataBuffer initialized: {buffer}")

    # ══════════════════════════════════════════════════════════════════════════
    # CALIBRATION MONITOR: Tracks reconstruction errors for graceful degradation
    # If errors are persistently high, activates shadow-only mode to protect account
    # ══════════════════════════════════════════════════════════════════════════
    calibration_monitor = CalibrationMonitor(
        error_threshold=settings.calibration.error_threshold,
        consecutive_threshold=settings.calibration.consecutive_threshold,
        window_size=settings.calibration.window_size,
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # OPT5: Load Challenger Models (A/B Testing)
    # ══════════════════════════════════════════════════════════════════════════
    challengers = []
    challenger_dir = Path(settings.system.challenger_model_dir)
    if challenger_dir.exists():
        console_log(f"Scanning for challengers in {challenger_dir}...", "INIT")
        for ckpt_path in challenger_dir.glob("*.pt"):
            try:
                c_stack = create_challenger_stack(
                    settings=settings,
                    shadow_store=shadow_store,
                    checkpoint_path=ckpt_path,
                    device=stack["device"]
                )
                challengers.append(c_stack)
                console_log(f"Loaded challenger: {c_stack['version']} ({ckpt_path.name})", "SUCCESS")
            except Exception as e:
                logger.error(f"Failed to load challenger {ckpt_path}: {e}")
                console_log(f"Failed to load challenger {ckpt_path.name}", "ERROR")
    
    # Force shadow-only mode if requested via CLI
    if hasattr(args, "shadow_only") and args.shadow_only:
        calibration_monitor.shadow_only_mode = True
        calibration_monitor.shadow_only_reason = "Manual override via --shadow-only"
        console_log("Status: SHADOW MODE (Real trading disabled)", "WARN")
        logger.warning("[SHADOW-ONLY] Real trading disabled by user argument")

    console_log("Calibration monitor ready", "SUCCESS")
    logger.info(
        f"CalibrationMonitor active: shadow-only triggers after "
        f"{calibration_monitor.consecutive_threshold} consecutive high errors"
    )

    # Executor will be created after connection
    executor = None

    try:
        # Connect to Deriv with retry logic for transient API errors
        console_log("=" * 60, "INFO")
        console_log("CONNECTING TO DERIV API...", "NET")
        max_connect_retries = 5
        for attempt in range(max_connect_retries):
            try:
                console_log(f"Connection attempt {attempt + 1}/{max_connect_retries}...", "WAIT")
                await client.connect()
                balance = await client.get_balance()
                console_log(f"CONNECTED! Account balance: ${balance:.2f}", "SUCCESS")
                logger.info(f"Connected. Balance: ${balance:.2f}")
                break
            except Exception as e:
                if attempt < max_connect_retries - 1:
                    wait_time = min(2 ** (attempt + 1), 30)
                    console_log(f"Connection failed: {e}", "WARN")
                    console_log(f"Retrying in {wait_time}s...", "WAIT")
                    logger.warning(
                        f"[NETWORK] Connection failed (attempt {attempt + 1}/{max_connect_retries}): {e}"
                    )
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    console_log(f"Failed to connect after {max_connect_retries} attempts", "ERROR")
                    logger.error(
                        f"[NETWORK] Failed to connect after {max_connect_retries} attempts"
                    )
                    
                    alert_manager.trigger(
                        "connection_failure_fatal",
                        f"Failed to connect to Deriv API after {max_connect_retries} attempts. System shutting down.",
                        AlertLevel.CRITICAL
                    )
                    raise

        # Initialize Position Sizer based on arguments
        from execution.position_sizer import (
            CompoundingPositionSizer,
            FixedStakeSizer,
            KellyPositionSizer,
            MartingalePositionSizer,
        )

        sizer = None
        base_stake = settings.trading.stake_amount
        
        # Determine strategy from args
        strategy = "fixed"
        if hasattr(args, "compound") and args.compound:
            strategy = "compound"
        elif hasattr(args, "strategy") and args.strategy:
            strategy = args.strategy.lower()

        logger.info(f"Initializing position sizer for strategy: {strategy}")

        if strategy == "compound":
            # Parse multiplier
            multiplier = None # Default to reinvest profit
            if hasattr(args, "x_amount") and args.x_amount:
                if args.x_amount.lower() != "reinvest":
                    try:
                        multiplier = float(args.x_amount.replace("x", ""))
                    except ValueError:
                        logger.warning(f"Invalid x_amount '{args.x_amount}', defaulting to reinvest")
            
            max_streak = int(getattr(args, "winstrikes", 5))
            
            sizer = CompoundingPositionSizer(
                base_stake=base_stake,
                max_consecutive_wins=max_streak,
                streak_multiplier=multiplier
            )
            console_log(f"Strategy: COMPOUNDING (Multiplier: {multiplier or 'Profit'}, MaxStreak: {max_streak})", "INFO")
        
        elif strategy == "martingale":
            multiplier = 2.0
            if hasattr(args, "x_amount") and args.x_amount:
                try:
                    multiplier = float(args.x_amount.replace("x", ""))
                except ValueError:
                    pass
            
            max_streak = int(getattr(args, "winstrikes", 5)) # Using same arg for simplicity or add max_loss_streak
            
            sizer = MartingalePositionSizer(
                base_stake=base_stake,
                multiplier=multiplier,
                max_streak=max_streak
            )
            console_log(f"Strategy: WARNING - MARTINGALE (Multiplier: {multiplier}x, MaxLosses: {max_streak})", "WARN")
            
        elif strategy == "kelly":
            sizer = KellyPositionSizer(
                base_stake=base_stake,
                safety_factor=0.5,
                default_payout_ratio=settings.trading.payout_ratio,
            )
            console_log(f"Strategy: KELLY CRITERION (Safety: 0.5, Payout: {settings.trading.payout_ratio})", "INFO")
            
        else:
            sizer = FixedStakeSizer(stake=base_stake)
            console_log(f"Strategy: FIXED STAKE (${base_stake})", "INFO")

        # Create real trade tracker for outcome tracking via API (enables compounding)
        real_trade_tracker = RealTradeTracker(
            client=client,
            sizer=sizer,
            executor=None,  # Will be set after SafeTradeExecutor is created
            model_monitor=model_monitor,
        )
        console_log("Real trade tracker initialized (API mode)", "SUCCESS")

        # Create Strategy Adapter (IMPORTANT-005)
        # Bridges Decision Engine (Signals) -> Execution (Requests)
        from execution.strategy_adapter import StrategyAdapter
        strategy_adapter = StrategyAdapter(settings, position_sizer=sizer)
        console_log("Strategy Adapter initialized (Sizing + Durations)", "SUCCESS")

        # Create trade executor with SAFETY WRAPPER (production-grade controls)
        # The SafeTradeExecutor provides: rate limiting, P&L caps, kill switch
        console_log("Setting up trade executor with safety controls...", "WAIT")
        # Inject the chosen sizer into the raw executor
        # ID-001 Fix: Pass policy to executor for circuit breaker support
        from execution.executor import DerivTradeExecutor
        # Executor no longer needs sizer, it just executes requests
        raw_executor = DerivTradeExecutor(client, settings, policy=engine.policy)

        from execution.safety import ExecutionSafetyConfig, SafeTradeExecutor
        safety_config = ExecutionSafetyConfig(
            max_trades_per_minute=settings.execution_safety.max_trades_per_minute,
            max_trades_per_minute_per_symbol=settings.execution_safety.max_trades_per_minute_per_symbol,
            max_daily_loss=settings.execution_safety.max_daily_loss,
            max_stake_per_trade=settings.execution_safety.max_stake_per_trade,
            max_retry_attempts=settings.execution_safety.max_retry_attempts,
            retry_base_delay=settings.execution_safety.retry_base_delay,
            kill_switch_enabled=settings.execution_safety.kill_switch_enabled,
        )

        # State file for crash recovery (prevents bypass of risk limits via restarts)
        # Using SQLite for ACID compliance
        # L05: Unified DB path to prevent split-brain state
        safety_state_file = Path(settings.system.system_db_path)

        executor = SafeTradeExecutor(
            inner_executor=raw_executor, 
            config=safety_config, 
            state_file=safety_state_file,
            policy=engine.policy # Register vetoes
        )
        # Register executor with health monitor
        system_monitor.register_component("executor", create_executor_health_checker(executor))
        
        # Connect executor to trade tracker for P&L updates
        real_trade_tracker.executor = executor
        
        console_log("Trade executor ready with safety controls", "SUCCESS")

        logger.info(
            f"Safety wrapper active: rate_limit={safety_config.max_trades_per_minute}/min, "
            f"max_daily_loss=${safety_config.max_daily_loss}, "
            f"max_stake=${safety_config.max_stake_per_trade}"
        )

        # C03: Recover pending trades from previous session
        # Use try-except to prevent blocking startup on recovery failure
        try:
            console_log("Checking for pending trades...", "WAIT")
            recovered_count = await real_trade_tracker.recover_pending_trades()
            if recovered_count > 0:
                console_log(f"Recovered {recovered_count} pending trades", "SUCCESS")
            else:
                logger.info("No pending trades to recover")
        except Exception as e:
            logger.error(f"Failed to recover pending trades: {e}")
            console_log(f"Recovery warning: {e}", "WARN")

        # Test mode - just verify connection
        if args.test:
            console_log("Test mode - connection verified. Exiting.", "SUCCESS")
            logger.info("Test mode - connection verified. Exiting.")
            await client.disconnect()
            return 0

        # ══════════════════════════════════════════════════════════════════════
        # H11: STARTUP SYNCHRONIZATION (Subscribe-then-Fetch)
        # We start subscribing buffer events BEFORE fetching history.
        # This ensures no data gap between history end and live stream start.
        # ══════════════════════════════════════════════════════════════════════
        console_log("=" * 60, "INFO")
        console_log("STARTING LIVE STREAMING (BUFFERING)...", "DATA")
        logger.info("Creating normalized event bus...")
        
        # Initialize Shadow Trade Resolver
        from execution.shadow_resolution import ShadowTradeResolver
        resolver = ShadowTradeResolver(
             shadow_store=shadow_store, 
             client=client, 
             timeframe=settings.trading.timeframe
        )
        
        event_bus = DerivEventAdapter(client)  # Implements MarketEventBus
        symbol = settings.trading.symbol

        # Shared state for heartbeat
        tick_count = 0
        candle_count = 0
        inference_count = 0
        last_tick_time = datetime.now()
        last_tick_log_count = 0  # For refined tick logging
        last_inference_time = 0.0  # CRITICAL RULE 1: Inference Cooldown
        
        # Issue G: Progress counters for per-component staleness detection
        import time as time_module
        last_candle_progress_time = time_module.time()
        last_inference_progress_time = time_module.time()
        
        # C02 (FIXED): Persistent State dictionary with backoff tracking
        hot_reload_state = {
            "last_ckpt_mtime": 0,
            "consecutive_failures": 0,
            "backoff_until": 0.0
        }
        
        # H11: Startup buffers and synchronization
        startup_buffer_ticks: list[float] = []
        startup_buffer_candles: list[Any] = []  # Store full event objects
        startup_complete = asyncio.Event()

        # Observability shared components
        from observability.performance_tracker import PerformanceTracker
        from training.auto_retrain import RetrainingTrigger
        perf_tracker = PerformanceTracker()
        retrain_trigger = RetrainingTrigger()
        first_tick_received = asyncio.Event()
    
    
        # CRITICAL-001/002 (FIXED): Unified Synchronization State Manager
        synchronizer = StartupSynchronizer(buffer)
        
        # Issue 6 (FIXED): Inference Orchestrator
        orchestrator = InferenceOrchestrator(
            model=model,
            engine=engine,
            executor=executor,
            feature_builder=feature_builder,
            device=device,
            settings=settings,
            metrics=metrics,
            calibration_monitor=calibration_monitor,
            trade_tracker=real_trade_tracker,
            strategy_adapter=strategy_adapter,
            tracer=tracer
        )
        
        # Initialize challenger semaphore for Issue 3
        # (Managed internally by Orchestrator now, but ensuring we pass challengers list)

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 5: Create LiveTradingContext for extracted module integration
        # ══════════════════════════════════════════════════════════════════════
        ctx = LiveTradingContext(
            settings=settings,
            device=device,
            checkpoint_path=checkpoint_path,
            client=client,
            model=model,
            engine=engine,
            executor=executor,
            buffer=buffer,
            feature_builder=feature_builder,
            shadow_store=shadow_store,
            orchestrator=orchestrator,
            resolver=resolver,
            trade_tracker=real_trade_tracker,
            strategy_adapter=strategy_adapter,
            calibration_monitor=calibration_monitor,
            system_monitor=system_monitor,
            metrics=metrics,
            # Event Processing (Phase 5)
            synchronizer=synchronizer,
            event_bus=event_bus,
            regime_veto=regime_veto,
            challengers=challengers,
            args=args,
        )
        logger.info("LiveTradingContext initialized for extracted modules")

        # NOTE: process_ticks() and process_candles() functions removed
        # Now using extracted modules: tick_processor(ctx) and candle_processor(ctx)

        # ══════════════════════════════════════════════════════════════════════
        # SYNCHRONIZATION PHASE
        # 1. Fetch historical data to fill buffer BEFORE starting long-running tasks
        # 2. Then start tasks with TaskGroup for deterministic exception handling
        # ══════════════════════════════════════════════════════════════════════
        console_log("Synchronizing with market history...", "WAIT")
        
        # Fetch history before starting background tasks
        hist_ticks = await client.get_historical_ticks(count=settings.data_shapes.sequence_length_ticks)
        hist_candles_raw = await client.get_historical_candles(
            count=settings.data_shapes.sequence_length_candles, 
            interval=60
        )
        
        console_log(f"Fetched history: {len(hist_ticks)} ticks, {len(hist_candles_raw)} candles", "INFO")
        
        # Atomic Transition: populate buffer and switch to LIVE mode
        synchronizer.finalize_startup(hist_ticks, hist_candles_raw)
        startup_complete.set()
        logger.info("Startup synchronization complete. Live processing enabled.")

        # Issue C (FIXED): Use TaskGroup for deterministic exception propagation
        # TaskGroup automatically cancels all tasks when any task raises an exception
        # and re-raises the exception after cleanup.
        # PHASE 5: ALL tasks now use extracted modules with LiveTradingContext
        # CRYTICAL-001 Fix: Python 3.10 Compatibility (TaskGroup is 3.11+)
        # Using explicit task management instead
        
        tasks = []
        try:
            # Create tasks
            ctx.tasks["tick_processor"] = asyncio.create_task(tick_processor(ctx), name="tick_processor")
            ctx.tasks["candle_processor"] = asyncio.create_task(candle_processor(ctx), name="candle_processor")
            ctx.tasks["heartbeat"] = asyncio.create_task(heartbeat_task(ctx), name="heartbeat_monitor")
            ctx.tasks["maintenance"] = asyncio.create_task(maintenance_task(ctx), name="maintenance_worker")
            
            # Collect for waiting
            tasks = list(ctx.tasks.values())
            
            # Wait for all tasks - if any fails with exception, it will raise here if we use gather
            # graceful_shutdown sets shutdown_event, which tasks observe and exit cleanly
            await asyncio.gather(*tasks, return_exceptions=False)
            
        except asyncio.CancelledError:
            logger.info("Main task group cancelled")
        except Exception as e:
            logger.critical(f"Critical task failure: {e}", exc_info=True)
        finally:
            # cleanup is handled by finally block in main(), but we ensure tasks are done
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            if tasks:
                # Wait for cancellation to complete
                await asyncio.gather(*tasks, return_exceptions=True)
            
    except Exception as e:
        logger.critical(f"FATAL ERROR in main loop: {e}", exc_info=True)
        console_log(f"CRITICAL ERROR: {e}", "ERROR")
        return 1
    finally:
        # Cleanup with exception handling to avoid masking errors
        try:
            if client:
                await client.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting client: {e}")
        
        # Log statistics
        try:
            if engine:
                stats = engine.get_statistics()
                logger.info(f"Session statistics: {stats}")
        except Exception as e:
            logger.error(f"Error getting engine statistics: {e}")

        try:
            if executor:
                # Log safety wrapper statistics
                safety_stats = executor.get_safety_statistics()
                logger.info(f"Safety statistics: {safety_stats}")
                # IMPORTANT-001: Graceful shutdown
                await executor.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down executor: {e}")

        console_log("System shutdown complete", "INFO")

    return 0


if __name__ == "__main__":
    from scripts.shutdown_handler import run_async_with_graceful_shutdown
    from scripts.cli import parse_live_trading_args

    args = parse_live_trading_args()
    sys.exit(run_async_with_graceful_shutdown(run_live_trading(args)))



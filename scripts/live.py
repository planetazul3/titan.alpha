#!/usr/bin/env python3
"""
Unified live trading script (x.titan).

Orchestrates the complete trading lifecycle:
1.  **Ingestion**: Connects to Deriv API via `DerivClient` (async/stream).
2.  **Buffering**: `MarketDataBuffer` accumulates ticks/candles for feature engineering.
3.  **Inference**: `DerivOmniModel` (PyTorch) generates probabilities from `FeatureBuilder`.
4.  **Decision**: `DecisionEngine` applies `RegimeVeto` (Absolute Authority) and confidence filters.
5.  **Execution**: `SafeTradeExecutor` enforces "Swiss Cheese" safety (Kill Switch, Circuit Breaker).
6.  **Observability**: Real-time metrics via `PerformanceTracker` and `CalibrationMonitor`.

Usage:
    python scripts/live.py                    # Run live trading (Paper/Real based on .env)
    python scripts/live.py --test             # Verify connection and configuration (No trades)
    python scripts/live.py --checkpoint best  # Load specific model checkpoint
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
from utils.bootstrap import create_trading_stack

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

    # M13: Disk Usage Management - DEPRECATED: Moved to background task

    # ══════════════════════════════════════════════════════════════════════
    # STRUCTURED METRICS - for production observability
    # Records: inference latency, trade outcomes, regime assessments, P&L
    # ══════════════════════════════════════════════════════════════════════
    console_log("Initializing metrics collector...", "WAIT")
    # Records: inference latency, trade outcomes, regime assessments, P&L
    # ══════════════════════════════════════════════════════════════════════
    console_log("Initializing metrics collector...", "WAIT")
    metrics = TradingMetrics(enable_prometheus=settings.observability.enable_prometheus)
    
    # Initialize AlertManager
    from observability.alerting import get_alert_manager, AlertLevel, LogAlertChannel
    alert_manager = get_alert_manager()
    # Configure suppression based on settings
    alert_manager._suppression_interval = settings.observability.alert_suppression_interval
    
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
    challenger_dir = Path("checkpoints/challengers")
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
        last_tick_time = datetime.now()
        last_tick_log_count = 0  # For refined tick logging
        last_inference_time = 0.0  # CRITICAL RULE 1: Inference Cooldown
        
        # C02 (FIXED): Persistent State dictionary
        hot_reload_state = {}
        
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
    
    
        # CRITICAL-001 (FIXED): Unified Synchronization State
        # Replaces broken lock with explicit state flags
        # Default: Buffering active, standard processing disabled
        startup_state = {
            "buffering_active": True,
            "buffer_ticks": [],
            "buffer_candles": []
        }

        async def process_ticks():
            """Process tick events from normalized MarketEventBus."""
            nonlocal tick_count, last_tick_time, last_tick_log_count
            first_tick = True
            async for tick_event in event_bus.subscribe_ticks(symbol):
                # H8 (FIXED): Update liveness immediately on receipt
                last_tick_time = datetime.now()
                try:
                    # Update heartbeat timestamp
                    last_tick_time = datetime.now()
                    
                    logger.debug(f"Tick received: {tick_event}")
                    
                    
                    # CRITICAL-001 (FIXED): Check flag instead of acquiring lock
                    if startup_state["buffering_active"]:
                        startup_state["buffer_ticks"].append(tick_event.price)
                        if not first_tick_received.is_set():
                             first_tick_received.set()
                        return # Skip live processing during startup buffering

                    # Strategy sees TickEvent, not broker-specific message
                    buffer.append_tick(tick_event.price)
                    tick_count += 1
                    
                    # Log first tick and then every 100 ticks
                    if first_tick:
                        console_log(f"First LIVE tick received: {tick_event.price:.2f}", "SUCCESS")
                        first_tick = False
                    elif tick_count - last_tick_log_count >= 100:
                        console_log(
                            f"Received {tick_count} ticks (latest: {tick_event.price:.2f})", "DATA"
                        )
                        last_tick_log_count = tick_count
                except Exception as e:
                    logger.error(f"Error processing tick: {e}", exc_info=True)

        # C04 (FIXED): Removed duplicate historical data preloading.
        # History is now fetched exclusively in the Synchronization Phase.
        logger.info("[STARTUP] Skipping legacy preloading (consolidated into sync phase)")


        async def process_candles():
            """Process candle events from normalized MarketEventBus."""
            
            nonlocal candle_count, inference_count
            first_candle = True
            import time
            last_inference_time = 0.0

            
            async for candle_event in event_bus.subscribe_candles(symbol, interval=60):
                start_time = datetime.now()
                try:
                    if not candle_event:
                        continue
                        
                    # CRITICAL-001 (FIXED): Startup buffering via flag
                    if startup_state["buffering_active"]:
                        startup_state["buffer_candles"].append(candle_event)
                        logger.info(f"[STARTUP] Buffered candle at {candle_event.timestamp}")
                        return # Skip live processing
                        
                    # Buffer handles candle close detection internally
                    is_new_candle = buffer.update_candle(candle_event)
                    candle_count += 1

                    # Update regime detector with current price history on new candles
                    # This prevents the detector from becoming stale after startup
                    if is_new_candle and hasattr(regime_veto, 'update_prices'):
                        closes = buffer.get_candles_array(include_forming=False)[:, 3]
                        regime_veto.update_prices(closes)

                    if first_candle:
                        console_log(
                            f"First LIVE candle received (O:{candle_event.open:.2f} H:{candle_event.high:.2f} L:{candle_event.low:.2f} C:{candle_event.close:.2f})",
                            "SUCCESS",
                        )
                        first_candle = False

                    # H07: Stale Data Check
                    # Prevent trading on old data if system lags
                    # CRITICAL RULE 2: Timezone for Shadow Resolution (Must be UTC)
                    now_utc = datetime.now(timezone.utc)
                    latency = (now_utc - candle_event.timestamp).total_seconds()
                    
                    stale_threshold = settings.heartbeat.stale_data_threshold_seconds
                    
                    if latency > stale_threshold:
                        logger.warning(
                            f"[LATENCY] Skipping stale candle (closed {latency:.1f}s ago). "
                            f"Threshold: {stale_threshold:.1f}s"
                        )
                        console_log(f"Skipping stale candle ({latency:.1f}s lag)", "WARN")
                        continue
                    
                    # Log processing time for observability
                    process_time = (datetime.now() - start_time).total_seconds()
                    
                    if is_new_candle and buffer.is_ready():
                         candle_msg = f"Running inference #{inference_count + 1}..."
                    else:
                         candle_msg = f"Skipping (ready={buffer.is_ready()}, new={is_new_candle})"

                    console_log(f"Candle closed @ {candle_event.close:.2f} - {candle_msg} (latency: {latency:.1f}s)", "BRAIN")

                    # C03 (FIXED): Ambiguous Control Flow
                    # Separated Inference and Shadow Resolution logic explicitly 
                    
                    # 1. SHADOW RESOLUTION (Runs on EVERY candle close)
                    # CRITICAL RULE 3: Deterministic resolution
                    if is_new_candle:
                         # Use candle timestamp for deterministic resolution
                        candle_time = candle_event.timestamp
                        resolved_count = await resolver.resolve_trades(
                            current_price=candle_event.close,
                            current_time=candle_time,
                            high_price=candle_event.high,
                            low_price=candle_event.low,
                        )
                        if resolved_count > 0:
                            console_log(
                                f"🎯 Resolved {resolved_count} shadow trade(s)",
                                "SUCCESS",
                            )
                            logger.info(f"Resolved {resolved_count} shadow trades this candle")

                    # 2. INFERENCE TRIGGER
                    should_run_inference = False
                    
                    if is_new_candle and buffer.is_ready():
                        import time
                        time_since_last = time.time() - last_inference_time
                        cooldown_active = time_since_last < 30.0
                        
                        if cooldown_active:
                            logger.debug(f"[COOLDOWN] Skipping inference ({time_since_last:.1f}s < 30s)")
                        else:
                            should_run_inference = True

                    # 3. EXECUTE INFERENCE
                    if should_run_inference:
                        console_log(
                            f"Candle closed @ {candle_event.close:.2f} - Running inference #{inference_count + 1}... "
                            f"(latency: {latency:.1f}s)",
                            "MODEL",
                        )
                        logger.info(f"Candle closed: running inference (latency: {latency:.3f}s)")

                        try:
                            # C05 (FIXED): Use snapshot for consistent data during inference
                            # Prevents race conditions where buffer changes during async inference
                            snapshot = buffer.get_snapshot()
                            
                            await run_inference(
                                model,
                                engine,
                                executor,
                                snapshot, # Pass snapshot instead of buffer
                                feature_builder,
                                device,
                                settings,
                                metrics,
                                calibration_monitor=calibration_monitor,
                                trade_tracker=real_trade_tracker,
                                challengers=challengers,
                                strategy_adapter=strategy_adapter,
                            )
                            inference_count += 1
                            last_inference_time = time.time()
                        except Exception as inf_e:
                            logger.error(f"Inference cycle failed: {inf_e}", exc_info=True)
                            metrics.record_error("inference_failure")

                except Exception as e:
                    logger.error(f"Error processing candle: {e}", exc_info=True)

        async def maintenance_task():
            """
            Background maintenance task for logs and database pruning.
            Runs once every 24 hours.
            """
            interval = 86400  # 24 hours
            while True:
                try:
                    logger.info("[MAINTENANCE] Starting background maintenance...")
                    
                    # 1. Log cleanup
                    try:
                        from config.logging_config import cleanup_logs
                        
                        # Run in executor to avoid blocking loop with file IO
                        loop = asyncio.get_running_loop()
                        deleted_logs = await loop.run_in_executor(
                            None, 
                            lambda: cleanup_logs(log_dir, retention_days=settings.system.log_retention_days)
                        )
                        
                        if deleted_logs > 0:
                            logger.info(f"[MAINTENANCE] Deleted {deleted_logs} old log files")
                    except Exception as e:
                        logger.error(f"[MAINTENANCE] Log cleanup failed: {e}")

                    # 2. DB Pruning
                    try:
                        if shadow_store:
                            # Run VACUUM/Prune in executor as it locks DB
                            pruned_count = await loop.run_in_executor(
                                None,
                                lambda: shadow_store.prune(retention_days=settings.system.db_retention_days)
                            )
                            if pruned_count > 0:
                                logger.info(f"[MAINTENANCE] Pruned {pruned_count} old shadow records")
                    except Exception as e:
                        logger.error(f"[MAINTENANCE] DB pruning failed: {e}")

                    logger.info("[MAINTENANCE] Background maintenance completed.")
                except Exception as e:
                    logger.error(f"[MAINTENANCE] Unexpected error in maintenance task: {e}")
                
                await asyncio.sleep(interval)

        async def heartbeat():
            """Periodic status logging for observability."""
            from observability.shadow_metrics import ShadowTradeMetrics
            import time
            
            heartbeat_interval = settings.heartbeat.interval_seconds
            stale_threshold = settings.heartbeat.stale_data_threshold_seconds
            
            # Cache shadow metrics
            shadow_metrics_cache = ShadowTradeMetrics()
            last_metrics_update = 0
            metrics_update_interval = 300
            
            while True:
                await asyncio.sleep(heartbeat_interval)
                now = datetime.now()
                now_ts = time.time()
                stale_seconds = (now - last_tick_time).total_seconds()

                # Get current statistics
                stats = engine.get_statistics()
                executor.get_safety_statistics() if executor else {}
                cal_stats = calibration_monitor.get_statistics()
                
                # Update shadow trade metrics
                if now_ts - last_metrics_update > metrics_update_interval:
                    try:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None, shadow_metrics_cache.update_from_store, shadow_store
                        )
                        last_metrics_update = now_ts
                    except Exception as e:
                        logger.error(f"Failed to update shadow metrics: {e}")

                # M12 (FIXED): Robust Hot-Reloading
                # Uses persistent state dictionary instead of locals()
                # C06 (FIXED): Atomic reload with rollback
                if checkpoint_path and checkpoint_path.exists():
                    try:
                        current_mtime = checkpoint_path.stat().st_mtime
                        
                        # Initialize if missing (C02 Fix)
                        if "last_ckpt_mtime" not in hot_reload_state:
                            hot_reload_state["last_ckpt_mtime"] = current_mtime
                        
                        if current_mtime > hot_reload_state["last_ckpt_mtime"]:
                            console_log(f"Checkpoint update detected! Reloading... ({checkpoint_path.name})", "MODEL")
                            logger.info(f"[HOT-RELOAD] Checkpoint modified at {datetime.fromtimestamp(current_mtime)}")
                            
                            # 1. Load into temporary model first (Atomic Step 1)
                            new_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                            
                            # 2. Validate Architecture (Atomic Step 2)
                            current_keys = set(model.state_dict().keys())
                            new_keys = set(new_checkpoint["model_state_dict"].keys())
                            
                            missing = current_keys - new_keys
                            unexpected = new_keys - current_keys
                            
                            if missing or unexpected:
                                logger.warning(f"[HOT-RELOAD] Mismatch! Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                                console_log("Hot-reload aborted: architecture mismatch", "WARN")
                            else:
                                # 3. Apply changes (Atomic Step 3)
                                model.load_state_dict(new_checkpoint["model_state_dict"], strict=False)
                                model.eval()
                                
                                # 4. Update Metadata
                                if "manifest" in new_checkpoint:
                                    try:
                                        from utils.bootstrap import validate_model_compatibility
                                        validate_model_compatibility(new_checkpoint, settings)
                                        new_ver = new_checkpoint["manifest"].get("model_version", "unknown")
                                        if hasattr(engine, 'model_version'):
                                            engine.model_version = new_ver
                                        console_log(f"Model reloaded: v{new_ver}", "SUCCESS")
                                    except Exception as ve:
                                        logger.error(f"[HOT-RELOAD] Version validation failed: {ve}")

                                # 5. Update State & Reset Monitors
                                hot_reload_state["last_ckpt_mtime"] = current_mtime
                                calibration_monitor.reset() # C06 Fix: Explicit reset method
                                logger.info("[HOT-RELOAD] Success. Calibration monitor reset.")
                                
                    except Exception as e:
                        logger.error(f"[HOT-RELOAD] Failed: {e}")
                        console_log(f"Hot-reload failed: {e}", "ERROR")
                        # Do not update mtime so it retries next heartbeat? 
                        # Or update to avoid infinite error loop?
                        # Better to update mtime to prevent log spam loop.
                        hot_reload_state["last_ckpt_mtime"] = current_mtime

                # HEARTBEAT LOGGING
                stale_threshold = settings.heartbeat.stale_data_threshold_seconds
                if stale_seconds > stale_threshold:
                    console_log(f"WARNING: No ticks for {stale_seconds:.1f}s - possible connection issue", "WARN")
                    logger.warning(f"[STALE] No ticks received for {stale_seconds:.1f}s")

                logger.info(
                    f"[HEARTBEAT] ticks={tick_count}, candles={candle_count}, "
                    f"inferences={inference_count}, stale_sec={stale_seconds:.1f}, "
                    f"shadow_win_rate={shadow_metrics_cache.win_rate:.3f}"
                )

        # Start background tasks
        tasks = [
            asyncio.create_task(process_ticks()),
            asyncio.create_task(process_candles()),
            asyncio.create_task(heartbeat()),
            asyncio.create_task(maintenance_task()),
        ]

        # ══════════════════════════════════════════════════════════════════════
        # SYNCHRONIZATION PHASE
        # 1. Background tasks are already buffering input
        # 2. Fetch historical data to fill buffer
        # 3. Replay buffered data
        # 4. Enable real-time processing
        # ══════════════════════════════════════════════════════════════════════
        console_log("Synchronizing with market history...", "WAIT")
        try:
            # Fetch history
            hist_ticks = await client.get_historical_ticks(count=settings.data_shapes.sequence_length_ticks)
            hist_candles = await client.get_historical_candles(
                count=settings.data_shapes.sequence_length_candles, 
                interval=60
            )
            
            console_log(f"Fetched history: {len(hist_ticks)} ticks, {len(hist_candles)} candles", "INFO")
            
            # Populate buffer with history
            for price in hist_ticks:
                buffer.append_tick(price)
            
            for c in hist_candles:
                # Convert dict to CandleEvent-like structure or pass dict if buffer handles it
                # Buffer expects CandleEvent objects usually? 
                # Let's check update_candle. It expects CandleEvent.
                # Client returns dicts. We need to convert.
                from data.events import CandleEvent
                ce = CandleEvent(
                    symbol=symbol,
                    open=float(c["open"]),
                    high=float(c["high"]),
                    low=float(c["low"]),
                    close=float(c["close"]),
                    volume=0.0,
                    timestamp=datetime.fromtimestamp(c["epoch"], tz=timezone.utc),
                    metadata={"source": "history"}
                )
                buffer.update_candle(ce)
                
            console_log("History buffered. Replaying startup buffer...", "INFO")
            
            # Replay buffered live events
            # Note: We are modifying buffer while background tasks are appending to *startup_buffer*
            # We need to act atomically or just process what we have and then switch flag
            
            # Better approach:
            # 1. Process current startup buffer
            # 2. Set complete flag (background tasks will switch to direct buffer append)
            # 3. BUT race condition: between processing and setting flag, new items might be added to startup buffer?
            #    Yes. 
            # safe way:
            #   Iterate startup buffer.
            #   Set flag.
            #   Wait, if we set flag, background tasks write to 'buffer'.
            #   But what if we missed some items in 'startup_buffer' that were added strictly before flag set but after we iterated?
            #
            #   Actually, the background task writes to startup_buffer IF not set.
            #   So we should:
            #   1. Lock mechanism? Or just accepting a small race is hard.
            #   Python asyncio is single threaded. 
            #   We can consume the list, clear it, check if empty, then set flag?
            #   No, while we are here, background task is NOT running (cooperative multitasking).
            #   So we can safely drain the list and set the flag without race condition!
            #   Because we are in 'await' free block here (except if we await something).
            
            # CRITICAL-001 (FIXED): Startup Synchronization logic
            # Explicit, lock-free state transition
            
            # 1. Drain buffering that happened while we fetched history
            #    (Background tasks are continuing to append to startup_state["buffer"])
            #    We need to seamlessly transition to live buffer.
            
            # Since we are in the main coroutine and tasks yield to us,
            # we can atomically modify the state as long as we don't await.
            
            # Capture what has been buffered so far
            startup_ticks = list(startup_state["buffer_ticks"])
            startup_candles = list(startup_state["buffer_candles"])
            
            # FLIP THE SWITCH - Enable live processing in background tasks
            # This is atomic (no await between read and write)
            startup_state["buffering_active"] = False
            
            # Clear startup buffers to free memory
            startup_state["buffer_ticks"] = []
            startup_state["buffer_candles"] = []
            
            # Now replay everything in correct order:
            # 1. Historical Data (fetched above)
            # 2. Startup Buffered Data (captured just now)
            # 3. New data (will go directly to buffer now)
            
            console_log(f"Replaying buffer: {len(startup_ticks)} ticks, {len(startup_candles)} candles", "INFO")
            
            for t in startup_ticks:
                 buffer.append_tick(t)
            
            for c in startup_candles:
                 buffer.update_candle(c)
                 
            console_log(f"Synchronization complete.", "SUCCESS")
            startup_complete.set()
            
            for t_price in replay_ticks:
                buffer.append_tick(t_price)
                tick_count += 1
                
            for c_event in replay_candles:
                buffer.update_candle(c_event)
                candle_count += 1

            console_log(f"Synchronization complete. Replayed {len(replay_ticks)} ticks.", "SUCCESS")
            logger.info("Startup synchronization complete. Live processing enabled.")

        except Exception as e:
            logger.critical(f"Synchronization failed: {e}")
            console_log(f"Sync failed: {e}", "ERROR")
            # Cancel tasks and exit
            for t in tasks: t.cancel()
            return 1


        if not args.test:
             # Add real trade tracker task if not in test mode
             # RealTradeTracker does not have a start method, it works proactively
             # tasks.append(asyncio.create_task(real_trade_tracker.start()))
             pass
        
        # Monitor tasks
        # If any task fails, we should probably stop the system
        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        
        # H04 (FIXED): Graceful cancellation of background tasks
        for task in pending:
            task.cancel()
        
        # Await cancelled tasks to allow them to handle CancelledError and cleanup
        if pending:
             await asyncio.gather(*pending, return_exceptions=True)
            
    except Exception as e:
        logger.critical(f"FATAL ERROR in main loop: {e}", exc_info=True)
        console_log(f"CRITICAL ERROR: {e}", "ERROR")
        return 1
    finally:
        if client:
            await client.disconnect()
        
        # Log statistics
        if 'engine' in locals():
            stats = engine.get_statistics()
            logger.info(f"Session statistics: {stats}")

        if 'executor' in locals() and executor:
            # Log safety wrapper statistics
            safety_stats = executor.get_safety_statistics()
            logger.info(f"Safety statistics: {safety_stats}")
            # IMPORTANT-001: Graceful shutdown
            await executor.shutdown()

        console_log("System shutdown complete", "INFO")

    return 0



async def run_inference(
    model,
    engine,
    executor,
    market_snapshot, # C05 (FIXED): Takes snapshot dict instead of buffer object
    feature_builder,
    device,
    settings,
    metrics,
    return_recon_error: bool = False,
    calibration_monitor: CalibrationMonitor | None = None,
    trade_tracker: Any | None = None,  # RealTradeTracker for outcome tracking
    challengers: list[dict[str, Any]] | None = None, # OPT5: List of challenger stacks
    strategy_adapter: Any | None = None, # IMPORTANT-005
):
    """
    Run single inference cycle with regime veto integration.

    Uses the CANONICAL FeatureBuilder to ensure consistent feature engineering.

    This function:
    1. Uses FeatureBuilder to process raw data (CANONICAL PATH)
    2. Gets model predictions AND reconstruction error
    3. Passes reconstruction error to DecisionEngine for regime assessment
    4. Uses TradeExecutor abstraction for broker-isolated execution
    5. Records structured metrics for observability
    6. Tracks reconstruction errors for graceful degradation

    Args:
        model: The DerivOmniModel instance
        engine: DecisionEngine for trade decisions
        executor: SafeTradeExecutor for safe execution
        market_snapshot: Snapshot dict from MarketDataBuffer
        feature_builder: FeatureBuilder for canonical feature engineering
        device: Torch device for inference
        settings: Application settings
        metrics: TradingMetrics instance for recording metrics
        return_recon_error: If True, return reconstruction error for validation
        calibration_monitor: Optional CalibrationMonitor for graceful degradation
        strategy_adapter: StrategyAdapter for signal-to-request conversion (optional for backward compat/testing)

    Returns:
        If return_recon_error is True, returns the reconstruction error float.
        Otherwise returns None.
    """
    import time
    
    # ... [Inference logic skipped for brevity - unchanged] ...
    
    inference_start = time.perf_counter()
    
    # R01: Create span for entire inference cycle if tracing enabled
    span = None
    if TRACING_ENABLED and tracer:
        span = tracer.start_span("run_inference")
        span.set_attribute("device", str(device))

    try:
        # C05 (FIXED): unpack snapshot
        t_np = market_snapshot['ticks']
        c_np = market_snapshot['candles']
        
        # NOTE: snapshot ensures t_np and c_np are consistent copies.
        # But wait, what if buffer was empty?
        if len(t_np) == 0 or len(c_np) == 0:
             logger.warning("Empty snapshot during inference!")
             return None

        # CRITICAL-004: Pass validation timestamp
        features = feature_builder.build(
            ticks=t_np, 
            candles=c_np,
            timestamp=time.time()
        )

        t_tensor = features["ticks"].unsqueeze(0).to(device)
        c_tensor = features["candles"].unsqueeze(0).to(device)
        v_tensor = features["vol_metrics"].unsqueeze(0).to(device)

        # Inference with timing
        with torch.no_grad():
            # print("Running model prediction...", flush=True) # Un-comment if needed
            # C01: Offload synchronous model inference to thread pool to avoid blocking event loop
            loop = asyncio.get_running_loop()
            from functools import partial
            
            # Run prediction in executor
            probs = await loop.run_in_executor(
                None, 
                partial(model.predict_probs, t_tensor, c_tensor, v_tensor)
            )

            # CRITICAL: Get reconstruction error for regime veto assessment
            # This allows the regime authority to block trades during anomalous conditions
            # Also offload this if it's heavy, though usually lighter than predict_probs
            reconstruction_error = await loop.run_in_executor(
                None,
                lambda: model.get_volatility_anomaly_score(v_tensor).item()
            )

            if span:
                span.set_attribute("reconstruction_error", reconstruction_error)

        # Record inference latency
        inference_latency = time.perf_counter() - inference_start
        metrics.record_inference_latency(inference_latency)

        # Record reconstruction error gauge
        metrics.set_reconstruction_error(reconstruction_error)

        sample_probs = {k: v.item() for k, v in probs.items()}
        logger.info(f"Predictions: {sample_probs}")
        logger.info(
            f"Reconstruction error: {reconstruction_error:.4f} (latency: {inference_latency * 1000:.1f}ms)"
        )

        # ══════════════════════════════════════════════════════════════════════════
        # CALIBRATION MONITORING: Track errors for graceful degradation
        # If errors are persistently high, activate shadow-only mode
        # ══════════════════════════════════════════════════════════════════════════
        if calibration_monitor is not None:
            calibration_monitor.record(reconstruction_error)
            calibration_monitor.recover_if_healthy()

        # Get entry price (last completed candle close for accurate context)
        # Candle format: [open, high, low, close, volume, timestamp]
        entry_price = float(c_np[-1, 3]) if len(c_np) > 0 else float(t_np[-1])

        # Decision with FULL CONTEXT CAPTURE via process_with_context
        # This stores tick/candle windows in ShadowTradeStore for:
        #   1. Accurate outcome resolution
        #   2. Retraining on production data
        
        decision_start = time.perf_counter()

        # ══════════════════════════════════════════════════════════════════════════
        # OPT5: Challenger Inference (Concurrent A/B Testing)
        # ══════════════════════════════════════════════════════════════════════════
        if challengers:
            for c_stack in challengers:
                # Fire-and-forget concurrent tasks for challengers
                # We don't wait for them to block real trading
                asyncio.create_task(_run_challenger_inference(
                    c_stack=c_stack,
                    t_tensor=t_tensor,
                    c_tensor=c_tensor,
                    v_tensor=v_tensor,
                    t_np=t_np,
                    c_np=c_np,
                    entry_price=entry_price,
                    buffer=buffer
                ))
        real_trades = await engine.process_with_context(
            probs=sample_probs,
            reconstruction_error=reconstruction_error,
            tick_window=t_np,
            candle_window=c_np,
            entry_price=entry_price,
            market_data={
                "ticks_count": market_snapshot["tick_count"], 
                "candles_count": market_snapshot["candle_count"]
            },
        )
        decision_latency = time.perf_counter() - decision_start
        metrics.record_decision_latency(decision_latency)
        
        # Log slow decisions
        if decision_latency > 0.1:
            logger.warning(f"[LATENCY] Decision engine slow: {decision_latency*1000:.1f}ms")

        # Record regime assessment (check engine statistics for last decision)
        stats = engine.get_statistics()
        if stats.get("vetoed_count", 0) > 0:
            # Check if this inference was vetoed
            metrics.record_regime_assessment("VETO")
        else:
            metrics.record_regime_assessment("TRUSTED")

        # ══════════════════════════════════════════════════════════════════════════
        # SHADOW-ONLY MODE: If calibration is poor, skip real trades but keep shadow
        # This protects the account while still learning from production data
        # ══════════════════════════════════════════════════════════════════════════
        if calibration_monitor is not None and calibration_monitor.should_skip_real_trades():
            if real_trades:
                logger.warning(
                    f"[SHADOW-ONLY] Skipping {len(real_trades)} real trade(s) due to calibration issues. "
                    f"Trades logged as shadow only."
                )
                metrics.record_trade_attempt(outcome="blocked_shadow_only", contract_type="all")
            real_trades = []  # Clear real trades, but shadows are already logged

        logger.info(f"Real trades: {len(real_trades)}")

        # Calculate realized volatility for position sizing
        # Use close prices from buffer
        import numpy as np
        # Calculate realized volatility for position sizing
        # Use close prices from snapshot
        import numpy as np
        # candle format is [open, high, low, close, volume, timestamp] (idx 3 is close)
        closes = c_np[:, 3]
        
        # Ensure numpy
        if hasattr(closes, "cpu"):
             closes = closes.cpu().numpy()
        elif hasattr(closes, "numpy"):
             closes = closes.numpy()
             
        volatility = 0.0
        if len(closes) > 20 and np.all(closes > 0):
             # Simple annualized vol estimate
             log_returns = np.diff(np.log(closes[-20:]))
             volatility = float(np.std(log_returns) * np.sqrt(365 * 24 * 60))

        # Execute trades via abstraction (broker-isolated)
        for signal in real_trades:
            # Inject volatility context for sizing
            if signal.metadata is None:
                signal.metadata = {}
            signal.metadata['volatility'] = volatility
            
            # IMPORTANT-005: Transform Signal -> ExecutionRequest

            # IMPORTANT-005: Transform Signal -> ExecutionRequest
            execution_request = None
            if strategy_adapter:
                # Fetch balance logic (optional for Kelly)
                # Ideally this should be cached or fetched efficiently
                # For now assuming simple sizing or adapter handles it
                # Adapter.convert_signal(signal, account_balance=...)
                try:
                     # CRITICAL-004: Pass Regime Context
                     assessment = engine.get_regime_assessment(reconstruction_error)
                     regime_state = engine.get_regime_state_string(assessment)
                     
                     execution_request = await strategy_adapter.convert_signal(
                         signal, 
                         reconstruction_error=reconstruction_error,
                         regime_state=regime_state
                     )
                     logger.info(f"EXECUTING Request: {execution_request}")
                except Exception as e:
                     logger.error(f"Strategy Adapter failed to convert signal: {e}")
                     continue
            else:
                 logger.error("No StrategyAdapter provided! Cannot execute.")
                 continue

            
            
            # C07 (FIXED): Atomic Intent Execution with Context Manager
            async with trade_tracker.intent(
                direction=signal.direction,
                entry_price=entry_price,
                stake=stake,
                probability=signal.probability,
                contract_type=contract_type
            ) as intent_id:
            
                exec_start = time.perf_counter()
                result = await executor.execute(execution_request)
                exec_latency = time.perf_counter() - exec_start
                
                metrics.record_execution_latency(exec_latency)
                
                if result.success:
                     logger.info(f"Trade executed: contract_id={result.contract_id}")
                     metrics.record_trade_attempt(outcome="executed", contract_type=contract_type)
                     
                     # Confirm intent
                     if trade_tracker:
                         trade_tracker.confirm_intent(intent_id, result.contract_id)
                         await trade_tracker.register_trade(
                            contract_id=result.contract_id,
                            direction=signal.direction,
                            entry_price=entry_price,
                            stake=stake,
                            probability=signal.probability,
                            contract_type=str(contract_type),
                        )
                else:
                    logger.error(f"Trade failed: {result.error}")
                    # Capture meaningful errors
                    if "rate limit" in (result.error or "").lower():
                        metrics.record_trade_attempt(outcome="blocked_rate_limit", contract_type=contract_type)
                    elif "kill switch" in (result.error or "").lower():
                        metrics.record_trade_attempt(outcome="blocked_kill_switch", contract_type=contract_type)
                    elif "loss limit" in (result.error or "").lower():
                        metrics.record_trade_attempt(outcome="blocked_pnl_cap", contract_type=contract_type)
                    else:
                        metrics.record_trade_attempt(outcome="failed", contract_type=contract_type)
                    
                    # Manual cleanup of intent is handled by context manager automatically on exit if needed?
                    # The Context Manager cleans up if exception raised.
                    # Here we didn't raise, we just failed.
                    # We should probably clean up explicitly or rely on the CM to do nothing?
                    # The CM description said "On context exit, automatically cleanup if result wasn't confirmed".
                    # Let's check implementation of CM.
                    # Implementation:
                    #     try: yield intent_id
                    #     except Exception: cleanup()
                    # It DOES NOT auto-cleanup on normal exit if not confirmed.
                    # So we must call cleanup explicitly if we know it failed.
                    if trade_tracker:
                         trade_tracker.cleanup_intent(intent_id)

        # Return reconstruction error for startup validation if requested
        if return_recon_error:
            return reconstruction_error
        return None
        
    finally:
        # R01: Close span with final attributes
        if span:
            span.set_attribute("inference_latency_ms", (time.perf_counter() - inference_start) * 1000)
            span.end()


if __name__ == "__main__":
    from scripts.shutdown_handler import run_async_with_graceful_shutdown

    parser = argparse.ArgumentParser(description="Live trading")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint name to load (e.g., 'best_model')"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument("--test", action="store_true", help="Test connection only, don't trade")
    parser.add_argument(
        "--shadow-only", action="store_true", help="Run in shadow mode (no real trades)"
    )
    # I02: Checkpoint verification bypass (for development/debugging only)
    parser.add_argument(
        "--skip-checkpoint-verify", action="store_true", 
        help="Skip checkpoint verification (not recommended for production)"
    )
    # Compounding / Position Sizing Arguments
    parser.add_argument(
        "--compound", action="store_true", help="Enable compounding strategy (alias for --strategy compound)"
    )
    parser.add_argument(
        "--strategy", type=str, choices=["fixed", "compound", "martingale", "kelly"], 
        help="Position sizing strategy (default: fixed)"
    )
    parser.add_argument(
        "--x-amount", type=str, default=None, 
        help="Multiplier/Type for compound/martingale (e.g., '2x', 'reinvest')"
    )
    parser.add_argument(
        "--winstrikes", type=int, default=5, 
        help="Max consecutive wins/losses for compounding/martingale (default: 5)"
    )

    args = parser.parse_args()
    sys.exit(run_async_with_graceful_shutdown(run_live_trading(args)))


async def _run_challenger_inference(
    c_stack: dict[str, Any],
    t_tensor: torch.Tensor,
    c_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    t_np: np.ndarray,
    c_np: np.ndarray,
    entry_price: float,
    buffer: Any
) -> None:
    """
    Helper to run challenger inference concurrently.
    
    This replicates the core inference loop but for a challenger stack.
    Exceptions are swallowed to strictly prevent impacting the Champion loop.
    """
    import asyncio
    
    model = c_stack["model"]
    engine = c_stack["engine"]
    version = c_stack["version"]
    
    try:
        loop = asyncio.get_running_loop()
        from functools import partial
        
        # 1. Prediction (offload to thread)
        probs = await loop.run_in_executor(
            None, 
            partial(model.predict_probs, t_tensor, c_tensor, v_tensor)
        )
        sample_probs = {k: v.item() for k, v in probs.items()}
        
        # 2. Reconstruction Error (required for regime assessment)
        reconstruction_error = await loop.run_in_executor(
            None,
            lambda: model.get_volatility_anomaly_score(v_tensor).item()
        )
        
        # 3. Decision & Shadow Logging
        # Process with full context to store shadow trades in shared DB
        # The engine is configured in "SHADOW" mode, so it tracks that metadata
        await engine.process_with_context(
            probs=sample_probs,
            reconstruction_error=reconstruction_error,
            tick_window=t_np,
            candle_window=c_np,
            entry_price=entry_price,
            market_data={"ticks_count": buffer.tick_count(), "candles_count": buffer.candle_count()}
        )
    except Exception as e:
        logger.error(f"[CHALLENGER {version}] Failed: {e}")
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
from utils.bootstrap import create_trading_stack, create_challenger_stack
# NEW IMPORTS
from execution.synchronizer import StartupSynchronizer
from execution.orchestrator import InferenceOrchestrator

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

        async def process_ticks():
            """Process tick events from normalized MarketEventBus."""
            nonlocal tick_count, last_tick_time, last_tick_log_count
            first_tick = True
            async for tick_event in event_bus.subscribe_ticks(symbol):
                # H8 (FIXED): Update liveness immediately on receipt
                last_tick_time = datetime.now()
                try:
                    # Update heartbeat timestamp
                    # Update heartbeat timestamp
                    
                    logger.debug(f"Tick received: {tick_event}")
                    
                    
                    # CRITICAL-001 (FIXED): Check Synchronizer (CONTINUE, don't RETURN)
                    if synchronizer.handle_tick(tick_event.price):
                        # Event buffered, skip live processing
                        if not first_tick_received.is_set():
                             first_tick_received.set()
                        continue  # Correct control flow: continue loop, don't exit coroutine!

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
                        
                    # CRITICAL-001 (FIXED): Startup buffering via Synchronizer
                    if synchronizer.handle_candle(candle_event):
                        logger.info(f"[STARTUP] Buffered candle at {candle_event.timestamp}")
                        continue  # Correct control flow: continue loop, don't exit coroutine!
                        
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
                            # Issue 6 (FIXED): Delegated to Orchestrator
                            # Issue 3 (FIXED): Orchestrator handles bounded concurrency for challengers
                            
                            snapshot = buffer.get_snapshot()
                            
                            reconstruction_error = await orchestrator.run_cycle(
                                market_snapshot=snapshot,
                                challengers=challengers
                            )
                            
                            # Use return value for trend monitoring
                            if reconstruction_error is not None and reconstruction_error > 0.3:
                                console_log(f"⚠️ High reconstruction error: {reconstruction_error:.3f}", "WARN")
                            
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
                        
                        # Skip if in backoff period
                        import time as time_module
                        if time_module.time() < hot_reload_state.get("backoff_until", 0):
                            pass  # Silent skip during backoff
                        elif current_mtime > hot_reload_state["last_ckpt_mtime"]:
                            console_log(f"Checkpoint update detected! Reloading... ({checkpoint_path.name})", "MODEL")
                            logger.info(f"[HOT-RELOAD] Checkpoint modified at {datetime.fromtimestamp(current_mtime)}")
                            
                            # 1. Load into temporary model first (Atomic Step 1)
                            new_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                            
                            # Issue E: Support multiple common checkpoint key variants
                            state_dict_key = None
                            for key in ["model_state_dict", "state_dict", "model"]:
                                if key in new_checkpoint:
                                    state_dict_key = key
                                    break
                            
                            if state_dict_key is None:
                                logger.error(f"[HOT-RELOAD] No valid state_dict key found. Available: {list(new_checkpoint.keys())}")
                                console_log("Hot-reload aborted: no valid state_dict key", "WARN")
                                continue
                            
                            # 2. Validate Architecture (Atomic Step 2)
                            current_keys = set(model.state_dict().keys())
                            new_keys = set(new_checkpoint[state_dict_key].keys())
                            
                            missing = current_keys - new_keys
                            unexpected = new_keys - current_keys
                            
                            if missing or unexpected:
                                logger.warning(f"[HOT-RELOAD] Mismatch! Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                                console_log("Hot-reload aborted: architecture mismatch", "WARN")
                            else:
                                # 3. Apply changes (Atomic Step 3)
                                model.load_state_dict(new_checkpoint[state_dict_key], strict=False)
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
                                hot_reload_state["consecutive_failures"] = 0  # Reset on success
                                calibration_monitor.reset() # C06 Fix: Explicit reset method
                                logger.info("[HOT-RELOAD] Success. Calibration monitor reset.")
                                
                    except Exception as e:
                        # Exponential backoff for persistent failures
                        import time as time_module
                        hot_reload_state["consecutive_failures"] = hot_reload_state.get("consecutive_failures", 0) + 1
                        failures = hot_reload_state["consecutive_failures"]
                        
                        # Backoff: 60s, 120s, 240s... max 30min
                        backoff_seconds = min(60 * (2 ** (failures - 1)), 1800)
                        hot_reload_state["backoff_until"] = time_module.time() + backoff_seconds
                        
                        logger.error(f"[HOT-RELOAD] Failed (attempt {failures}): {e}. Next retry in {backoff_seconds}s")
                        console_log(f"Hot-reload failed: {e}", "ERROR")
                        
                        # Alert after 3 consecutive failures
                        if failures >= 3:
                            alert_manager.trigger(
                                "hot_reload_persistent_failure",
                                f"Model hot-reload failed {failures} times for {checkpoint_path.name}",
                                AlertLevel.WARNING
                            )

                # HEARTBEAT LOGGING
                stale_threshold = settings.heartbeat.stale_data_threshold_seconds
                if stale_seconds > stale_threshold:
                    console_log(f"WARNING: No ticks for {stale_seconds:.1f}s - possible connection issue", "WARN")
                    logger.warning(f"[STALE] No ticks received for {stale_seconds:.1f}s")

                logger.info(
                    f"[HEARTBEAT] status={'LIVE' if synchronizer.is_live() else 'BUFFERING'}, "
                    f"ticks={tick_count}, candles={candle_count}, "
                    f"inferences={inference_count}, stale_sec={stale_seconds:.1f}, "
                    f"shadow_win_rate={shadow_metrics_cache.win_rate:.3f}"
                )
                
                # Issue 4 (FIXED): Task Liveness Check
                for t in tasks:
                    if t.done():
                         # Task shouldn't be done unless it crashed or we are shutting down
                         # If we are here, we are not shutting down (yet)
                         if t.exception():
                              logger.critical(f"FATAL: Background task failed: {t.get_name()} error={t.exception()}")
                              alert_manager.trigger("task_failure", f"Task {t.get_name()} died unexpectedly", AlertLevel.CRITICAL)
                         else:
                              logger.warning(f"Background task {t.get_name()} finished unexpectedly")

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
        async with asyncio.TaskGroup() as tg:
            tg.create_task(process_ticks(), name="tick_processor")
            tg.create_task(process_candles(), name="candle_processor")
            tg.create_task(heartbeat(), name="heartbeat_monitor")
            tg.create_task(maintenance_task(), name="maintenance_worker")
            
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
            if 'engine' in locals():
                stats = engine.get_statistics()
                logger.info(f"Session statistics: {stats}")
        except Exception as e:
            logger.error(f"Error getting engine statistics: {e}")

        try:
            if 'executor' in locals() and executor:
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



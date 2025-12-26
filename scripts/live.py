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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from dotenv import load_dotenv

load_dotenv()

from config.settings import load_settings
from data.buffer import MarketDataBuffer  # Abstracted buffering logic
from data.features import FeatureBuilder  # CANONICAL feature pipeline
from data.ingestion.client import DerivClient
from data.ingestion.deriv_adapter import DerivEventAdapter
from execution.decision import DecisionEngine
from execution.executor import DerivTradeExecutor
from execution.regime import RegimeVeto
from execution.safety import ExecutionSafetyConfig, SafeTradeExecutor

# from execution.shadow_logger import ShadowLogger  # DEPRECATED: Use SQLiteShadowStore
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


from utils.logging_setup import setup_logging

# Configure logging to both console and file
logger, log_dir, log_file = setup_logging(script_name="live_trading")

# Import shared console logging utilities
from scripts.console_utils import console_log


class CalibrationMonitor:
    """
    Monitor reconstruction errors for calibration issues.

    Provides:
    - Tracking of reconstruction error history
    - Shadow-only mode when errors are persistently high
    - Escalating alerts for sustained calibration issues

    This enables graceful degradation: instead of blocking all trades
    when thresholds are miscalibrated, the system can fall back to
    shadow-only mode to continue learning while protecting the account.
    """

    def __init__(
        self, error_threshold: float = 1.0, consecutive_threshold: int = 5, window_size: int = 20
    ):
        """
        Args:
            error_threshold: Errors above this trigger shadow-only mode
            consecutive_threshold: Number of consecutive high errors to trigger alert
            window_size: Size of rolling window for statistics
        """
        self.error_threshold = error_threshold
        self.consecutive_threshold = consecutive_threshold
        self.window_size = window_size

        self.errors: list = []
        self.consecutive_high_count = 0
        self.shadow_only_mode = False
        self.shadow_only_reason = ""
        self.alert_escalation_level = 0  # 0=none, 1=warning, 2=critical

    def record(self, error: float) -> None:
        """Record a new reconstruction error and update state."""
        self.errors.append(error)
        if len(self.errors) > self.window_size:
            self.errors.pop(0)

        # Track consecutive high errors
        if error > self.error_threshold:
            self.consecutive_high_count += 1
        else:
            self.consecutive_high_count = 0

        # Activate shadow-only mode if too many consecutive high errors
        if self.consecutive_high_count >= self.consecutive_threshold:
            if not self.shadow_only_mode:
                self.shadow_only_mode = True
                self.shadow_only_reason = (
                    f"Reconstruction error exceeded {self.error_threshold} for "
                    f"{self.consecutive_high_count} consecutive inferences"
                )
                logger.critical(
                    f"[SHADOW-ONLY] Activating shadow-only mode: {self.shadow_only_reason}"
                )
                self.alert_escalation_level = 2

        # Escalate alerts based on persistence
        if len(self.errors) >= self.window_size:
            high_error_ratio = sum(1 for e in self.errors if e > self.error_threshold) / len(
                self.errors
            )
            if high_error_ratio > 0.5 and self.alert_escalation_level < 2:
                self.alert_escalation_level = 1
                logger.warning(
                    f"[CALIBRATION] {high_error_ratio * 100:.0f}% of recent inferences have "
                    f"high reconstruction error. Consider model retraining."
                )

    def should_skip_real_trades(self) -> bool:
        """Return True if real trades should be skipped (shadow-only mode)."""
        return self.shadow_only_mode

    def recover_if_healthy(self) -> None:
        """Recover from shadow-only mode if errors normalize."""
        if not self.shadow_only_mode:
            return

        if len(self.errors) >= self.window_size:
            recent_high_ratio = sum(1 for e in self.errors if e > self.error_threshold) / len(
                self.errors
            )
            if recent_high_ratio < 0.2:
                self.shadow_only_mode = False
                self.shadow_only_reason = ""
                self.alert_escalation_level = 0
                logger.info(
                    "[SHADOW-ONLY] Recovered: reconstruction errors normalized. "
                    "Real trading re-enabled."
                )

    def get_statistics(self) -> dict:
        """Get current calibration monitoring statistics."""
        if not self.errors:
            return {"samples": 0}

        return {
            "samples": len(self.errors),
            "mean_error": sum(self.errors) / len(self.errors),
            "max_error": max(self.errors),
            "min_error": min(self.errors),
            "shadow_only_mode": self.shadow_only_mode,
            "alert_level": self.alert_escalation_level,
            "consecutive_high": self.consecutive_high_count,
        }


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

    # M13: Disk Usage Management
    # Perform cleanup at startup to prevent unbounded growth
    try:
        from utils.logging_setup import cleanup_logs
        deleted_logs = cleanup_logs(log_dir, retention_days=settings.system.log_retention_days)
        if deleted_logs > 0:
            console_log(f"Cleaned up {deleted_logs} old log files", "INFO")
            logger.info(f"Deleted {deleted_logs} old log files (> {settings.system.log_retention_days} days)")
    except Exception as e:
        logger.error(f"Log cleanup failed: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # STRUCTURED METRICS - for production observability
    # Records: inference latency, trade outcomes, regime assessments, P&L
    # ══════════════════════════════════════════════════════════════════════
    console_log("Initializing metrics collector...", "WAIT")
    metrics = TradingMetrics(enable_prometheus=True)
    console_log("Metrics collector ready", "SUCCESS")
    logger.info(
        f"Metrics collector initialized (prometheus={'enabled' if metrics.use_prometheus else 'disabled'})"
    )

    # Determine checkpoint to load
    checkpoint_name = args.checkpoint
    checkpoint_path = None
    if not checkpoint_name:
        default_ckpt = Path(args.checkpoint_dir) / "best_model.pt"
        if default_ckpt.exists():
            checkpoint_name = "best_model"
            checkpoint_path = default_ckpt
            logger.info("No checkpoint specified, auto-selecting 'best_model'")
        else:
            if not args.test:
                logger.critical(
                    "FATAL: No checkpoint found and 'best_model.pt' missing. "
                    "Cannot trade with random weights. Exiting."
                )
                console_log("ERROR: No model checkpoint found - refusing to trade", "ERROR")
                return 1
            logger.warning("Test mode: Model initialized with RANDOM weights (acceptable for testing).")
    else:
        checkpoint_path = Path(args.checkpoint_dir) / f"{checkpoint_name}.pt"

    # Pre-inspect checkpoint to determine architecture (BiLSTM vs TFT)
    # This prevents crashing when loading legacy BiLSTM checkpoints into a TFT model
    checkpoint = None
    if checkpoint_path and checkpoint_path.exists():
        console_log(f"Inspecting checkpoint: {checkpoint_name}...", "MODEL")
        try:
            # Load to CPU first to inspect structure
            # SECURITY: Try weights_only=True first (safer)
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            except Exception:
                logger.warning("[SECURITY] Checkpoint requires pickle (weights_only=False). This is risky if source is untrusted.")
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False) # nosec
            
            # Detect architecture type
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            has_tft = any("temporal.tft" in k for k in state_dict.keys())
            
            if has_tft:
                settings.hyperparams.use_tft = True
                logger.info("[INIT] Detected TFT architecture in checkpoint")
                console_log("Architecture: TFT (Transformer)", "INFO")
            else:
                settings.hyperparams.use_tft = False
                logger.info("[INIT] Detected BiLSTM architecture in checkpoint (Legacy)")
                console_log("Architecture: BiLSTM (Legacy)", "INFO")
                
        except Exception as e:
            logger.error(f"Failed to inspect checkpoint: {e}")
            # Fall through - will try to load normally and might fail
    
    # Initialize model with correct architecture
    console_log("Loading neural network model...", "MODEL")
    model = DerivOmniModel(settings).to(device)
    model.eval()
    console_log(f"Model ready: {model.count_parameters():,} parameters", "SUCCESS")
    logger.info(f"Model initialized ({model.count_parameters():,} parameters)")

    # Load checkpoint weights
    # Load checkpoint weights
    if checkpoint:
        console_log(f"Loading weights from {checkpoint_name}...", "MODEL")
        # Move state dict to correct device
        # Note: checkpoint is already loaded
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Extract semantic version from manifest for proper shadow trade tracking
        if "manifest" in checkpoint:
            manifest = checkpoint["manifest"]
            model_version = manifest.get("model_version", checkpoint_name)
            git_sha = manifest.get("git_sha", "unknown")[:8]
            logger.info(f"Loaded checkpoint with manifest: version={model_version}, git={git_sha}")
            console_log(f"Checkpoint loaded (v{model_version})", "SUCCESS")
        else:
            # Fallback for old checkpoints without manifest
            model_version = checkpoint_name
            console_log("Checkpoint loaded (no manifest)", "SUCCESS")
        
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        model_version = "live_unversioned"

    # Initialize components
    console_log("Initializing trading components...", "WAIT")
    client = DerivClient(settings)
    # NOTE: ShadowLogger deprecated per architectural audit
    # SQLiteShadowStore is now the sole shadow persistence path

    # SQLite-backed shadow store for FULL CONTEXT CAPTURE
    # This enables the continual learning pipeline by storing tick/candle windows
    shadow_store = SQLiteShadowStore(Path("data_cache/shadow_trades.db"))
    
    # M13: DB Pruning
    try:
        pruned_count = shadow_store.prune(retention_days=settings.system.db_retention_days)
        if pruned_count > 0:
             console_log(f"Pruned {pruned_count} old shadow records", "INFO")
    except Exception as e:
        logger.error(f"DB pruning failed: {e}")
        
    console_log("Shadow store ready (SQLite)", "SUCCESS")

    # Regime veto authority (can block all trades during anomalous conditions)
    # Regime veto authority (can block all trades during anomalous conditions)
    regime_veto = RegimeVeto(
        threshold_caution=settings.hyperparams.regime_caution_threshold,
        threshold_veto=settings.hyperparams.regime_veto_threshold,
    )

    engine = DecisionEngine(
        settings, regime_veto=regime_veto, shadow_store=shadow_store, model_version=model_version
    )

    # Shadow Trade Resolver (Resolves outcomes after configured duration)
    resolver = ShadowTradeResolver(
        shadow_store, 
        duration_minutes=settings.shadow_trade.duration_minutes,
        client=client
    )

    # Initialize Observability Monitors
    model_monitor = ModelHealthMonitor()
    system_monitor = SystemHealthMonitor()
    system_monitor.register_component("model", create_model_health_checker(model_monitor))

    # CANONICAL FEATURE PIPELINE - single entry point for all feature engineering
    feature_builder = FeatureBuilder(settings)
    logger.info(f"Feature pipeline v{feature_builder.get_schema_version()}: canonical path active")

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

        # Create trade executor with SAFETY WRAPPER (production-grade controls)
        # The SafeTradeExecutor provides: rate limiting, P&L caps, kill switch
        console_log("Setting up trade executor with safety controls...", "WAIT")
        # Inject the chosen sizer into the raw executor
        raw_executor = DerivTradeExecutor(client, settings, position_sizer=sizer)

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
        safety_state_file = Path("data_cache/safety_state.db")

        # C06: Inject stake resolver for safety checks
        def stake_resolver(symbol: str, metadata: dict) -> float:
            # Fallback for determining stake if missing in signal
            # We use the sizer's suggestion mechanism
            return sizer.suggest_stake_for_signal(None)

        executor = SafeTradeExecutor(
            raw_executor, 
            safety_config, 
            state_file=safety_state_file,
            stake_resolver=stake_resolver
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
        event_bus = DerivEventAdapter(client)  # Implements MarketEventBus
        symbol = settings.trading.symbol

        # Shared state for heartbeat
        tick_count = 0
        candle_count = 0
        inference_count = 0
        last_tick_time = datetime.now()
        last_tick_log_count = 0  # For periodic tick logging
        
        # H11: Startup buffers and synchronization
        startup_buffer_ticks: list[float] = []
        startup_buffer_candles: list[Any] = []  # Store full event objects
        startup_complete = asyncio.Event()

        # Observability shared components
        from observability.performance_tracker import PerformanceTracker
        from training.auto_retrain import RetrainingTrigger
        perf_tracker = PerformanceTracker()
        retrain_trigger = RetrainingTrigger()

        async def process_ticks():
            """Process tick events from normalized MarketEventBus."""
            nonlocal tick_count, last_tick_time, last_tick_log_count
            first_tick = True
            async for tick_event in event_bus.subscribe_ticks(symbol):
                try:
                    # Update heartbeat timestamp
                    last_tick_time = datetime.now()
                    
                    # H11: Buffer during startup
                    if not startup_complete.is_set():
                        startup_buffer_ticks.append(tick_event.price)
                        # Log sparsely to avoid spam during startup
                        if len(startup_buffer_ticks) % 10 == 0:
                            logger.debug(f"[STARTUP] Buffered {len(startup_buffer_ticks)} ticks")
                        return # Continue to next iteration (skip processing)

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

        async def process_candles():
            """Process candle events from normalized MarketEventBus."""
            
            nonlocal candle_count, inference_count
            first_candle = True
            
            async for candle_event in event_bus.subscribe_candles(symbol, interval=60):
                try:
                    # H11: Buffer during startup
                    if not startup_complete.is_set():
                        startup_buffer_candles.append(candle_event)
                        logger.debug(f"[STARTUP] Buffered candle at {candle_event.timestamp}")
                        return # Continue to next iteration
                        
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
                    now_utc = datetime.now(timezone.utc)
                    latency = (now_utc - candle_event.timestamp).total_seconds()
                    
                    if latency > 5.0:
                        logger.warning(
                            f"[LATENCY] Skipping stale candle (closed {latency:.1f}s ago). "
                            f"Threshold: 5.0s"
                        )
                        console_log(f"Skipping stale candle ({latency:.1f}s lag)", "WARN")
                        continue

                    # Run inference only when: candle closed + buffer ready
                    # M01: Removed flaky wall-clock cooldown. Event-driven is_new_candle is sufficient.
                    if is_new_candle and buffer.is_ready():
                        console_log(
                            f"Candle closed @ {candle_event.close:.2f} - Running inference #{inference_count + 1}... "
                            f"(latency: {latency:.1f}s)",
                            "MODEL",
                        )
                        logger.info(f"Candle closed: running inference (latency: {latency:.3f}s)")

                        try:
                            await run_inference(
                                model,
                                engine,
                                executor,
                                buffer,
                                feature_builder,
                                device,
                                settings,
                                metrics,
                                calibration_monitor=calibration_monitor,
                                trade_tracker=real_trade_tracker,
                            )
                            inference_count += 1
                        except Exception as inf_e:
                            logger.error(f"Inference cycle failed: {inf_e}", exc_info=True)
                            metrics.record_error("inference_failure")

                    # Resolve shadow trades on EVERY candle close (not just during inference)
                    # This ensures 1-minute trades resolve immediately after expiry
                    if is_new_candle:
                        # Use candle timestamp for deterministic resolution (handles lag/replay)
                        # candle_event.timestamp is already a timezone-aware datetime object
                        candle_time = candle_event.timestamp
                        resolved_count = await resolver.resolve_trades(
                            current_price=candle_event.close,
                            current_time=candle_time
                        )
                        if resolved_count > 0:
                            console_log(
                                f"🎯 Resolved {resolved_count} shadow trade(s) - check logs for results",
                                "SUCCESS",
                            )
                            logger.info(f"Resolved {resolved_count} shadow trades this candle")

                except Exception as e:
                    logger.error(f"Error processing candle: {e}", exc_info=True)

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

                # M12: Model Hot-Reloading
                # Check if checkpoint file has been modified (new weights from online training)
                if checkpoint_path and checkpoint_path.exists():
                    try:
                        current_mtime = checkpoint_path.stat().st_mtime
                        # Initialize last_mtime on first run if needed, but it should be set
                        if 'last_ckpt_mtime' not in locals():
                             last_ckpt_mtime = current_mtime
                        
                        if current_mtime > last_ckpt_mtime:
                            console_log(f"Checkpoint update detected! Reloading... ({checkpoint_path.name})", "MODEL")
                            logger.info(f"[HOT-RELOAD] Checkpoint modified at {datetime.fromtimestamp(current_mtime)}")
                            
                            # Load new weights (off-thread to avoid blocking loop? load is fast enough usually)
                            # Actually, torch.load can be slow for large models. Let's do it carefully.
                            new_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                            
                            # Update model
                            model.load_state_dict(new_checkpoint["model_state_dict"])
                            model.eval() # Ensure eval mode
                            
                            # Update version info
                            if "manifest" in new_checkpoint:
                                new_manifest = new_checkpoint["manifest"]
                                new_version = new_manifest.get("model_version", "unknown")
                                logger.info(f"[HOT-RELOAD] Loaded version: {new_version}")
                                console_log(f"Model reloaded: v{new_version}", "SUCCESS")
                                # Update engine's model version if possible (it's immutable usually, but maybe we can update it)
                                # engine.model_version is public? No, it's used in prepare_trade_metadata.
                                # Check decision engine implementation.
                                if hasattr(engine, 'model_version'):
                                    engine.model_version = new_version
                            
                            last_ckpt_mtime = current_mtime
                            
                            # Reset calibration monitor as behavior might change
                            calibration_monitor.errors = [] 
                            calibration_monitor.consecutive_high_count = 0
                            
                    except Exception as e:
                        logger.error(f"[HOT-RELOAD] Failed to reload model: {e}")
                        console_log(f"Hot-reload failed: {e}", "ERROR")

                # Performance and Retraining Checks
                perf_stats = perf_tracker.get_summary()
                latency = perf_stats["latency"]
                should_retrain, retrain_reason = retrain_trigger.should_retrain(shadow_metrics_cache)
                
                # HUMAN-READABLE CONSOLE OUTPUT
                console_log(
                    f"═══ HEARTBEAT ═══════════════════════════════",
                    "HEART",
                )
                console_log(
                    f"Ticks: {tick_count} | Candles: {candle_count} | Inferences: {inference_count}",
                    "DATA",
                )
                console_log(
                    f"Latency: p50={latency['p50']:.1f}ms | p95={latency['p95']:.1f}ms | RAM: {perf_stats['memory_mb']:.1f}MB",
                    "PERF"
                )
                
                # Display Shadow Trade Performance
                if shadow_metrics_cache.resolved_trades > 0:
                    console_log(
                        f"───────── SHADOW TRADING PERFORMANCE ─────────",
                        "INFO",
                    )
                    console_log(
                        f"⚠️  Resolved: {shadow_metrics_cache.resolved_trades} | "
                        f"Win Rate: {shadow_metrics_cache.win_rate * 100:.1f}% | "
                        f"Wins: {shadow_metrics_cache.wins} | Losses: {shadow_metrics_cache.losses}",
                        "SUCCESS" if shadow_metrics_cache.win_rate > 0.5 else "WARN",
                    )
                    console_log(
                        f"{'✅' if shadow_metrics_cache.simulated_pnl > 0 else '❌'} "
                        f"Simulated P&L: ${shadow_metrics_cache.simulated_pnl:.2f} | "
                        f"ROI: {shadow_metrics_cache.simulated_roi:.1f}%",
                        "SUCCESS" if shadow_metrics_cache.simulated_pnl > 0 else "ERROR",
                    )
                    
                    if should_retrain:
                        console_log(f"⚠️ RETRAINING RECOMMENDED: {retrain_reason}", "WARN")

                console_log(
                    f"═══════════════════════════════════════════════",
                    "INFO",
                )

                logger.info(
                    f"[HEARTBEAT] ticks={tick_count}, candles={candle_count}, "
                    f"inferences={inference_count}, stale_sec={stale_seconds:.1f}, "
                    f"trades={stats.get('real', 0)}, shadow={stats.get('shadow', 0)}, "
                    f"vetoed={stats.get('regime_vetoed', 0)}, "
                    f"shadow_only={cal_stats.get('shadow_only_mode', False)}, "
                    f"shadow_win_rate={shadow_metrics_cache.win_rate:.3f}, "
                    f"shadow_pnl={shadow_metrics_cache.simulated_pnl:.2f}"
                )

                # Update system health dashboard
                try:
                    health_data = system_monitor.get_dashboard_data()
                    health_file = log_dir / settings.observability.health_check_file_path
                    # Atomic write
                    tmp_file = health_file.with_suffix(".tmp")
                    health_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(tmp_file, "w") as f:
                        import json
                        json.dump(health_data, f, indent=2)
                    tmp_file.rename(health_file)
                except Exception as e:
                    logger.error(f"Failed to update health dashboard: {e}")

                # Stale data warning
                if stale_seconds > stale_threshold:
                    console_log(
                        f"WARNING: No ticks for {stale_seconds:.1f}s - possible connection issue",
                        "WARN",
                    )
                    logger.warning(f"[STALE] No ticks received for {stale_seconds:.1f}s")

        # START TASKS (Early, to catch events)
        console_log("Starting background tasks (Stream + Heartbeat)...", "WAIT")
        tick_task = asyncio.create_task(process_ticks())
        candle_task = asyncio.create_task(process_candles())
        heartbeat_task = asyncio.create_task(heartbeat())
        
        # Wait a moment for subscription to establish and start buffering
        console_log("Waiting for stream to stabilize...", "WAIT")
        await asyncio.sleep(2.0)

        # Pre-load historical data (While streaming is buffering)
        console_log("=" * 60, "INFO")
        console_log("LOADING HISTORICAL DATA...", "DATA")
        logger.info("[INIT] Fetching historical data while buffering live events...")

        console_log(f"Fetching {tick_len} historical ticks...", "WAIT")
        hist_ticks = await client.get_historical_ticks(count=tick_len)
        buffer.append_ticks(hist_ticks)
        console_log(f"Loaded {buffer.tick_count()} ticks", "SUCCESS")
        logger.info(f"[INIT] Pre-loaded {buffer.tick_count()} ticks")

        console_log(f"Fetching {candle_len} historical candles...", "WAIT")
        hist_candles = await client.get_historical_candles(count=candle_len, interval=60)
        candle_arrays = [
            [
                float(c["open"]),
                float(c["high"]),
                float(c["low"]),
                float(c["close"]),
                0.0,
                float(c["epoch"]),
            ]
            for c in hist_candles
        ]
        buffer.preload_candles(candle_arrays)
        console_log(f"Loaded {buffer.candle_count()} candles", "SUCCESS")
        logger.info(
            f"[INIT] Pre-loaded {buffer.candle_count()} candles. Buffer ready: {buffer.is_ready()}"
        )

        # H11: FLUSH STARTUP BUFFERS (Atomic Handover)
        logger.info("[INIT] Flushing startup buffers...")
        console_log(f"Flushing buffered startup events ({len(startup_buffer_ticks)} ticks, {len(startup_buffer_candles)} candles)...", "DATA")
        
        # Flush Ticks
        for t in startup_buffer_ticks:
            buffer.append_tick(t)
            tick_count += 1
            
        # Flush Candles (with deduplication)
        for c in startup_buffer_candles:
            buffer.update_candle(c)
            candle_count += 1
            
        # Clear buffers and set flag to enable live processing
        startup_buffer_ticks.clear()
        startup_buffer_candles.clear()
        startup_complete.set()
        
        console_log("Startup synchronization complete - LIVE MODE ACTIVE", "SUCCESS")
        logger.info("[INIT] Startup synchronization complete. Live processing active.")

        # Warm up regime detector with historical close prices
        if hasattr(regime_veto, 'update_prices') and candle_arrays:
            import numpy as np
            close_prices = np.array([c[3] for c in candle_arrays])  # Index 3 is close price
            regime_veto.update_prices(close_prices)
            console_log(f"Regime veto warmed up with {len(close_prices)} prices", "SUCCESS")

        # Run initial inference if buffer ready
        if buffer.is_ready():
            console_log("=" * 60, "INFO")
            console_log("RUNNING INITIAL INFERENCE...", "MODEL")
            initial_recon_error = await run_inference(
                model,
                engine,
                executor,
                buffer,
                feature_builder,
                device,
                settings,
                metrics,
                return_recon_error=True,  # Get reconstruction error for validation
                trade_tracker=real_trade_tracker,
            )
            console_log(
                f"Initial inference complete! Reconstruction error: {initial_recon_error:.4f}",
                "SUCCESS",
            )
            
            # Calibration checks
            if initial_recon_error is not None:
                if initial_recon_error > 1.0:
                    logger.critical(
                        f"[CALIBRATION] Reconstruction error {initial_recon_error:.4f} is extremely high! "
                        f"Expected < 1.0. Model may need retraining with normalized features. "
                        f"Trading will continue in CAUTION mode."
                    )
                    console_log("CALIBRATION WARNING: High reconstruction error!", "WARN")
                elif initial_recon_error > 0.3:
                    logger.warning(
                        f"[CALIBRATION] Reconstruction error {initial_recon_error:.4f} exceeds VETO threshold (0.3). "
                        f"Consider retraining or adjusting thresholds."
                    )
                    console_log("CALIBRATION CAUTION: Error above veto threshold", "WARN")
                else:
                    logger.info(
                        f"[HEALTH] Reconstruction error {initial_recon_error:.4f} is within expected range."
                    )
                    console_log("Health check passed", "SUCCESS")
        else:
            console_log(
                f"Buffer not ready. Ticks: {buffer.tick_count()}, Candles: {buffer.candle_count()}",
                "WARN",
            )
            
        # ══════════════════════════════════════════════════════════════════════
        # Wait for tasks (Main Loop)
        # ══════════════════════════════════════════════════════════════════════

        await asyncio.gather(tick_task, candle_task, heartbeat_task)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except (ConnectionError, asyncio.TimeoutError, OSError) as e:
        # Network errors - may be recoverable
        logger.error(f"[NETWORK] Connection error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as e:
        # Inference/model errors
        logger.error(f"[INFERENCE] Model error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    except Exception as e:
        # Unexpected errors - log category for observability
        logger.error(f"[UNEXPECTED] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        await client.disconnect()

        # Log statistics
        stats = engine.get_statistics()
        logger.info(f"Session statistics: {stats}")

        if executor:
            # Log safety wrapper statistics
            safety_stats = executor.get_safety_statistics()
            logger.info(f"Safety statistics: {safety_stats}")

        # Export final metrics
        logger.info(f"Final metrics: {metrics.export()}")

    return 0


async def run_inference(
    model,
    engine,
    executor,
    buffer,
    feature_builder,
    device,
    settings,
    metrics,
    return_recon_error: bool = False,
    calibration_monitor: CalibrationMonitor | None = None,
    trade_tracker: Any | None = None,  # RealTradeTracker for outcome tracking
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
        buffer: MarketDataBuffer with tick and candle data
        feature_builder: FeatureBuilder for canonical feature engineering
        device: Torch device for inference
        settings: Application settings
        metrics: TradingMetrics instance for recording metrics
        return_recon_error: If True, return reconstruction error for validation
        calibration_monitor: Optional CalibrationMonitor for graceful degradation

    Returns:
        If return_recon_error is True, returns the reconstruction error float.
        Otherwise returns None.
    """
    import time

    inference_start = time.perf_counter()

    # CANONICAL FEATURE PROCESSING - extract arrays from buffer
    t_np = buffer.get_ticks_array()
    c_np = buffer.get_candles_array()

    features = feature_builder.build(ticks=t_np, candles=c_np)

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
    real_trades = engine.process_with_context(
        probs=sample_probs,
        reconstruction_error=reconstruction_error,
        tick_window=t_np,
        candle_window=c_np,
        entry_price=entry_price,
        market_data={"ticks_count": buffer.tick_count(), "candles_count": buffer.candle_count()},
    )

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

    # Execute trades via abstraction (broker-isolated)
    for signal in real_trades:
        logger.info(f"EXECUTING: {signal}")
        
        # ATOMIC EXECUTION SAFETY: Record intent BEFORE API call
        # If app crashes after API succeeds but before confirmation,
        # the trade will be in ATTEMPTING state for investigation
        intent_id = None
        stake = settings.trading.stake_amount
        contract_type = getattr(signal, "contract_type", "unknown")
        
        if trade_tracker and hasattr(trade_tracker, '_store'):
            import uuid
            intent_id = f"intent_{uuid.uuid4().hex[:16]}"
            trade_tracker._store.prepare_trade_intent(
                intent_id=intent_id,
                direction=signal.direction,
                entry_price=entry_price,
                stake=stake,
                probability=signal.probability,
                contract_type=str(contract_type),
            )
        
        exec_start = time.perf_counter()
        result = await executor.execute(signal)
        exec_latency = time.perf_counter() - exec_start

        # Record execution latency
        metrics.record_execution_latency(exec_latency)

        if result.success:
            logger.info(f"Trade executed: contract_id={result.contract_id}")
            metrics.record_trade_attempt(outcome="executed", contract_type=contract_type)
            
            # ATOMIC EXECUTION SAFETY: Confirm trade with real contract ID
            if trade_tracker and result.contract_id:
                if intent_id:
                    # Update ATTEMPTING -> CONFIRMED with real contract_id
                    trade_tracker._store.confirm_trade(intent_id, result.contract_id)
                
                # Continue with normal trade registration flow
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
            
            # ATOMIC EXECUTION SAFETY: Remove ATTEMPTING record on failure
            # The trade never executed, so no phantom risk
            if intent_id and trade_tracker and hasattr(trade_tracker, '_store'):
                trade_tracker._store.remove_trade(intent_id)
            
            # Categorize the failure
            if "rate limit" in (result.error or "").lower():
                metrics.record_trade_attempt(
                    outcome="blocked_rate_limit", contract_type=contract_type
                )
            elif "kill switch" in (result.error or "").lower():
                metrics.record_trade_attempt(
                    outcome="blocked_kill_switch", contract_type=contract_type
                )
            elif "loss limit" in (result.error or "").lower():
                metrics.record_trade_attempt(outcome="blocked_pnl_cap", contract_type=contract_type)
            else:
                metrics.record_trade_attempt(outcome="failed", contract_type=contract_type)

    # Return reconstruction error for startup validation if requested
    if return_recon_error:
        return reconstruction_error
    return None


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
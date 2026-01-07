"""
LiveTradingContext - Central dependency injection container.

Replaces implicit closure state with explicit context object for:
- Loose coupling between extracted modules
- Testability via dependency injection
- Observable system state
- Resource management via async context manager

Reference: docs/plans/live_script_refactoring.md Section 3.1
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config.settings import Settings
    from data.buffer import MarketDataBuffer
    from data.features import FeatureBuilder
    from data.ingestion.client import DerivClient
    from execution.decision_logic import DecisionEngine
    from execution.safety import SafeTradeExecutor
    from execution.signal_adapter_service import SignalAdapterService
    from execution.sqlite_shadow_store import SQLiteShadowStore
    from execution.shadow_resolution import ShadowTradeResolver
    from execution.real_trade_tracker import RealTradeTracker
    from models.core import DerivOmniModel
    from observability import TradingMetrics
    from observability.calibration import CalibrationMonitor
    from observability.dashboard import SystemHealthMonitor
    from training.inference import InferenceOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class HotReloadState:
    """
    State for hot-reload with exponential backoff.
    
    Critical for M12 (Atomic Hot-Reload) safety requirement.
    """
    last_ckpt_mtime: float = 0.0
    consecutive_failures: int = 0
    backoff_until: float = 0.0


@dataclass
class LiveTradingContext:
    """
    Central context object for live trading system.
    
    Design principles (from research):
    - Dependency injection for loose coupling
    - Explicit state management (no nonlocal)
    - Async-safe via asyncio.Event for shutdown
    - Testable via mocked dependencies
    
    Safety Requirements Preserved:
    - Circuit Breaker Sync [C-01]: executor respects client circuit state
    - Atomic Hot-Reload [M12]: hot_reload_state tracks failures + backoff
    - Staleness Veto [H6]: settings.max_candle_latency checked in handlers
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # Core Configuration
    # ═══════════════════════════════════════════════════════════════════════
    settings: Settings
    device: Any  # torch.device
    checkpoint_path: Path | None = None
    
    # ═══════════════════════════════════════════════════════════════════════
    # Service Dependencies (Immutable after initialization)
    # ═══════════════════════════════════════════════════════════════════════
    client: DerivClient | None = None
    model: DerivOmniModel | None = None
    engine: DecisionEngine | None = None
    executor: SafeTradeExecutor | None = None
    buffer: MarketDataBuffer | None = None
    feature_builder: FeatureBuilder | None = None
    shadow_store: SQLiteShadowStore | None = None
    
    # ═══════════════════════════════════════════════════════════════════════
    # Orchestration Components
    # ═══════════════════════════════════════════════════════════════════════
    orchestrator: InferenceOrchestrator | None = None
    resolver: ShadowTradeResolver | None = None
    trade_tracker: RealTradeTracker | None = None
    strategy_adapter: SignalAdapterService | None = None
    
    # ═══════════════════════════════════════════════════════════════════════
    # Monitors & Observability
    # ═══════════════════════════════════════════════════════════════════════
    calibration_monitor: CalibrationMonitor | None = None
    system_monitor: SystemHealthMonitor | None = None
    metrics: TradingMetrics | None = None
    
    # ═══════════════════════════════════════════════════════════════════════
    # Shared Mutable State (Replaces nonlocal variables)
    # ═══════════════════════════════════════════════════════════════════════
    tick_count: int = 0
    candle_count: int = 0
    inference_count: int = 0
    last_tick_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_inference_time: float = 0.0
    last_candle_progress_time: float = 0.0
    last_inference_progress_time: float = 0.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # Hot Reload State (Critical for M12 Safety)
    # ═══════════════════════════════════════════════════════════════════════
    hot_reload_state: HotReloadState = field(default_factory=HotReloadState)
    
    # ═══════════════════════════════════════════════════════════════════════
    # Graceful Shutdown Support
    # ═══════════════════════════════════════════════════════════════════════
    shutdown_event: asyncio.Event = field(default_factory=asyncio.Event)
    startup_complete: asyncio.Event = field(default_factory=asyncio.Event)
    
    # ═══════════════════════════════════════════════════════════════════════
    # CLI Arguments (Preserved for compatibility)
    # ═══════════════════════════════════════════════════════════════════════
    args: Any | None = None  # argparse.Namespace
    
    def __post_init__(self):
        """Validate critical dependencies are set."""
        logger.debug("LiveTradingContext initialized")
    
    @property
    def is_ready(self) -> bool:
        """Check if all critical components are initialized."""
        return all([
            self.client is not None,
            self.model is not None,
            self.engine is not None,
            self.executor is not None,
            self.buffer is not None,
        ])
    
    @property
    def is_shadow_only(self) -> bool:
        """Check if system is in shadow-only mode (no real trades)."""
        if self.calibration_monitor is None:
            return True  # Default to shadow if no monitor
        return self.calibration_monitor.shadow_only_mode
    
    def increment_tick_count(self) -> int:
        """Thread-safe tick counter increment."""
        self.tick_count += 1
        self.last_tick_time = datetime.now(timezone.utc)
        return self.tick_count
    
    def increment_candle_count(self) -> int:
        """Thread-safe candle counter increment."""
        self.candle_count += 1
        return self.candle_count
    
    def increment_inference_count(self) -> int:
        """Thread-safe inference counter increment."""
        self.inference_count += 1
        return self.inference_count

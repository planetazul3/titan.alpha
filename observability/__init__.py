"""
Observability metrics for trading system.

Provides Prometheus-compatible metrics for monitoring:
- Trade counters (executed, blocked, vetoed)
- Latency histograms (inference, execution)
- Gauge metrics (P&L, reconstruction error)

Usage:
    >>> from observability.metrics import TradingMetrics
    >>> metrics = TradingMetrics()
    >>> metrics.record_trade_attempt(outcome='executed', contract_type='RISE_FALL')
    >>> metrics.record_inference_latency(0.015)  # 15ms
    >>> metrics.export()  # Get current metrics

Optional Prometheus integration:
    >>> from observability.metrics import start_prometheus_server
    >>> start_prometheus_server(port=8000)  # Expose /metrics endpoint
"""

import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try importing prometheus_client (optional dependency)
try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logger.debug("prometheus_client not available, using internal metrics only")


@dataclass
class HistogramBucket:
    """Histogram bucket for latency tracking."""

    le: float  # Less than or equal
    count: int = 0


class InternalHistogram:
    """Simple histogram implementation (no Prometheus dependency)."""

    # Default latency buckets in seconds
    DEFAULT_BUCKETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(self, name: str, buckets: list[float] | None = None):
        self.name = name
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._counts = [0] * len(self.buckets)
        self._sum = 0.0
        self._count = 0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        """Record an observation."""
        with self._lock:
            for i, b in enumerate(self.buckets):
                if value <= b:
                    self._counts[i] += 1
            self._sum += value
            self._count += 1

    def get_stats(self) -> dict[str, Any]:
        """Get histogram statistics."""
        with self._lock:
            return {
                "name": self.name,
                "count": self._count,
                "sum": self._sum,
                "mean": self._sum / self._count if self._count > 0 else 0.0,
                "buckets": {f"le_{b}": self._counts[i] for i, b in enumerate(self.buckets)},
            }

    def reset(self) -> None:
        """Reset histogram."""
        with self._lock:
            self._counts = [0] * len(self.buckets)
            self._sum = 0.0
            self._count = 0


class InternalCounter:
    """Simple counter implementation (no Prometheus dependency)."""

    def __init__(self, name: str, labels: list[str] | None = None):
        self.name = name
        self.labels = labels or []
        self._values: dict[tuple, int] = defaultdict(int)
        self._lock = threading.Lock()

    def inc(self, amount: int = 1, **label_values) -> None:
        """Increment counter."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        with self._lock:
            self._values[key] += amount

    def get(self, **label_values) -> int:
        """Get counter value for labels."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        with self._lock:
            return self._values[key]

    def get_all(self) -> dict[str, int]:
        """Get all counter values."""
        with self._lock:
            if not self.labels:
                return {"total": self._values.get((), 0)}
            return {"_".join(str(v) for v in k): v for k, v in self._values.items()}

    def reset(self) -> None:
        """Reset counter."""
        with self._lock:
            self._values.clear()


class InternalGauge:
    """Simple gauge implementation (no Prometheus dependency)."""

    def __init__(self, name: str):
        self.name = name
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        """Set gauge value."""
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0) -> None:
        """Increment gauge."""
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        """Decrement gauge."""
        with self._lock:
            self._value -= amount

    def get(self) -> float:
        """Get gauge value."""
        with self._lock:
            return self._value


class TradingMetrics:
    """
    Trading system metrics collector.

    Provides both internal metrics (always available) and optional
    Prometheus integration when prometheus_client is installed.

    Metrics tracked:
    - Trade counters: executed, blocked (rate_limit, kill_switch, pnl_cap), vetoed
    - Latency histograms: inference, execution, decision
    - Gauges: daily_pnl, reconstruction_error, account_balance
    """

    def __init__(self, enable_prometheus: bool = True):
        """
        Initialize metrics collector.

        Args:
            enable_prometheus: If True and prometheus_client available, use it
        """
        self.use_prometheus = enable_prometheus and HAS_PROMETHEUS

        # Initialize internal metrics (always available)
        self._init_internal_metrics()

        # Initialize Prometheus metrics if available
        if self.use_prometheus:
            self._init_prometheus_metrics()

        logger.info(
            f"TradingMetrics initialized (prometheus={'enabled' if self.use_prometheus else 'disabled'})"
        )

    def _init_internal_metrics(self) -> None:
        """Initialize internal metrics."""
        # Counters
        self.trades_total = InternalCounter("trades_total", ["outcome", "contract_type"])
        self.errors_total = InternalCounter("errors_total", ["type"])
        self.regime_assessments = InternalCounter("regime_assessments", ["state"])

        # Histograms
        self.inference_latency = InternalHistogram("inference_latency_seconds")
        self.execution_latency = InternalHistogram("execution_latency_seconds")
        self.decision_latency = InternalHistogram("decision_latency_seconds")

        # Gauges
        self.daily_pnl = InternalGauge("daily_pnl_usd")
        self.reconstruction_error = InternalGauge("reconstruction_error")
        self.account_balance = InternalGauge("account_balance_usd")

        # Alert thresholds
        self._alert_callbacks: list[Callable] = []
        self.reconstruction_error_threshold = 0.3  # Veto threshold

    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        self.prom_trades = Counter(
            "xtitan_trades_total", "Total trades attempted", ["outcome", "contract_type"]
        )

        self.prom_regime = Counter(
            "xtitan_regime_assessments_total", "Regime assessments by state", ["state"]
        )

        self.prom_inference_latency = Histogram(
            "xtitan_inference_latency_seconds",
            "Model inference latency",
            buckets=InternalHistogram.DEFAULT_BUCKETS,
        )

        self.prom_execution_latency = Histogram(
            "xtitan_execution_latency_seconds",
            "Trade execution latency",
            buckets=InternalHistogram.DEFAULT_BUCKETS,
        )

        self.prom_daily_pnl = Gauge("xtitan_daily_pnl_usd", "Daily profit/loss in USD")

        self.prom_recon_error = Gauge(
            "xtitan_reconstruction_error", "Volatility model reconstruction error"
        )

        self.prom_balance = Gauge("xtitan_account_balance_usd", "Current account balance in USD")

        self.prom_errors = Counter(
            "xtitan_errors_total", "Total application errors", ["type"]
        )

    def record_error(self, error_type: str) -> None:
        """
        Record an application error.

        Args:
            error_type: Error category (e.g., 'inference_failure', 'connection_error')
        """
        self.errors_total.inc(type=error_type)

        if self.use_prometheus:
            self.prom_errors.labels(type=error_type).inc()



    def record_trade_attempt(self, outcome: str, contract_type: str = "unknown") -> None:
        """
        Record a trade attempt.

        Args:
            outcome: 'executed', 'blocked_rate_limit', 'blocked_kill_switch',
                     'blocked_pnl_cap', 'vetoed'
            contract_type: 'RISE_FALL', 'TOUCH_NO_TOUCH', etc.
        """
        self.trades_total.inc(outcome=outcome, contract_type=contract_type)

        if self.use_prometheus:
            self.prom_trades.labels(outcome=outcome, contract_type=contract_type).inc()

    def record_regime_assessment(self, state: str) -> None:
        """
        Record a regime assessment.

        Args:
            state: 'TRUSTED', 'CAUTION', 'VETO'
        """
        self.regime_assessments.inc(state=state)

        if self.use_prometheus:
            self.prom_regime.labels(state=state).inc()

    def record_inference_latency(self, seconds: float) -> None:
        """
        Record model inference latency.

        Args:
            seconds: Latency in seconds
        """
        self.inference_latency.observe(seconds)

        if self.use_prometheus:
            self.prom_inference_latency.observe(seconds)

    def record_execution_latency(self, seconds: float) -> None:
        """
        Record trade execution latency.

        Args:
            seconds: Latency in seconds
        """
        self.execution_latency.observe(seconds)

        if self.use_prometheus:
            self.prom_execution_latency.observe(seconds)

    def record_decision_latency(self, seconds: float) -> None:
        """
        Record decision engine latency.

        Args:
            seconds: Latency in seconds
        """
        self.decision_latency.observe(seconds)

    def set_daily_pnl(self, value: float) -> None:
        """
        Set current daily P&L.

        Args:
            value: P&L in USD
        """
        self.daily_pnl.set(value)

        if self.use_prometheus:
            self.prom_daily_pnl.set(value)

    def set_reconstruction_error(self, value: float) -> None:
        """
        Set current reconstruction error and check alerts.

        Args:
            value: Reconstruction error value
        """
        self.reconstruction_error.set(value)

        if self.use_prometheus:
            self.prom_recon_error.set(value)

        # Check alert threshold
        if value >= self.reconstruction_error_threshold:
            self._trigger_alert(
                "reconstruction_error_high",
                f"Reconstruction error {value:.4f} >= threshold {self.reconstruction_error_threshold}",
            )

    def set_account_balance(self, value: float) -> None:
        """
        Set current account balance.

        Args:
            value: Balance in USD
        """
        self.account_balance.set(value)

        if self.use_prometheus:
            self.prom_balance.set(value)

    def register_alert_callback(self, callback: Callable[[str, str], None]) -> None:
        """
        Register callback for alerts.

        Args:
            callback: Function called with (alert_name, message) on alert
        """
        self._alert_callbacks.append(callback)

    def _trigger_alert(self, name: str, message: str) -> None:
        """Trigger alert callbacks."""
        logger.warning(f"ALERT [{name}]: {message}")

        for callback in self._alert_callbacks:
            try:
                callback(name, message)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def export(self) -> dict[str, Any]:
        """
        Export all metrics as dictionary.

        Returns:
            Dict with all metric values
        """
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "counters": {
                "trades": self.trades_total.get_all(),
                "regime_assessments": self.regime_assessments.get_all(),
            },
            "histograms": {
                "inference_latency": self.inference_latency.get_stats(),
                "execution_latency": self.execution_latency.get_stats(),
                "decision_latency": self.decision_latency.get_stats(),
            },
            "gauges": {
                "daily_pnl": self.daily_pnl.get(),
                "reconstruction_error": self.reconstruction_error.get(),
                "account_balance": self.account_balance.get(),
            },
        }

    def reset(self) -> None:
        """Reset all internal metrics (for testing)."""
        self.trades_total.reset()
        self.regime_assessments.reset()
        self.inference_latency.reset()
        self.execution_latency.reset()
        self.decision_latency.reset()
        self.daily_pnl.set(0.0)
        self.reconstruction_error.set(0.0)
        self.account_balance.set(0.0)


def start_prometheus_server(port: int = 8000) -> None:
    """
    Start Prometheus HTTP server.

    Args:
        port: Port to listen on

    Raises:
        ImportError: If prometheus_client not installed
    """
    if not HAS_PROMETHEUS:
        raise ImportError(
            "prometheus_client required for Prometheus server: pip install prometheus_client"
        )

    start_http_server(port)
    logger.info(f"Prometheus metrics server started on port {port}")


class LatencyTimer:
    """
    Context manager for timing operations.

    Usage:
        >>> with LatencyTimer(metrics.record_inference_latency):
        ...     result = model(inputs)
    """

    def __init__(self, callback: Callable[[float], None]):
        self.callback = callback
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        self.callback(elapsed)
        return False


# Global metrics instance (optional singleton pattern)
_global_metrics: TradingMetrics | None = None


def get_metrics() -> TradingMetrics:
    """Get global metrics instance (creates if needed)."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = TradingMetrics()
    return _global_metrics

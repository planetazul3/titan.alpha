import logging

logger = logging.getLogger(__name__)

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
        self.shadow_only_reason = ""
        self.alert_escalation_level = 0  # 0=none, 1=warning, 2=critical

    def reset(self) -> None:
        """Reset monitor state (e.g. after model reload)."""
        self.errors.clear()
        self.consecutive_high_count = 0
        self.shadow_only_mode = False
        self.shadow_only_reason = ""
        self.alert_escalation_level = 0

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

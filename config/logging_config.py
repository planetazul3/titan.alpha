"""
Standardized logging configuration for DerivOmniModel.

Provides:
- Consistent JSON-formatted logs for production (machine-readable)
- Human-readable console logs for development
- Structured log categories for filtering and alerting

Usage:
    >>> from config.logging_config import setup_logging, get_logger
    >>> setup_logging(json_format=True, level="INFO")
    >>> logger = get_logger(__name__)
    >>> logger.info("Trade executed", extra={"contract_id": "123", "pnl": 5.0})
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
import re


# Log categories for filtering
class LogCategory:
    """Standard log categories for observability."""

    NETWORK = "[NETWORK]"  # Connection, API calls
    INFERENCE = "[INFERENCE]"  # Model predictions
    TRADE = "[TRADE]"  # Trade execution
    SAFETY = "[SAFETY]"  # Rate limits, kill switch
    REGIME = "[REGIME]"  # Regime veto assessments
    CALIBRATION = "[CALIBRATION]"  # Model calibration issues
    HEARTBEAT = "[HEARTBEAT]"  # Periodic status
    DATA = "[DATA]"  # Market data processing


class TokenSanitizer(logging.Filter):
    """
    Sanitizes sensitive tokens (e.g. API keys) from log records.
    
    Specifically targets 'authorize' messages in Deriv API traffic.
    """
    
    # Regex to catch "authorize": "TOKEN" patterns in JSON strings
    # Captures: "authorize": "..." -> "authorize": "***"
    _AUTH_PATTERN = re.compile(r'("authorize"\s*:\s*)"[^"]+"')

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._AUTH_PATTERN.sub(r'\1"***"', record.msg)
        return True


class JsonFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.

    Produces machine-readable logs suitable for ingestion by
    log aggregation systems (ELK, Splunk, CloudWatch, etc.)

    Example output:
        {"timestamp": "2024-01-01T12:00:00Z", "level": "INFO",
         "logger": "live", "message": "Trade executed",
         "contract_id": "123", "pnl": 5.0}
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields (structured data)
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in (
                    "name",
                    "msg",
                    "args",
                    "created",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "stack_info",
                    "exc_info",
                    "exc_text",
                    "thread",
                    "threadName",
                    "message",
                    "taskName",
                ):
                    log_data[key] = value

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class ColorFormatter(logging.Formatter):
    """
    Colored console formatter for human-readable output.

    Uses ANSI colors: DEBUG=gray, INFO=green, WARNING=yellow,
    ERROR=red, CRITICAL=bold red
    """

    COLORS = {
        "DEBUG": "\033[90m",  # Gray
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[1;91m",  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname:8}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO", json_format: bool = False, log_file: Path | None = None
) -> None:
    """
    Configure standardized logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: If True, use JSON format for machine parsing
        log_file: Optional path to write logs to file

    Example:
        >>> setup_logging(json_format=True, level="DEBUG")
        >>> logger = logging.getLogger("live")
        >>> logger.info("Starting", extra={"symbol": "R_100"})
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers = []

    # Token Sanitizer
    sanitizer = TokenSanitizer()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.addFilter(sanitizer)  # Apply sanitizer

    if json_format:
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler.setFormatter(
            ColorFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    root_logger.addHandler(console_handler)

    # File handler (always JSON for parsing)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.addFilter(sanitizer)  # Apply sanitizer
        file_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the standard configuration.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

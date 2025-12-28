"""
Standardized logging configuration for DerivOmniModel.

Provides:
- Consistent JSON-formatted logs for production (machine-readable)
- Human-readable console logs for development
- Structured log categories for filtering and alerting
- Log file rotation for long-running processes
- Log cleanup for retention management

Usage:
    >>> from config.logging_config import setup_logging, get_logger, cleanup_logs
    >>> log_file = setup_logging(script_name="live", level="INFO")
    >>> logger = get_logger(__name__)
    >>> logger.info("Trade executed", extra={"contract_id": "123", "pnl": 5.0})
"""

import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path


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


# Sensitive keys to sanitize in logs
SENSITIVE_KEYS = frozenset([
    "authorize", "api_key", "token", "password", "secret", 
    "apikey", "api-key", "auth_token", "access_token"
])


class TokenSanitizer(logging.Filter):
    """
    Sanitizes sensitive tokens (e.g. API keys) from log records.
    
    Targets multiple sensitive key patterns in JSON-like strings.
    """
    
    # Regex patterns for sensitive values
    _PATTERNS = [
        re.compile(rf'("{key}"\s*:\s*)"[^"]*"', re.IGNORECASE)
        for key in SENSITIVE_KEYS
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pattern in self._PATTERNS:
                record.msg = pattern.sub(r'\1"***"', record.msg)
        
        # Also sanitize extra fields
        if hasattr(record, "__dict__"):
            for key in list(record.__dict__.keys()):
                if key.lower() in SENSITIVE_KEYS:
                    record.__dict__[key] = "***"
        
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

    # Standard LogRecord attributes to exclude from extra fields
    _EXCLUDED_ATTRS = frozenset([
        "name", "msg", "args", "created", "filename", "funcName",
        "levelname", "levelno", "lineno", "module", "msecs",
        "pathname", "process", "processName", "relativeCreated",
        "stack_info", "exc_info", "exc_text", "thread", "threadName",
        "message", "taskName",
    ])

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields (structured data)
        for key, value in record.__dict__.items():
            if key not in self._EXCLUDED_ATTRS:
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


def _find_project_root() -> Path:
    """Find project root by looking for marker files."""
    current_dir = Path(__file__).resolve().parent
    
    for parent in [current_dir] + list(current_dir.parents):
        if any((parent / marker).exists() for marker in [".git", ".env", "pyproject.toml"]):
            return parent
    
    return Path.cwd()


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Path | str | None = None,
    script_name: str | None = None,
    log_dir: Path | str | None = None,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
) -> Path | None:
    """
    Configure standardized logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: If True, use JSON format for console output
        log_file: Explicit path to log file (overrides script_name)
        script_name: Generate timestamped log file with this name
        log_dir: Directory for log files (default: {project_root}/logs)
        max_bytes: Max file size before rotation (default: 10MB)
        backup_count: Number of rotated files to keep (default: 5)

    Returns:
        Path to log file if file logging enabled, None otherwise

    Example:
        >>> log_file = setup_logging(script_name="live", level="INFO")
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
    console_handler.addFilter(sanitizer)

    if json_format:
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler.setFormatter(
            ColorFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    root_logger.addHandler(console_handler)

    # Determine log file path
    actual_log_file: Path | None = None
    
    if log_file:
        actual_log_file = Path(log_file)
        actual_log_file.parent.mkdir(parents=True, exist_ok=True)
    elif script_name:
        if log_dir:
            log_directory = Path(log_dir)
        else:
            log_directory = _find_project_root() / "logs"
        log_directory.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        actual_log_file = log_directory / f"{script_name}_{timestamp}.log"

    # File handler with rotation (always JSON for parsing)
    if actual_log_file:
        file_handler = RotatingFileHandler(
            actual_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.addFilter(sanitizer)
        file_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"Logging initialized. Writing to {actual_log_file}")

    return actual_log_file


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the standard configuration.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def cleanup_logs(log_dir: Path | str, retention_days: int = 7) -> int:
    """
    Remove log files older than retention period.
    
    Args:
        log_dir: Directory containing logs
        retention_days: Retention period in days
        
    Returns:
        Number of files deleted
    """
    log_path = Path(log_dir)
    
    if not log_path.exists():
        return 0
        
    cutoff_time = time.time() - (retention_days * 86400)
    deleted_count = 0
    
    try:
        for log_file in log_path.glob("*.log*"):  # Include rotated files
            try:
                if log_file.is_file() and log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    deleted_count += 1
            except Exception as e:
                print(f"Failed to delete old log {log_file}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error during log cleanup: {e}", file=sys.stderr)
        
    return deleted_count

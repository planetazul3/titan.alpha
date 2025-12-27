#!/usr/bin/env python3
"""
Rich console logging utilities for all scripts.

Provides emoji-based visual indicators for better UX when running scripts.
"""

from datetime import datetime


def console_log(message: str, level: str = "INFO", symbol: str = "●"):
    """
    Print rich console output with timestamp for user visibility.

    This function prints directly to stdout with flush for immediate visibility,
    complementing the file-based logger.

    Args:
        message: The message to display
        level: Log level (INFO, WARN, ERROR, SUCCESS, WAIT, TRADE, HEART, DATA, MODEL, NET, TRAIN)
        symbol: Visual indicator symbol (fallback)
    """
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Color and symbol mapping for visual distinction
    level_styles = {
        "INFO": "📊",
        "WARN": "⚠️ ",
        "ERROR": "❌",
        "SUCCESS": "✅",
        "WAIT": "⏳",
        "TRADE": "💰",
        "HEART": "💓",
        "DATA": "📈",
        "MODEL": "🧠",
        "NET": "🌐",
        "TRAIN": "🏋️",
        "SAVE": "💾",
        "LOAD": "📂",
        "PROGRESS": "🔄",
    }

    sym = level_styles.get(level, symbol)
    print(f"[{timestamp}] {sym} {message}", flush=True)


def console_header(title: str):
    """Print a header section."""
    console_log("=" * 60, "INFO")
    console_log(title, "SUCCESS")
    console_log("=" * 60, "INFO")


def console_separator():
    """Print a visual separator."""
    console_log("-" * 40, "INFO")


def format_size(bytes_size: float) -> str:
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} TB"


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.1f}s"
    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m {int(remaining_seconds)}s"

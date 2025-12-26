#!/usr/bin/env python3
"""
Graceful shutdown utilities for all scripts.

Provides clean handling of Ctrl+C (SIGINT) and other termination signals
to avoid ugly tracebacks.
"""

import asyncio
import signal
import sys
from collections.abc import Callable
from datetime import datetime
from typing import Any


class GracefulShutdown:
    """
    Context manager and signal handler for graceful script shutdown.

    Usage:
        with GracefulShutdown() as shutdown:
            while not shutdown.requested:
                # do work
                pass

    Or for async:
        async with GracefulShutdown() as shutdown:
            await some_long_running_task()
    """

    def __init__(self, callback: Callable | None = None):
        """
        Args:
            callback: Optional callback to run on shutdown request
        """
        self.requested = False
        self.callback = callback
        self._original_handlers: dict[int, Any] = {}

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        if self.requested:
            # Second Ctrl+C - force exit
            console_print("Force exit requested.", "WARN")
            sys.exit(1)

        self.requested = True
        signal_name = signal.Signals(signum).name
        console_print(f"Shutdown requested ({signal_name}). Cleaning up...", "WARN")

        if self.callback:
            self.callback()

    def __enter__(self):
        """Install signal handlers."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._original_handlers[sig] = signal.signal(sig, self._signal_handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original handlers and suppress KeyboardInterrupt."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)

        # Suppress KeyboardInterrupt traceback
        if exc_type is KeyboardInterrupt:
            console_print("Shutdown complete.", "SUCCESS")
            return True  # Suppress the exception

        return False

    async def __aenter__(self):
        """Async context manager entry."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return self.__exit__(exc_type, exc_val, exc_tb)


def console_print(message: str, level: str = "INFO"):
    """Print with timestamp (minimal version for shutdown module)."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    symbols = {
        "INFO": "📊",
        "WARN": "⚠️ ",
        "ERROR": "❌",
        "SUCCESS": "✅",
    }
    sym = symbols.get(level, "●")
    print(f"[{timestamp}] {sym} {message}", flush=True)


def run_with_graceful_shutdown(main_func, *args, **kwargs):
    """
    Run a synchronous function with graceful shutdown handling.

    Usage:
        def main():
            # your code here
            return 0

        if __name__ == "__main__":
            run_with_graceful_shutdown(main)
    """
    with GracefulShutdown():
        try:
            result = main_func(*args, **kwargs)
            return result
        except KeyboardInterrupt:
            console_print("Interrupted by user.", "WARN")
            return 130  # Standard exit code for SIGINT


def run_async_with_graceful_shutdown(coro_func, *args, **kwargs):
    """
    Run an async function with graceful shutdown handling.

    Usage:
        async def main():
            # your async code here
            return 0

        if __name__ == "__main__":
            run_async_with_graceful_shutdown(main())
    """
    with GracefulShutdown():
        try:
            return asyncio.run(coro_func)
        except KeyboardInterrupt:
            console_print("Interrupted by user.", "WARN")
            return 130
        except asyncio.CancelledError:
            console_print("Async tasks cancelled.", "WARN")
            return 130

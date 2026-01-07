"""
ShutdownHandler - Graceful shutdown with signal handling.

Implements production best practices for async shutdown:
- Use asyncio.Event to coordinate shutdown (not loop.stop())
- Register handlers via loop.add_signal_handler()
- Allow in-flight operations to complete
- Timeout for forced shutdown (prevents hang)
- Gather all tasks with return_exceptions=True

Reference: docs/adr/009-live-script-modularization.md
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from scripts.context import LiveTradingContext

logger = logging.getLogger(__name__)

# Exit codes following Unix conventions
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_SIGINT = 130  # 128 + SIGINT (2)
EXIT_SIGTERM = 143  # 128 + SIGTERM (15)


class ShutdownHandler:
    """
    Handles SIGTERM and SIGINT for graceful shutdown.
    
    Best practices (from research):
    - Use asyncio.Event to signal shutdown (not loop.stop())
    - Register handlers via loop.add_signal_handler()
    - Allow in-flight operations to complete
    - Timeout for forced shutdown (prevents hang)
    - Gather all tasks with return_exceptions=True
    
    Usage:
        async with TradingStackManager() as context:
            handler = ShutdownHandler(context)
            handler.setup_signal_handlers()
            handler.register_cleanup(context.client.disconnect)
            # ... main loop
    """
    
    def __init__(self, context: LiveTradingContext, timeout: float = 30.0):
        """
        Initialize shutdown handler.
        
        Args:
            context: Trading context with shutdown_event
            timeout: Maximum seconds to wait for clean shutdown
        """
        self.context = context
        self.timeout = timeout
        self._shutdown_actions: list[Callable[[], Awaitable[None]]] = []
        self._shutdown_started = False
        self._received_signal: signal.Signals | None = None
    
    def register_cleanup(self, action: Callable[[], Awaitable[None]]) -> None:
        """
        Register cleanup actions to run during shutdown.
        
        Actions are executed in LIFO order (last registered = first executed).
        
        Args:
            action: Async callable to execute during shutdown
        """
        self._shutdown_actions.append(action)
        logger.debug(f"Registered cleanup action: {action.__name__ if hasattr(action, '__name__') else action}")
    
    def setup_signal_handlers(self) -> None:
        """
        Register signal handlers for SIGTERM and SIGINT.
        
        Note: asyncio.run() already handles SIGINT by cancelling the main task,
        but SIGTERM requires explicit registration for containerized environments.
        
        Must be called from within the async event loop (after asyncio.run starts).
        """
        try:
            loop = asyncio.get_running_loop()
            
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(
                        self._handle_shutdown(s),
                        name=f"shutdown_{s.name}"
                    )
                )
            
            logger.info(f"Signal handlers registered for SIGTERM and SIGINT (timeout={self.timeout}s)")
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            logger.warning("Signal handlers not supported on this platform (Windows?)")
    
    async def _handle_shutdown(self, sig: signal.Signals) -> None:
        """
        Graceful shutdown sequence:
        1. Set shutdown event (stops new operations)
        2. Wait for in-flight operations
        3. Run cleanup actions
        4. Cancel remaining tasks
        5. Force exit if timeout exceeded
        
        Args:
            sig: Signal that triggered shutdown
        """
        if self._shutdown_started:
            logger.warning(f"Received {sig.name} again, forcing immediate exit")
            raise SystemExit(EXIT_ERROR)
        
        self._shutdown_started = True
        self._received_signal = sig
        logger.warning(f"Received {sig.name}, initiating graceful shutdown...")
        
        # Set shutdown event to signal all components
        self.context.shutdown_event.set()
        
        try:
            # Use asyncio.timeout (Python 3.11+)
            async with asyncio.timeout(self.timeout):
                # Run registered cleanup actions in reverse order (LIFO)
                for action in reversed(self._shutdown_actions):
                    action_name = getattr(action, '__name__', str(action))
                    try:
                        logger.debug(f"Running cleanup: {action_name}")
                        await action()
                        logger.debug(f"Cleanup complete: {action_name}")
                    except Exception as e:
                        logger.error(f"Cleanup action {action_name} failed: {e}")
                
                # Cancel and await all remaining tasks
                current_task = asyncio.current_task()
                tasks = [t for t in asyncio.all_tasks() if t is not current_task and not t.done()]
                
                if tasks:
                    logger.info(f"Cancelling {len(tasks)} remaining tasks...")
                    for task in tasks:
                        task.cancel()
                    
                    # Wait for cancellations with exceptions captured
                    await asyncio.gather(*tasks, return_exceptions=True)
                
        except TimeoutError:
            logger.error(f"Shutdown timeout ({self.timeout}s) exceeded, forcing exit")
        
        logger.info("Shutdown complete")
    
    @property
    def exit_code(self) -> int:
        """Get appropriate exit code based on received signal."""
        if self._received_signal == signal.SIGINT:
            return EXIT_SIGINT
        elif self._received_signal == signal.SIGTERM:
            return EXIT_SIGTERM
        elif self._shutdown_started:
            return EXIT_SUCCESS
        return EXIT_SUCCESS


async def cleanup_with_timeout(
    cleanup_fn: Callable[[], Awaitable[None]],
    timeout: float = 5.0,
    name: str = "cleanup"
) -> bool:
    """
    Execute cleanup function with timeout.
    
    Args:
        cleanup_fn: Async cleanup function
        timeout: Maximum seconds to wait
        name: Name for logging
        
    Returns:
        True if cleanup succeeded, False if timed out or failed
    """
    try:
        async with asyncio.timeout(timeout):
            await cleanup_fn()
        return True
    except TimeoutError:
        logger.warning(f"{name} timed out after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"{name} failed: {e}")
        return False

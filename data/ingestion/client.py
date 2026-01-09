"""
Deriv API client wrapper for trading system.

Provides a clean interface to the Deriv.com API with error handling,
retries, and async streaming support for ticks and candles.

Example:
    >>> from data.ingestion.client import DerivClient
    >>> client = DerivClient(settings)
    >>> await client.connect()
    >>> balance = await client.get_balance()
"""

import asyncio
import logging
import random
import time
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any, Callable, cast

from deriv_api import APIError, DerivAPI

from config.settings import Settings

logger = logging.getLogger(__name__)


# I01: Circuit Breaker States
class CircuitState(Enum):
    """Circuit breaker states for connection resilience."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking all attempts (failures exceeded threshold)
    HALF_OPEN = "half_open"  # Allowing probe attempt


class CircuitBreaker:
    """
    I01 Fix: Circuit breaker pattern for graceful degradation.
    
    Prevents cascade failures by temporarily blocking operations
    after repeated failures, allowing time for recovery.
    
    States:
        CLOSED: Normal operation, requests flow through
        OPEN: Blocking requests, waiting for cooldown
        HALF_OPEN: Allowing one probe request to test recovery
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        initial_cooldown: float = 60.0,
        max_cooldown: float = 300.0,
        on_state_change: Callable[[CircuitState], None] | None = None,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Consecutive failures before opening circuit
            initial_cooldown: Initial cooldown period in seconds
            max_cooldown: Maximum cooldown period in seconds
            on_state_change: Optional callback when state changes
        """
        self.failure_threshold = failure_threshold
        self.initial_cooldown = initial_cooldown
        self.max_cooldown = max_cooldown
        self.on_state_change = on_state_change
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._current_cooldown = initial_cooldown
        
        # Probe lock for HALF_OPEN state - only one request allowed
        self._probing: bool = False
        # Timestamp for tracking duration in each state
        self._state_entered_at: float = time.monotonic()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state, auto-transitioning if cooldown elapsed."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._current_cooldown:
                self._set_state(CircuitState.HALF_OPEN)
        return self._state
    
    def _set_state(self, new_state: CircuitState) -> None:
        """Set state and notify callback with duration logging."""
        if new_state != self._state:
            old_state = self._state
            now = time.monotonic()
            duration = now - self._state_entered_at
            
            self._state = new_state
            self._state_entered_at = now
            
            # Reset probe lock when leaving HALF_OPEN
            if old_state == CircuitState.HALF_OPEN:
                self._probing = False
            
            logger.info(
                f"Circuit breaker: {old_state.value} → {new_state.value} "
                f"(was in {old_state.value} for {duration:.1f}s)"
            )
            if self.on_state_change:
                try:
                    self.on_state_change(new_state)
                except Exception as e:
                    logger.error(f"Circuit state change callback error: {e}")
    
    def record_success(self) -> None:
        """Record a successful operation, resetting the circuit."""
        self._failure_count = 0
        self._current_cooldown = self.initial_cooldown
        self._probing = False  # Reset probe lock
        self._set_state(CircuitState.CLOSED)
    
    def record_failure(self) -> None:
        """Record a failed operation."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        
        if self._failure_count >= self.failure_threshold:
            self._set_state(CircuitState.OPEN)
            # Exponential backoff for cooldown
            self._current_cooldown = min(
                self._current_cooldown * 2, 
                self.max_cooldown
            )
            logger.warning(
                f"Circuit breaker opened after {self._failure_count} failures. "
                f"Cooldown: {self._current_cooldown:.0f}s"
            )
    
    def should_allow_request(self) -> bool:
        """
        Check if a request should be allowed.
        
        In HALF_OPEN state, only ONE probe request is allowed at a time.
        Other concurrent requests must wait or fail until the probe completes.
        """
        current_state = self.state  # This may auto-transition
        if current_state == CircuitState.CLOSED:
            return True
        elif current_state == CircuitState.HALF_OPEN:
            # Only allow first probe request; others must wait
            if self._probing:
                return False
            self._probing = True
            return True
        else:  # OPEN
            return False
    
    async def wait_if_open(self) -> None:
        """
        Wait for cooldown if circuit is open.
        
        PERF-002 FIX: Optimized wait logic.
        """
        while self.state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            remaining = self._current_cooldown - elapsed
            
            if remaining <= 0:
                break
                
            # Log only if waiting for a significant time
            if remaining > 1.0:
                logger.info(f"Circuit open, waiting {remaining:.1f}s for cooldown...")
            
            # Sleep the exact amount needed (or max 5s for responsiveness/logging)
            # asyncio.sleep is non-blocking to the loop, so this is safe.
            await asyncio.sleep(min(remaining, 5.0))



class DerivClient:
    """
    Wrapper around python-deriv-api for unified trading system.
    Handles connection, authentication, streaming, and execution.
    
    Features:
    - **Robust Connection**: Automatic reconnection with exponential backoff.
    - **Streaming Recovery**: Detects stale streams and reconnects automatically.
    - **Error Handling**: Wraps API errors with context for upstream handling.
    - **Circuit Breaker** (I01): Graceful degradation when connectivity is lost.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api: DerivAPI | None = None
        self.symbol = settings.trading.symbol
        self.app_id = settings.deriv_app_id
        self.token = settings.deriv_api_token.get_secret_value()
        self._keep_alive_task: asyncio.Task | None = None
        self._reconnect_lock = asyncio.Lock()
        
        # Anti-cascade mechanism: epoch counter increments on each successful reconnection
        # Stream generators check this to avoid thundering herd on resubscription
        self._reconnect_epoch: int = 0
        self._subscriber_count: int = 0  # Track active stream subscribers
        
        # I01 Fix: Circuit breaker for graceful degradation
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            initial_cooldown=60.0,
            max_cooldown=300.0,
            on_state_change=self._on_circuit_state_change,
        )
    
    def _on_circuit_state_change(self, new_state: CircuitState) -> None:
        """Handle circuit breaker state changes."""
        if new_state == CircuitState.OPEN:
            logger.warning("🔴 Circuit breaker OPEN - connection degraded, entering cooldown")
        elif new_state == CircuitState.HALF_OPEN:
            logger.info("🟡 Circuit breaker HALF-OPEN - attempting recovery probe")
        elif new_state == CircuitState.CLOSED:
            logger.info("🟢 Circuit breaker CLOSED - connection recovered")

    @property
    def circuit_state(self) -> CircuitState:
        """Expose current circuit state."""
        return self.circuit_breaker.state

    @property
    def is_connected(self) -> bool:
        """Check if client is currently connected and authorized."""
        return self.api is not None and self._keep_alive_task is not None and not self._keep_alive_task.done()

    async def connect(self, max_retries: int = 3) -> None:
        """
        Initialize API connection and authenticate.

        Args:
            max_retries: Maximum connection retry attempts

        Raises:
            APIError: If connection/auth fails after retries
            ValueError: If API credentials are invalid
        """
        if not self.token:
            raise ValueError("API token not configured")

        logger.info(f"Connecting to Deriv (App ID: {self.app_id})...")

        for attempt in range(max_retries):
            try:
                self.api = DerivAPI(app_id=self.app_id)

                # Check connection with ping
                await self.api.ping({"ping": 1})

                # Authorize
                logger.info("Authorizing...")
                auth = await self.api.authorize(self.token)
                logger.info(
                    f"Authorized. Balance: {auth['authorize']['balance']} "
                    f"{auth['authorize']['currency']}"
                )

                # Start keep-alive loop
                self._keep_alive_task = asyncio.create_task(self._keep_alive_loop())
                
                # H10: Garbage collection on connect (Clean slate)
                try:
                     # Forget any lingering streams from previous sessions if they exist on the server context
                     # This is a best-effort cleanup
                     await self.api.forget_all(["proposal", "proposal_open_contract", "ticks", "candles"])
                     logger.debug("GC: Cleared lingering server streams on connect")
                except Exception as e:
                     # Non-critical, just log
                     logger.debug(f"GC: clean-slate forget failed (expected if clean): {e}")
                
                return

            except APIError as e:
                logger.error(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    # M14: Add jitter to prevent thundering herd
                    base_delay = 2**attempt
                    jitter = random.uniform(0.5 * base_delay, 1.5 * base_delay)
                    logger.info(f"Retrying in {jitter:.2f}s...")
                    await asyncio.sleep(jitter)
                else:
                    raise

    async def disconnect(self) -> None:
        """Clean disconnect from API."""
        if self._keep_alive_task:
            self._keep_alive_task.cancel()
            try:
                await self._keep_alive_task
            except asyncio.CancelledError:
                pass
            self._keep_alive_task = None

        if self.api:
            try:
                # H10: GC on disconnect
                logger.info("GC: Forgetting all streams before disconnect...")
                try:
                    await asyncio.wait_for(
                        self.api.forget_all(["proposal", "proposal_open_contract", "ticks", "candles"]),
                        timeout=5.0
                    )
                except Exception as e:
                     logger.warning(f"GC: Error during disconnect cleanup: {e}")
                
                await self.api.clear()
                logger.info("Disconnected from Deriv API")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self.api = None

    async def _keep_alive_loop(self, interval: int = 30) -> None:
        """
        Send periodic pings to keep the connection alive.
        """
        logger.info("Starting keep-alive loop")
        while True:
            try:
                await asyncio.sleep(interval)
                if self.api:
                    await self.api.ping({"ping": 1})
                    logger.debug("Keep-alive ping sent")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Keep-alive ping failed: {e}")
                # We don't break here to retry on next interval,
                # unless connection is truly dead which operations will catch

    async def _reconnect(self, max_attempts: int = 5) -> bool:
        """
        Attempt to reconnect after connection failure.

        Uses exponential backoff between attempts.

        Args:
            max_attempts: Maximum reconnection attempts

        Returns:
            True if reconnection successful, False otherwise
        """
        if self._reconnect_lock.locked():
             logger.info("Reconnection already in progress, waiting...")

        async with self._reconnect_lock:
            # Double-check connection state after acquiring lock
            if self.is_connected:
                logger.info("Client already reconnected by another task")
                return True

            for attempt in range(max_attempts):
                try:
                    logger.warning(f"Reconnection attempt {attempt + 1}/{max_attempts}")
                    await self.disconnect()  # Clean up old connection
                    
                    # M14: Jittered backoff
                    base_delay = 2**attempt
                    jitter = random.uniform(0.5 * base_delay, 1.5 * base_delay)
                    logger.info(f"Waiting {jitter:.2f}s before reconnecting...")
                    await asyncio.sleep(jitter)
                    
                    await self.connect()
                    # Increment epoch on successful reconnection to signal streams
                    self._reconnect_epoch += 1
                    logger.info(f"Reconnection successful (epoch={self._reconnect_epoch})")
                    return True
                except Exception as e:
                    logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")

            logger.error(f"Failed to reconnect after {max_attempts} attempts")
            return False

    async def get_balance(self) -> float:
        """
        Fetch current account balance.

        Returns:
            Current balance as float

        Raises:
            RuntimeError: If client not connected
        """
        if not self.api:
            raise RuntimeError("Client not connected")
        res = await self.api.balance()
        return float(res["balance"]["balance"])

    async def get_api_limits(self) -> dict[str, Any]:
        """
        Risk 4 Mitigation: Fetch current API rate limits from Deriv server.
        
        Uses the website_status call to retrieve the authoritative rate limits,
        enabling reconciliation between client-side tracking and server state.
        
        Returns:
            dict: API call limits containing:
                - max_proposal_subscription: Max concurrent proposal subscriptions
                - max_requests_general: Max general requests per window
                - max_requests_outcome: Max outcome requests per window  
                - max_requests_pricing: Max pricing requests per window
        
        Raises:
            RuntimeError: If client not connected
            
        Example:
            >>> limits = await client.get_api_limits()
            >>> print(f"Max general requests: {limits.get('max_requests_general')}")
        """
        if not self.api:
            raise RuntimeError("Client not connected")
        
        try:
            res = await self.api.website_status()
            limits = res.get("website_status", {}).get("api_call_limits", {})
            
            logger.debug(f"API limits from server: {limits}")
            return cast(dict[str, Any], limits)
        except Exception as e:
            logger.warning(f"Failed to fetch API limits: {e}")
            return {}

    async def get_historical_ticks(self, count: int = 1000) -> list[float]:
        """
        Fetch historical tick data to pre-fill buffers.
        Returns list of tick prices (most recent last).
        """
        if not self.api:
            raise RuntimeError("Client not connected")

        logger.info(f"Fetching {count} historical ticks for {self.symbol}...")
        res = await self.api.ticks_history(
            {"ticks_history": self.symbol, "count": count, "end": "latest", "style": "ticks", "adjust_start_time": 1}
        )

        ticks = [float(p) for p in res.get("history", {}).get("prices", [])]
        logger.info(f"Fetched {len(ticks)} historical ticks")
        return ticks

    async def get_historical_candles(
        self, count: int = 200, interval: int = 60
    ) -> list[dict[str, Any]]:
        """
        Fetch historical candle data to pre-fill buffers.
        Returns list of OHLCV candles (most recent last).
        """
        if not self.api:
            raise RuntimeError("Client not connected")

        requests = {
            "ticks_history": self.symbol,
            "count": count,
            "end": "latest",
            "style": "candles",
            "granularity": interval,
        }
        logger.debug(f"Requesting candles: {requests}")

        try:
            res = await asyncio.wait_for(self.api.ticks_history(requests), timeout=10.0)
            logger.debug("Candle response received.")
        except asyncio.TimeoutError:
            logger.error("Timeout fetching historical candles!")
            raise

        candles = res.get("candles", [])
        logger.info(f"Fetched {len(candles)} historical candles")
        return cast(list[dict[str, Any]], candles)

    async def get_historical_candles_by_range(
        self, start_time: int, end_time: int, interval: int = 60
    ) -> list[dict[str, Any]]:
        """
        Fetch historical candles for a specific time range.
        
        Args:
            start_time: Start epoch timestamp
            end_time: End epoch timestamp
            interval: Candle granularity in seconds
            
        Returns:
            List of OHLC candles
        """
        if not self.api:
            raise RuntimeError("Client not connected")

        requests = {
            "ticks_history": self.symbol,
            "style": "candles",
            "granularity": interval,
            "start": start_time,
            "end": end_time,
            "adjust_start_time": 1,
        }
        
        try:
            res = await self.api.ticks_history(requests)
            return cast(list[dict[str, Any]], res.get("candles", []))
        except Exception as e:
            logger.error(f"Failed to fetch historical candles by range: {e}")
            return []

    async def stream_ticks(self, stale_timeout: float = 30.0) -> AsyncGenerator[float, None]:
        """
        Stream ticks for the configured symbol with automatic reconnection.

        Features:
        - Automatic reconnection on stream errors or completion
        - Stale data detection if no ticks received for stale_timeout seconds
        - Exponential backoff on reconnection attempts
        - I01: Circuit breaker for graceful degradation

        Args:
            stale_timeout: Seconds without ticks before triggering reconnect

        Yields:
            Price float

        Raises:
            ConnectionError: If reconnection fails after max attempts
        """
        while True:  # Outer reconnection loop
            # I01 Fix: Wait if circuit is open
            await self.circuit_breaker.wait_if_open()
            
            if not self.api:
                raise RuntimeError("Client not connected")

            # Anti-cascade: capture epoch at start of this streaming attempt
            starting_epoch = self._reconnect_epoch
            
            logger.info(f"Subscribing to ticks for {self.symbol}...")

            try:
                source = await self.api.subscribe({"ticks": self.symbol})
                
                # I01: Record success on subscription
                self.circuit_breaker.record_success()

                # RxPY to async generator adapter
                # I02 Fix: Bounded queue to prevent memory exhaustion under backpressure
                queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
                stream_ended = False
                overflow_count = 0  # Track dropped items for observability

                def on_next(item):
                    nonlocal stream_ended, overflow_count
                    if stream_ended:
                        return
                    try:
                        quote = float(item["tick"]["quote"])
                        try:
                            queue.put_nowait(quote)
                        except asyncio.QueueFull:
                            # Drop oldest to make room for newest (LIFO-drop policy)
                            try:
                                queue.get_nowait()  # Discard oldest
                                queue.put_nowait(quote)
                            except asyncio.QueueEmpty:
                                pass  # Race condition, queue drained
                            overflow_count += 1
                            if overflow_count % 100 == 1:  # Log every 100 drops
                                logger.warning(
                                    f"Tick queue overflow: dropped {overflow_count} items. "
                                    f"Consumer may be lagging."
                                )
                    except Exception as e:
                        logger.error(f"Error parsing tick: {e}")

                def on_error(e):
                    nonlocal stream_ended
                    stream_ended = True
                    logger.error(f"Tick stream error: {e}")
                    queue.put_nowait(None)

                def on_completed():
                    nonlocal stream_ended
                    stream_ended = True
                    logger.info("Tick stream completed")
                    queue.put_nowait(None)

                subscription = source.subscribe(on_next=on_next, on_error=on_error, on_completed=on_completed)

                try:
                    while True:
                        try:
                            # Wait for tick with stale data timeout
                            item = await asyncio.wait_for(queue.get(), timeout=stale_timeout)
                            if item is None:
                                logger.warning("Tick stream ended, attempting reconnection...")
                                break  # Exit inner loop to trigger reconnection
                            yield item
                        except asyncio.TimeoutError:
                            logger.warning(
                                f"Stale data detected: no ticks for {stale_timeout}s, attempting reconnection..."
                            )
                            break  # Exit inner loop to trigger reconnection
                finally:
                    if subscription:
                        subscription.dispose()
                        logger.debug("Disposed tick stream subscription")

            except Exception as e:
                logger.error(f"Tick stream subscription error: {e}")
                # I01: Record failure
                self.circuit_breaker.record_failure()

            # Anti-cascade: check if another task already reconnected while we were streaming
            if self._reconnect_epoch > starting_epoch:
                # Another task already reconnected - add jittered delay to stagger resubscription
                jitter = random.uniform(0.5, 2.0)
                logger.info(f"Epoch changed ({starting_epoch} -> {self._reconnect_epoch}), "
                           f"waiting {jitter:.1f}s before resubscribing (anti-cascade)")
                await asyncio.sleep(jitter)
                continue  # Skip reconnect attempt, just resubscribe

            # Attempt reconnection
            if not await self._reconnect():
                # I01: Record failure on reconnection failure
                self.circuit_breaker.record_failure()
                if self.circuit_breaker.state == CircuitState.OPEN:
                    logger.error("Circuit breaker open - will retry after cooldown")
                    continue  # Continue loop, wait_if_open will block
                raise ConnectionError("Failed to reconnect tick stream after multiple attempts")

    async def stream_candles(
        self, interval: int = 60, stale_timeout: float = 120.0
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream OHLC candles with automatic reconnection.

        Features:
        - Automatic reconnection on stream errors or completion
        - Stale data detection if no candles received for stale_timeout seconds
        - Exponential backoff on reconnection attempts

        Note: Deriv API streams the ACTIVE candle repeatedly with OHLC updates.
        The caller must handle deduplication (update-in-place vs append).

        Args:
            interval: Candle interval in seconds (default 60 for 1m)
            stale_timeout: Seconds without candles before triggering reconnect

        Yields:
            OHLC dict with 'open', 'high', 'low', 'close', 'epoch' keys

        Raises:
            ConnectionError: If reconnection fails after max attempts
        """
        while True:  # Outer reconnection loop
            # I01 Fix: Wait if circuit is open
            await self.circuit_breaker.wait_if_open()
            
            if not self.api:
                raise RuntimeError("Client not connected")

            # Anti-cascade: capture epoch at start of this streaming attempt
            starting_epoch = self._reconnect_epoch

            logger.info(f"Subscribing to candle stream ({interval}s) for {self.symbol}...")

            try:
                source = await self.api.subscribe(
                    {
                        "ticks_history": self.symbol,
                        "adjust_start_time": 1,
                        "count": 1,
                        "end": "latest",
                        "style": "candles",
                        "granularity": interval,
                    }
                )
                
                # I01: Record success on subscription
                self.circuit_breaker.record_success()

                # I02 Fix: Bounded queue to prevent memory exhaustion
                queue: asyncio.Queue = asyncio.Queue(maxsize=200)
                stream_ended = False
                overflow_count = 0

                def on_next(item):
                    nonlocal stream_ended, overflow_count
                    if stream_ended:
                        return
                    # item structure: {'ohlc': {'open': ..., 'high': ..., ...}}
                    if "ohlc" in item:
                        try:
                            queue.put_nowait(item["ohlc"])
                        except asyncio.QueueFull:
                            # Drop oldest candle to make room
                            try:
                                queue.get_nowait()
                                queue.put_nowait(item["ohlc"])
                            except asyncio.QueueEmpty:
                                pass
                            overflow_count += 1
                            if overflow_count % 10 == 1:
                                logger.warning(
                                    f"Candle queue overflow: dropped {overflow_count} items."
                                )
                    elif "candles" in item:
                        # Initial history response - skip
                        pass

                def on_error(e):
                    nonlocal stream_ended
                    stream_ended = True
                    logger.error(f"Candle stream error: {e}")
                    queue.put_nowait(None)

                def on_completed():
                    nonlocal stream_ended
                    stream_ended = True
                    logger.info("Candle stream completed")
                    queue.put_nowait(None)

                subscription = source.subscribe(on_next=on_next, on_error=on_error, on_completed=on_completed)

                try:
                    while True:
                        try:
                            # Wait for candle with stale data timeout
                            item = await asyncio.wait_for(queue.get(), timeout=stale_timeout)
                            if item is None:
                                logger.warning("Candle stream ended, attempting reconnection...")
                                break  # Exit inner loop to trigger reconnection
                            yield item
                        except asyncio.TimeoutError:
                            logger.warning(
                                f"Stale data detected: no candles for {stale_timeout}s, attempting reconnection..."
                            )
                            break  # Exit inner loop to trigger reconnection
                finally:
                    if subscription:
                        subscription.dispose()
                        logger.debug("Disposed candle stream subscription")

            except Exception as e:
                logger.error(f"Candle stream subscription error: {e}")
                # I01: Record failure
                self.circuit_breaker.record_failure()

            # Anti-cascade: check if another task already reconnected while we were streaming
            if self._reconnect_epoch > starting_epoch:
                # Another task already reconnected - add jittered delay to stagger resubscription
                jitter = random.uniform(0.5, 2.0)
                logger.info(f"Epoch changed ({starting_epoch} -> {self._reconnect_epoch}), "
                           f"waiting {jitter:.1f}s before resubscribing (anti-cascade)")
                await asyncio.sleep(jitter)
                continue  # Skip reconnect attempt, just resubscribe

            # Attempt reconnection
            if not await self._reconnect():
                # I01: Record failure on reconnection failure  
                self.circuit_breaker.record_failure()
                if self.circuit_breaker.state == CircuitState.OPEN:
                    logger.error("Circuit breaker open - will retry after cooldown")
                    continue  # Continue loop, wait_if_open will block
                raise ConnectionError("Failed to reconnect candle stream after multiple attempts")

    async def buy(
        self, 
        contract_type: str, 
        amount: float, 
        duration: int, 
        duration_unit: str = "m",
        barrier: str | None = None,
        barrier2: str | None = None
    ) -> dict[str, Any]:
        """
        Execute a binary options trade on Deriv.
        
        Workflow:
        1. Request proposal for the specified contract
        2. Buy the proposal to execute the trade
        
        Args:
            contract_type: Contract type (CALL, PUT, ONETOUCH, NOTOUCH, etc.)
            amount: Stake amount in account currency (USD)
            duration: Contract duration
            duration_unit: Duration unit - 't' (ticks), 's' (seconds), 
                          'm' (minutes), 'h' (hours), 'd' (days)
        
        Returns:
            dict: Buy response containing:
                - contract_id: Unique identifier for the contract
                - buy_price: Actual buy price
                - balance_after: Account balance after purchase
                - Other contract details
        
        Raises:
            RuntimeError: If client not connected
            APIError: If proposal or buy request fails
        
        Example:
            >>> result = await client.buy("CALL", 1.0, 1, "m")
            >>> print(f"Contract ID: {result['contract_id']}")
        """
        if not self.api:
            raise RuntimeError("Client not connected")
        
        # OPT1: Check circuit breaker
        if not self.circuit_breaker.should_allow_request():
            raise RuntimeError("Circuit breaker open - trading suspended")
        
        c_type = contract_type.upper()
        
        logger.info(f"Requesting proposal: {c_type} {self.symbol} ${amount}...")

        proposal_req = {
            "proposal": 1,
            "amount": amount,
            "basis": "stake",
            "contract_type": c_type,
            "currency": "USD",
            "duration": duration,
            "duration_unit": duration_unit,
            "symbol": self.symbol,
        }
        
        # C05: Add barrier support
        if barrier:
            proposal_req["barrier"] = barrier

        prop_id = None
        try:
            prop = await self.api.proposal(proposal_req)
            prop_id = prop["proposal"]["id"]

            logger.info(f"Buying proposal {prop_id} with max price: {amount}")
            # M02: Strict slippage protection. For basis='stake', price should typically equal amount.
            # We set limit exactly to amount to reject any unexpected premium/fees.
            buy = await self.api.buy({"buy": prop_id, "price": amount})

            # OPT1: Record success
            self.circuit_breaker.record_success()

            return cast(dict[str, Any], buy["buy"])
        except Exception as e:
            # OPT1: Record failure for API errors
            logger.error(f"Buy failed: {e}")
            self.circuit_breaker.record_failure()
            raise e
        finally:
            # H10: Garbage collection - forget the proposal stream to prevent leaks
            if prop_id:
                try:
                    # Use forget(), not forget_all(), for specific ID
                    await self.api.forget(prop_id)
                    logger.debug(f"GC: Forgot proposal {prop_id}")
                except Exception as e:
                    logger.warning(f"GC: Failed to forget proposal {prop_id}: {e}")

    async def get_open_contracts(self) -> list[dict[str, Any]]:
        """
        Fetch all currently active (open) contracts.
        
        Returns all contracts that have not yet settled. Useful for
        checking portfolio state and managing open positions.
        
        Returns:
            list: List of open contract details, or the raw API response
                  if parsing fails.
        
        Raises:
            RuntimeError: If client not connected
        
        Note:
            For real-time updates on a specific contract, use 
            subscribe_contract() instead.
        """
        if not self.api:
            raise RuntimeError("Client not connected")
        
        # OPT1: Check circuit breaker
        if not self.circuit_breaker.should_allow_request():
            raise RuntimeError("Circuit breaker open")
        
        try:
            res = await self.api.proposal_open_contract({
                "proposal_open_contract": 1, 
                "scope": "open"
            })
            self.circuit_breaker.record_success()
            return cast(list[dict[str, Any]], res)
        except Exception as e:
            logger.error(f"Failed to fetch open contracts: {e}")
            self.circuit_breaker.record_failure()
            raise e

    async def subscribe_contract(
        self, contract_id: str, on_update: Callable[[dict[str, Any]], None] | None = None, on_settled: Callable[[float, bool], None] | None = None
    ) -> bool:
        """
        Subscribe to real-time updates for a specific contract.
        
        Uses proposal_open_contract with subscribe=1 to receive updates
        until the contract is sold or expires. This enables tracking of
        trade outcomes for compounding and P&L management.
        
        The method blocks until the contract settles or times out.
        
        Args:
            contract_id: The Deriv contract ID to watch (as string)
            on_update: Optional callback(contract_data: dict) for each update.
                       Called with the full proposal_open_contract response.
            on_settled: Optional callback(profit: float, won: bool) when contract
                       settles. Called exactly once when is_expired or is_sold.
        
        Note:
            This uses the same RxPY Observable pattern as stream_ticks/stream_candles.
            The subscription is managed internally and cleaned up on settlement.
        
        Example:
            >>> async def on_win(profit, won):
            ...     print(f"Trade {'won' if won else 'lost'}: ${profit}")
            >>> await client.subscribe_contract("123456", on_settled=on_win)
        """
        if not self.api:
            raise RuntimeError("Client not connected")
        
        logger.info(f"[CONTRACT] Subscribing to contract {contract_id}")
        
        # Event to signal settlement completion
        settled = asyncio.Event()
        settlement_data = {"profit": 0.0, "won": False}
        
        def handle_update(msg: dict) -> None:
            """Process incoming contract update."""
            # Handle proposal_open_contract response
            if "proposal_open_contract" not in msg:
                return
            
            contract = msg["proposal_open_contract"]
            
            # Call update callback if provided
            if on_update:
                try:
                    on_update(contract)
                except Exception as e:
                    logger.error(f"[CONTRACT] Update callback error: {e}")
            
            # Check if contract has settled
            is_expired = contract.get("is_expired", False)
            is_sold = contract.get("is_sold", False)
            status = contract.get("status", "")
            
            if is_expired or is_sold or status in ("sold", "won", "lost"):
                profit = float(contract.get("profit", 0))
                won = profit > 0
                
                settlement_data["profit"] = profit
                settlement_data["won"] = won
                
                logger.info(
                    f"[CONTRACT] {contract_id} settled: "
                    f"{'WON' if won else 'LOST'}, profit=${profit:.2f}"
                )
                
                # Call settlement callback
                if on_settled:
                    try:
                        on_settled(profit, won)
                    except Exception as e:
                        logger.error(f"[CONTRACT] Settlement callback error: {e}")
                
                settled.set()
        
        def handle_error(e: Exception) -> None:
            """Handle subscription error."""
            logger.error(f"[CONTRACT] Subscription error for {contract_id}: {e}")
            settled.set()
        
        def handle_complete() -> None:
            """Handle subscription completion."""
            logger.debug(f"[CONTRACT] Subscription completed for {contract_id}")
            if not settled.is_set():
                settled.set()
        
        try:
            # OPT1: Check circuit breaker
            if not self.circuit_breaker.should_allow_request():
                logger.warning(f"[CONTRACT] Circuit breaker open, skipping subscription for {contract_id}")
                return False

            # Use api.subscribe() to get an RxPY Observable
            # Same pattern as stream_ticks and stream_candles
            source = await self.api.subscribe({
                "proposal_open_contract": 1,
                "contract_id": int(contract_id),
                "subscribe": 1,
            })
            
            # Record success on successful subscription
            self.circuit_breaker.record_success()
            
            # Subscribe to the Observable with callbacks
            # CRITICAL: Capture disposable to prevent memory leak in long-running app
            disposable = source.subscribe(
                on_next=handle_update,
                on_error=handle_error,
                on_completed=handle_complete,
            )
            
        except Exception as e:
            logger.error(f"[CONTRACT] Failed to subscribe to {contract_id}: {e}")
            self.circuit_breaker.record_failure()
            return True # Return True to avoid infinite retry loops in caller if checking strict boolean? 
            # Original code returned True on error (line 874). 
            # If we return False, caller might retry immediately. 
            # If we return True, caller thinks it's handled. 
            # Safety wrapper handles retries. Let's return False to indicate failure?
            # Looking at original code: "return True" at line 874.
            # "The method blocks until the contract settles or times out."
            # If subscription fails, we can't wait for settlement.
            # If we returned True, caller proceeds.
            return False
        
        # Wait for settlement with timeout (2 minutes for 1-min contracts + buffer)
        try:
            await asyncio.wait_for(settled.wait(), timeout=180)
            return True  # Settled successfully
        except asyncio.TimeoutError:
            logger.warning(f"[CONTRACT] Subscription timeout for {contract_id}")
            return False # Timed out
        finally:
            # CRITICAL: Dispose subscription to prevent memory leak
            # Without this, callbacks accumulate in memory (OOM after days/weeks)
            if disposable:
                try:
                    disposable.dispose()
                    logger.debug(f"[CONTRACT] Disposed subscription for {contract_id}")
                except Exception as e:
                    logger.warning(f"[CONTRACT] Warning during disposal: {e}")

    async def get_history(self, symbol: str, style: str, count: int, interval: int | str = 60) -> list[dict[str, Any]]:
        """
        Fetch historical data (ticks or candles).
        
        Args:
            symbol: Symbol to fetch for
            style: 'ticks' or 'candles'
            count: Number of data points
            interval: Granularity for candles (seconds or '1m' etc)
            
        Returns:
            List of history items (dicts)
        """
        await self.circuit_breaker.wait_if_open()
        if not self.api:
            raise RuntimeError("Client not connected")
            
        if isinstance(interval, str):
            if interval == "1m": interval = 60
            elif interval == "1h": interval = 3600
            
        req = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "style": style,
        }
        if style == "candles":
            req["granularity"] = interval
            
        try:
            resp = await self.api.ticks_history(req)
            
            if style == "ticks":
                if "history" in resp:
                    history = resp["history"]
                    times = history.get("times", [])
                    prices = history.get("prices", [])
                    return [{'quote': p, 'epoch': t} for p, t in zip(prices, times)]
            elif style == "candles":
                if "candles" in resp:
                    from typing import cast
                    return cast(list[dict[str, Any]], resp["candles"])
                    
            return []
            
        except Exception as e:
            logger.error(f"Failed to fetch history: {e}")
            self.circuit_breaker.record_failure()
            return []
            
        return []
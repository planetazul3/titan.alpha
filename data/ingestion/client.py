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
from collections.abc import AsyncGenerator
from collections.abc import AsyncGenerator
from typing import Any, Callable, cast

from deriv_api import APIError, DerivAPI

from config.settings import Settings

logger = logging.getLogger(__name__)


class DerivClient:
    """
    Wrapper around python-deriv-api for unified trading system.
    Handles connection, authentication, streaming, and execution.
    
    Features:
    - **Robust Connection**: Automatic reconnection with exponential backoff.
    - **Streaming Recovery**: Detects stale streams and reconnects automatically.
    - **Error Handling**: Wraps API errors with context for upstream handling.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api: DerivAPI | None = None
        self.symbol = settings.trading.symbol
        self.app_id = settings.deriv_app_id
        self.token = settings.deriv_api_token
        self._keep_alive_task: asyncio.Task | None = None
        self._reconnect_lock = asyncio.Lock()

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
                    logger.info("Reconnection successful")
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

    async def get_historical_ticks(self, count: int = 1000) -> list[float]:
        """
        Fetch historical tick data to pre-fill buffers.
        Returns list of tick prices (most recent last).
        """
        if not self.api:
            raise RuntimeError("Client not connected")

        logger.info(f"Fetching {count} historical ticks for {self.symbol}...")
        res = await self.api.ticks_history(
            {"ticks_history": self.symbol, "count": count, "end": "latest", "style": "ticks"}
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

        Args:
            stale_timeout: Seconds without ticks before triggering reconnect

        Yields:
            Price float

        Raises:
            ConnectionError: If reconnection fails after max attempts
        """
        while True:  # Outer reconnection loop
            if not self.api:
                raise RuntimeError("Client not connected")

            logger.info(f"Subscribing to ticks for {self.symbol}...")

            try:
                source = await self.api.subscribe({"ticks": self.symbol})

                # RxPY to async generator adapter
                queue: asyncio.Queue = asyncio.Queue()
                stream_ended = False

                def on_next(item):
                    nonlocal stream_ended
                    if stream_ended:
                        return
                    try:
                        quote = float(item["tick"]["quote"])
                        queue.put_nowait(quote)
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

            # Attempt reconnection
            if not await self._reconnect():
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
            if not self.api:
                raise RuntimeError("Client not connected")

            logger.info(f"Subscribing to candle stream ({interval}s) for {self.symbol}...")

            try:
                source = await self.api.subscribe(
                    {
                        "ticks_history": self.symbol,
                        "adjust_start_time": 1,
                        "count": 1,
                        "end": "latest",
                        "start": 1,
                        "style": "candles",
                        "granularity": interval,
                    }
                )

                queue: asyncio.Queue = asyncio.Queue()
                stream_ended = False

                def on_next(item):
                    nonlocal stream_ended
                    if stream_ended:
                        return
                    # item structure: {'ohlc': {'open': ..., 'high': ..., ...}}
                    if "ohlc" in item:
                        queue.put_nowait(item["ohlc"])
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

            # Attempt reconnection
            if not await self._reconnect():
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

        prop = await self.api.proposal(proposal_req)
        prop_id = prop["proposal"]["id"]

        logger.info(f"Buying proposal {prop_id} with max price: {amount}")
        # M02: Strict slippage protection. For basis='stake', price should typically equal amount.
        # We set limit exactly to amount to reject any unexpected premium/fees.
        buy = await self.api.buy({"buy": prop_id, "price": amount})

        return cast(dict[str, Any], buy["buy"])

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
        
        res = await self.api.proposal_open_contract({
            "proposal_open_contract": 1, 
            "scope": "open"
        })
        return cast(list[dict[str, Any]], res)

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
            # Use api.subscribe() to get an RxPY Observable
            # Same pattern as stream_ticks and stream_candles
            source = await self.api.subscribe({
                "proposal_open_contract": 1,
                "contract_id": int(contract_id),
                "subscribe": 1,
            })
            
            # Subscribe to the Observable with callbacks
            # CRITICAL: Capture disposable to prevent memory leak in long-running app
            disposable = source.subscribe(
                on_next=handle_update,
                on_error=handle_error,
                on_completed=handle_complete,
            )
            
        except Exception as e:
            logger.error(f"[CONTRACT] Failed to subscribe to {contract_id}: {e}")
            return
        
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
                    logger.warning(f"[CONTRACT] Error disposing subscription: {e}")
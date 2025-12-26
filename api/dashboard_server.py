"""
Dashboard API Server - READ-ONLY access to trading data.

CRITICAL SAFETY RULES:
1. This server ONLY READS from existing SQLite databases
2. It does NOT modify any trading system state
3. It does NOT import from existing trading modules
4. It runs independently of the trading system

Usage:
    uvicorn api.dashboard_server:app --reload --port 8000
"""

import asyncio
import logging
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.models.responses import (
    HealthResponse,
    ShadowTrade,
    ShadowTradeList,
    ShadowTradeStats,
    TradingMetrics,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database paths (read-only access)
PROJECT_ROOT = Path(__file__).parent.parent
SHADOW_DB_PATH = PROJECT_ROOT / "data_cache" / "shadow_trades.db"
SAFETY_DB_PATH = PROJECT_ROOT / "data_cache" / "safety_state.db"

# Binary options payout rate (Deriv standard: ~95%)
PAYOUT_RATE = 0.95


class ConnectionManager:
    """Manages WebSocket connections for real-time streaming."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        """Accept and track a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total clients: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        """Remove a disconnected WebSocket."""
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = []
        async with self._lock:
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            await self.disconnect(conn)


# Global connection manager
manager = ConnectionManager()


def readonly_connection(db_path: Path) -> sqlite3.Connection:
    """
    Create a read-only SQLite connection.

    Uses the file: URI with mode=ro to ensure read-only access.
    This prevents any accidental writes to the trading system databases.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_shadow_trade_stats() -> ShadowTradeStats:
    """
    Get aggregate statistics from shadow trades database.

    Returns default stats if database is unavailable.
    """
    try:
        with readonly_connection(SHADOW_DB_PATH) as conn:
            cursor = conn.cursor()

            # Total trades
            total = cursor.execute("SELECT COUNT(*) FROM shadow_trades").fetchone()[0]

            # Resolved trades (have outcome)
            resolved = cursor.execute(
                "SELECT COUNT(*) FROM shadow_trades WHERE outcome IS NOT NULL"
            ).fetchone()[0]

            # Wins
            wins = cursor.execute(
                "SELECT COUNT(*) FROM shadow_trades WHERE outcome = 1"
            ).fetchone()[0]

            losses = resolved - wins
            win_rate = wins / resolved if resolved > 0 else 0.0

            # Simulated P&L: wins pay 95%, losses lose 100%
            simulated_pnl = (wins * PAYOUT_RATE) - losses
            roi = (simulated_pnl / total * 100) if total > 0 else 0.0

            return ShadowTradeStats(
                total=total,
                resolved=resolved,
                unresolved=total - resolved,
                wins=wins,
                losses=losses,
                win_rate=round(win_rate, 4),
                simulated_pnl=round(simulated_pnl, 2),
                roi=round(roi, 2),
            )
    except FileNotFoundError:
        logger.warning(f"Shadow trades database not found: {SHADOW_DB_PATH}")
        return ShadowTradeStats(
            total=0, resolved=0, unresolved=0, wins=0, losses=0,
            win_rate=0.0, simulated_pnl=0.0, roi=0.0
        )
    except Exception as e:
        logger.error(f"Error reading shadow trades: {e}")
        return ShadowTradeStats(
            total=0, resolved=0, unresolved=0, wins=0, losses=0,
            win_rate=0.0, simulated_pnl=0.0, roi=0.0
        )


def get_safety_state() -> Dict[str, float]:
    """
    Get safety state metrics from safety database.

    Returns empty dict if database is unavailable.
    """
    try:
        with readonly_connection(SAFETY_DB_PATH) as conn:
            cursor = conn.cursor()
            metrics = {}
            for row in cursor.execute("SELECT key, value FROM safety_metrics"):
                metrics[row["key"]] = row["value"]
            return metrics
    except FileNotFoundError:
        logger.warning(f"Safety database not found: {SAFETY_DB_PATH}")
        return {}
    except Exception as e:
        logger.error(f"Error reading safety state: {e}")
        return {}


def get_current_metrics() -> Dict[str, Any]:
    """
    Get complete current metrics snapshot.

    This is the main data source for WebSocket streaming.
    """
    shadow_stats = get_shadow_trade_stats()
    safety_state = get_safety_state()

    # Determine system status
    system_status = "connected"
    if shadow_stats.total == 0:
        system_status = "no_data"

    return {
        "shadow_trades": shadow_stats.model_dump(),
        "safety_state": safety_state,
        "system_status": system_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def metrics_broadcaster():
    """
    Background task that polls databases and broadcasts to WebSocket clients.

    Runs every 2 seconds to provide near-real-time updates.
    """
    logger.info("Starting metrics broadcaster...")
    while True:
        try:
            if manager.active_connections:
                metrics = get_current_metrics()
                await manager.broadcast({
                    "type": "metrics",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": metrics,
                })
        except Exception as e:
            logger.error(f"Broadcast error: {e}")

        await asyncio.sleep(2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - start/stop background tasks."""
    logger.info("Starting Dashboard API server...")
    logger.info(f"Shadow DB path: {SHADOW_DB_PATH}")
    logger.info(f"Safety DB path: {SAFETY_DB_PATH}")

    # Start background broadcaster
    task = asyncio.create_task(metrics_broadcaster())
    yield

    # Cleanup
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    logger.info("Dashboard API server stopped.")


# Create FastAPI application
app = FastAPI(
    title="DerivOmniModel Dashboard API",
    description="Read-only API for real-time trading dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative dev server
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REST Endpoints
# =============================================================================


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/api/metrics/current", response_model=TradingMetrics)
async def get_metrics():
    """Get current trading metrics snapshot."""
    shadow_stats = get_shadow_trade_stats()
    safety_state = get_safety_state()

    return TradingMetrics(
        shadow_trades=shadow_stats,
        safety_state=safety_state,
        system_status="connected" if shadow_stats.total > 0 else "no_data",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/api/shadow-trades", response_model=ShadowTradeList)
async def get_shadow_trades(limit: int = 100, offset: int = 0):
    """
    Get recent shadow trades.

    Args:
        limit: Maximum number of trades to return (default: 100, max: 500)
        offset: Number of trades to skip for pagination
    """
    limit = min(limit, 500)  # Cap at 500 to prevent memory issues

    try:
        with readonly_connection(SHADOW_DB_PATH) as conn:
            cursor = conn.cursor()
            rows = cursor.execute(
                """
                SELECT trade_id, timestamp, contract_type, direction,
                       probability, entry_price, exit_price, outcome,
                       reconstruction_error, regime_state
                FROM shadow_trades
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

            trades = [
                ShadowTrade(
                    trade_id=row["trade_id"],
                    timestamp=row["timestamp"],
                    contract_type=row["contract_type"],
                    direction=row["direction"],
                    probability=row["probability"],
                    entry_price=row["entry_price"],
                    exit_price=row["exit_price"],
                    outcome=row["outcome"],
                    reconstruction_error=row["reconstruction_error"],
                    regime_state=row["regime_state"],
                )
                for row in rows
            ]

            return ShadowTradeList(trades=trades, count=len(trades))

    except FileNotFoundError:
        return ShadowTradeList(trades=[], count=0)
    except Exception as e:
        logger.error(f"Error fetching shadow trades: {e}")
        return ShadowTradeList(trades=[], count=0)


@app.get("/api/shadow-trades/stats")
async def get_shadow_stats_by_contract():
    """Get shadow trade statistics grouped by contract type."""
    try:
        with readonly_connection(SHADOW_DB_PATH) as conn:
            cursor = conn.cursor()

            by_contract = {}
            for row in cursor.execute(
                """
                SELECT contract_type,
                       COUNT(*) as total,
                       SUM(CASE WHEN outcome = 1 THEN 1 ELSE 0 END) as wins
                FROM shadow_trades
                WHERE outcome IS NOT NULL
                GROUP BY contract_type
                """
            ):
                total = row["total"]
                wins = row["wins"] or 0
                by_contract[row["contract_type"]] = {
                    "total": total,
                    "wins": wins,
                    "losses": total - wins,
                    "win_rate": round(wins / total, 4) if total > 0 else 0,
                }

            return {"by_contract_type": by_contract}

    except FileNotFoundError:
        return {"by_contract_type": {}}
    except Exception as e:
        logger.error(f"Error fetching contract stats: {e}")
        return {"by_contract_type": {}, "error": str(e)}


@app.get("/api/safety/state")
async def get_safety():
    """Get current safety state."""
    return {"safety_state": get_safety_state()}


# =============================================================================
# WebSocket Endpoint
# =============================================================================


@app.websocket("/ws/trading-stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time trading data streaming.

    Clients connect here to receive automatic updates every 2 seconds.
    """
    await manager.connect(websocket)

    try:
        # Send initial data immediately
        metrics = get_current_metrics()
        await websocket.send_json({
            "type": "metrics",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": metrics,
        })

        # Keep connection alive
        while True:
            # Wait for any message (used as keepalive ping)
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send heartbeat if no message received
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {},
                })

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(websocket)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.dashboard_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

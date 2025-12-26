# DerivOmniModel Trading Dashboard

Real-time dashboard for monitoring shadow trading performance, execution statistics, and system health.

## Quick Start

### 1. Start the API Server

```bash
cd x.titan
source venv/bin/activate
uvicorn api.dashboard_server:app --reload --port 8000
```

### 2. Start the Frontend (requires Node.js)

```bash
cd dashboard
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

## Features

- **Real-time WebSocket streaming** - Updates every 2 seconds
- **Shadow trade performance** - Win rate, P&L, ROI
- **Execution statistics** - Attempted, executed, blocked trades
- **Trade history table** - Filterable by outcome
- **Connection status indicator**

## Architecture

- **Backend**: FastAPI (Python) - Read-only access to SQLite databases
- **Frontend**: React + TypeScript + Tailwind CSS
- **Communication**: WebSocket for real-time updates, REST for historical data

## Safety

This dashboard is **read-only** and does not modify any trading system data.
It can run independently while `scripts/live.py` is active.

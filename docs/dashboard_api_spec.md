# x.titan Dashboard API Specification

**Version**: 1.0  
**Purpose**: Enable frontend dashboard development without full codebase access

---

## Python Class Definitions (Copy These)

### 1. config/constants.py (Enums)

```python
from enum import Enum

class CONTRACT_TYPES(str, Enum):
    """Supported contract types for Deriv binary options."""
    RISE_FALL = "RISE_FALL"
    TOUCH_NO_TOUCH = "TOUCH_NO_TOUCH"
    STAYS_BETWEEN = "STAYS_BETWEEN"

class SIGNAL_TYPES(str, Enum):
    """Classification of trade signals based on confidence."""
    REAL_TRADE = "REAL_TRADE"      # High confidence, execute
    SHADOW_TRADE = "SHADOW_TRADE"  # Medium confidence, track only
    IGNORE = "IGNORE"              # Low confidence, discard
```

---

### 2. execution/signals.py (TradeSignal)

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass(frozen=True)
class TradeSignal:
    """Base signal structure for trade opportunities."""
    signal_type: str          # REAL_TRADE | SHADOW_TRADE | IGNORE
    contract_type: str        # RISE_FALL | TOUCH_NO_TOUCH | STAYS_BETWEEN
    direction: str | None     # CALL | PUT | TOUCH | NO_TOUCH
    probability: float        # 0.0 - 1.0
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    signal_id: str = field(default="")  # Auto-generated SHA256[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type,
            "contract_type": self.contract_type,
            "direction": self.direction,
            "probability": self.probability,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
```

---

### 3. execution/shadow_store.py (ShadowTradeRecord)

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

SHADOW_STORE_SCHEMA_VERSION = "2.0"

@dataclass
class ShadowTradeRecord:
    """Immutable shadow trade record with rich metadata."""
    
    # Core trade info
    trade_id: str                    # UUID
    timestamp: datetime              # When signal was generated
    contract_type: str               # RISE_FALL | TOUCH_NO_TOUCH | STAYS_BETWEEN
    direction: str                   # CALL | PUT | TOUCH | NO_TOUCH
    probability: float               # Model's predicted probability (0-1)
    entry_price: float               # Price at signal generation

    # Risk/Regime metadata
    reconstruction_error: float      # Volatility expert error (0-1)
    regime_state: str                # TRUSTED | CAUTION | VETO

    # Versioning
    model_version: str = "unknown"
    feature_schema_version: str = "1.0"

    # Outcome (filled by resolution system)
    outcome: bool | None = None      # True=win, False=loss, None=pending
    exit_price: float | None = None
    resolved_at: datetime | None = None

    # Barrier levels (for TOUCH/RANGE contracts)
    barrier_level: float | None = None
    barrier2_level: float | None = None
    duration_minutes: int = 1

    # Extra metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp.isoformat(),
            "contract_type": self.contract_type,
            "direction": self.direction,
            "probability": self.probability,
            "entry_price": self.entry_price,
            "reconstruction_error": self.reconstruction_error,
            "regime_state": self.regime_state,
            "model_version": self.model_version,
            "outcome": self.outcome,
            "exit_price": self.exit_price,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "barrier_level": self.barrier_level,
            "duration_minutes": self.duration_minutes,
            "metadata": self.metadata
        }

    def is_resolved(self) -> bool:
        return self.outcome is not None
```

---

## JSON Schemas (For API Validation)

### ShadowTradeRecord
```json
{
  "trade_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-01-02T10:30:00Z",
  "contract_type": "RISE_FALL",
  "direction": "CALL",
  "probability": 0.82,
  "entry_price": 100.50,
  "reconstruction_error": 0.05,
  "regime_state": "TRUSTED",
  "outcome": true,
  "exit_price": 101.25,
  "resolved_at": "2026-01-02T10:31:00Z",
  "barrier_level": null,
  "duration_minutes": 1
}
```

### SystemStatistics
```json
{
  "total_trades": 1250,
  "resolved_trades": 1200,
  "wins": 720,
  "losses": 480,
  "win_rate": 0.60,
  "today_trades": 45,
  "today_win_rate": 0.58
}
```

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/trades` | List trades (paginated) |
| GET | `/api/trades/{id}` | Get single trade |
| GET | `/api/trades/stats` | Aggregate statistics |
| GET | `/api/trades/live` | Active unresolved trades |
| GET | `/api/system/regime` | Current regime state |

**Query Parameters** for `/api/trades`:
- `start`: ISO timestamp
- `end`: ISO timestamp  
- `resolved`: `true|false`
- `limit`: integer (max 500)

---

## WebSocket Channels

| Channel | Payload |
|---------|---------|
| `ws://host/ws/trades` | `ShadowTradeRecord.to_dict()` on new/resolved |
| `ws://host/ws/regime` | `{"state": "TRUSTED", "updated_at": "..."}` |

---

## Dashboard Views

1. **Overview**: Win rate gauge, P&L chart, regime indicator
2. **Trade History**: Filterable table, CSV export
3. **Analytics**: Win rate by contract type, confidence scatter
4. **Live Monitor**: Real-time trade feed, price chart

---

## Integration Options

| Option | Use Case | Effort |
|--------|----------|--------|
| **Flask/FastAPI** | Self-hosted REST API | Low |
| **Firebase Functions** | Serverless, real-time | Medium |
| **BigQuery + Looker** | Analytics dashboard | Medium |

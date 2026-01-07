# ADR-009: Live Script Modularization

**Date**: 2026-01-07  
**Status**: Accepted  
**Deciders**: Development Team  

## Context

`scripts/live.py` exceeded 974 lines with 4 nested functions using closures for state. This violated micro-modularity principles and obscured critical safety logic.

## Decision

Extract nested functions into standalone modules using `LiveTradingContext` for dependency injection:

| Module | Responsibility |
|--------|----------------|
| `context.py` | Dependency injection container |
| `event_handlers.py` | `tick_processor`, `candle_processor` |
| `heartbeat.py` | Hot-reload with M12/C06 safety |
| `maintenance.py` | Log cleanup, DB pruning |
| `cli.py` | Argument parsing |
| `shutdown.py` | Graceful shutdown handling |

## Consequences

**Positive**:
- `live.py` reduced to 599 lines (-38%)
- Safety logic (H6, M12, C01, C06) now explicitly visible
- Modules independently testable via mocked context
- Structured concurrency via `asyncio.TaskGroup`

**Negative**:
- 6 new files to maintain
- Context object adds indirection layer

## References

- [live_script_refactoring.md](file:///home/planetazul3/x.titan/docs/plans/live_script_refactoring.md)
- [ARCHITECTURE_SSOT.md](file:///home/planetazul3/x.titan/docs/reference/ARCHITECTURE_SSOT.md)

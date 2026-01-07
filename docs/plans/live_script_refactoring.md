# IMPORTANT-001: Live Script Complexity Refactoring Plan

**Status**: Draft / Planning  
**Created**: 2026-01-07  
**Priority**: Medium (Technical Debt)

---

## Problem Statement

`scripts/live.py` is **974 lines**, violating the project's micro-modularity principle (files typically < 200 lines per `ARCHITECTURE_SSOT.md`).

The main function `run_live_trading` spans **~850 lines** (85-931) and contains 4 nested async functions:
- `process_ticks()` (L487-522)
- `process_candles()` (L529-658)
- `maintenance_task()` (L660-703)
- `heartbeat()` (L705-867)

---

## Proposed Extraction Modules

### 1. `scripts/cli.py` - Argument Parsing (~50 lines)
Extract CLI argument definitions from main block.

```python
# scripts/cli.py
def parse_live_trading_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", ...)
    parser.add_argument("--test", ...)
    parser.add_argument("--shadow-only", ...)
    parser.add_argument("--strategy", ...)
    # ...
    return parser.parse_args()
```

### 2. `scripts/heartbeat.py` - Heartbeat Task (~160 lines)
Extract the heartbeat coroutine with hot-reload logic.

```python
# scripts/heartbeat.py
async def create_heartbeat_task(context: HeartbeatContext) -> None:
    """Periodic status logging and hot-reload monitoring."""
    ...
```

### 3. `scripts/event_handlers.py` - Tick/Candle Processing (~170 lines)
Extract event processing coroutines.

```python
# scripts/event_handlers.py
async def process_ticks(event_bus, buffer, synchronizer, ...) -> None:
    ...

async def process_candles(event_bus, buffer, orchestrator, ...) -> None:
    ...
```

### 4. `scripts/maintenance.py` - Background Tasks (~45 lines)
Extract log cleanup and DB pruning.

```python
# scripts/maintenance.py
async def maintenance_task(shadow_store, log_dir, settings) -> None:
    ...
```

### 5. `utils/bootstrap.py` - Position Sizer Factory (extend existing)
Move position sizer creation logic (~65 lines) to existing bootstrap module.

---

## Refactoring Strategy

| Phase | Scope | Risk | Effort |
|-------|-------|------|--------|
| 1 | Extract CLI args | Low | 1 hour |
| 2 | Extract maintenance_task | Low | 1 hour |
| 3 | Extract heartbeat | Medium | 2 hours |
| 4 | Extract event handlers | Medium | 2 hours |
| 5 | Move sizer factory | Low | 1 hour |

**Total Estimated Effort**: ~7-8 hours

---

## Dependencies & Risks

### Shared State
The nested functions access many outer-scope variables via closure:
- `buffer`, `engine`, `executor`, `shadow_store`
- `calibration_monitor`, `settings`, `metrics`
- `tick_count`, `candle_count`, `inference_count`

**Mitigation**: Use dataclass context objects to pass state cleanly.

### Testing
Current integration tests may depend on monolithic structure.

**Mitigation**: Run full test suite after each extraction phase.

---

## Success Criteria

- [ ] `live.py` reduced to < 300 lines
- [ ] Each extracted module < 200 lines
- [ ] All existing tests pass
- [ ] No behavioral changes (functional equivalence)
- [ ] Clean module boundaries with explicit dependencies

---

## Next Steps

1. Create ADR for refactoring decision
2. Implement Phase 1 (CLI extraction) as proof of concept
3. Review and iterate on module boundaries

# ADR-007: Defer SQLite Connection Pooling (REC-003)

**Status**: Accepted  
**Date**: 2026-01-07  
**Deciders**: Automated evaluation review

## Context

REC-003 from the evaluation report recommended implementing database connection pooling for `SQLiteShadowStore` to improve performance under high-frequency trading load.

## Current Implementation

`sqlite_shadow_store.py` already implements:
- **WAL mode** - Enables concurrent reads during writes
- **Thread-local connections** - Per-thread connection reuse
- **Optimistic locking** - Retry with exponential backoff for conflicts
- **Async wrappers** - `run_in_executor` for non-blocking I/O

## Decision

**Defer** connection pooling implementation (e.g., `aiosqlitepool`).

## Rationale

| Factor | Assessment |
|--------|------------|
| **Risk** | Low - current WAL + thread-local pattern is solid |
| **Benefit** | Marginal unless trade volume exceeds 100/minute |
| **Complexity** | Medium - new dependency, testing, migration |
| **Priority** | Low - current implementation handles expected workload |

### Research Findings

1. SQLite's **single-writer limitation** is fundamental - pooling doesn't bypass this
2. Benefits of pooling (hot page cache, reduced connection overhead) are marginal for SQLite
3. Current x.titan design already mitigates contention via per-thread connections and optimistic locking

## Consequences

### Positive
- No new dependencies
- No migration complexity
- Current proven approach continues

### Negative
- May need revisiting if scale increases significantly

## Revisit Conditions

Implement connection pooling if:
- `SQLITE_BUSY` errors appear in production logs
- Trade volume exceeds 50-100/minute sustained
- Database latency becomes a measured bottleneck

## References

- [aiosqlitepool](https://github.com/aiosqlitepool/aiosqlitepool)
- SQLite WAL mode documentation

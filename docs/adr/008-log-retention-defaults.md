# ADR-008: Log Retention Default (Risk 5)

**Status**: Accepted  
**Date**: 2026-01-07

## Context

Risk 5 from the evaluation report suggested the default `log_retention_days=7` may be insufficient for post-incident analysis.

## Research Findings

| Regulation | Requirement |
|------------|-------------|
| SEC/FINRA | 3-6 years for **trade records** (ledgers, confirmations) |
| MiFID II | 5-7 years for **trade records** |
| General logs | No specific requirement for operational logs |

**Key Distinction**: Regulatory requirements apply to **trade records**, not operational logs.

## Current Implementation

| Data Type | Storage | Retention |
|-----------|---------|-----------|
| Trade records | SQLite (`shadow_trades`) | Configurable via `SYSTEM__DB_RETENTION_DAYS` (default: 30 days) |
| Operational logs | Log files | Configurable via `SYSTEM__LOG_RETENTION_DAYS` (default: 7 days) |

## Decision

**Keep current defaults**. The separation is appropriate:

1. **7 days for operational logs** - Sufficient for debugging/incident response
2. **30 days for trade records in DB** - Can be extended if needed
3. **User-configurable** - Both values can be overridden via environment variables

## Consequences

- No code changes needed
- Recommend documenting that production deployments should consider increasing `DB_RETENTION_DAYS` based on regulatory requirements

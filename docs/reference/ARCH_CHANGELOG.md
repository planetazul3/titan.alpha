# Architecture Changelog

This document tracks the evolution of the x.titan architecture documentation. It specifically records the consolidation process that led to the creation of `ARCHITECTURE_SSOT.md`.

## [1.0.0] - 2026-01-04

### Consolidation: Creation of `ARCHITECTURE_SSOT.md`

**Baseline**: `unica.md` (Original Design Contract)

The following changes were applied to create the Single Source of Truth (SSOT), merging `unica.md` with operational realities discovered in `AGENTS.md`, `runbook.md`, and `docs/*`.

#### 1. Integration of Safety Layers (Source: `docs/safety_mechanisms.md`)
*   **Change**: Integrated the "Swiss Cheese" safety model (H1-H5 Vetoes) directly into the Architecture definition.
*   **Justification**: `unica.md` described the *intent* of safety, but `safety_mechanisms.md` and `policy_specification.md` contained the *concrete rules* (e.g., Regime Veto logic) that are critical for production safety. These are now first-class architectural constraints.

#### 2. Resource Constraints (Source: `AGENTS.md`)
*   **Change**: Added strict Resource Limits (3.7 GiB RAM).
*   **Justification**: `unica.md` did not specify hardware constraints. The 3.7 GiB limit is a hard constraint of the deployment environment (documented in `AGENTS.md`) and fundamentally influences data loading strategies (memory mapping).

#### 3. Component Refinement
*   **Change**: Renamed/Clarified `data` subsystem components.
    *   `preprocessor.py` (from `unica.md`) is mapped to `features.py` and `buffer.py` functionalities in the SSOT to reflect the actual split between offline (training) and online (buffering) needs.
*   **Justification**: `unica.md` proposed a monolithic preprocessor. Practical implementation required separating the *stateless* feature math (`features.py`) from the *stateful* live buffering (`buffer.py`) to ensure correctness in both training and inference.

#### 4. Shadow Store Evolution (Source: `architectural_master.md`)
*   **Change**: Explicitly defined `sqlite_shadow_store.py` as the logging mechanism, replacing generic logging.
*   **Justification**: `architectural_master.md` introduced ACID-compliant SQLite storage as a replacement for the original JSON file logging to ensure data integrity during crashes.

#### 5. Deprecations
*   **Implicitly Deprecated**: `unica.md` is now considered a historical reference. `architectural_master.md` is superseded by `ARCHITECTURE_SSOT.md`.

### Documentation Reorganization
*   **Structure**: Adopted **Diátaxis** framework (`reference`, `guides`, `explanation`, `audit`).
*   **Cleanup**: Removed `unica.md`, `architectural_master.md`, `architecture.md`, `runbook.md` (consolidated), `policy_specification.md`, and `safety_mechanisms.md`.
### Governance Protocol (Meta-Architecture)
*   **[NEW] Agent Governance Contract (`AGENTS.md`)**: Transformed the agent configuration guide into a binding self-governance protocol.
    *   **Directives**: Added mandatory "Deep Web Grounding" and "No-Drift" clauses.
    *   **Workflow**: Enforced a 4-phase execution lifecycle (Discovery, Validation, Execution, Audit) based on IEEE 15288/TOGAF principles.
### Audit Remediation (Jan 2026)
*   **[SSOT] Volatility Veto (H3)**: metric definition evolved from `ATR` to `Volatility Anomaly (Percentile/StdDev)`.
    *   **Rationale**: The `ATR` metric was insufficient for detecting relative regime shifts in synthetic indices. The system now uses a `VolatilityExpert` (Autoencoder) and statistical percentiles to detect out-of-distribution events.
    *   **Status**: Justified Evolution (Audit Finding #3).

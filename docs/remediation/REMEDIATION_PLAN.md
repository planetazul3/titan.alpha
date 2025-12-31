# x.titan Remediation Plan for Google Jules

This document outlines the step-by-step remediation strategy for the x.titan trading system following the December 2025 validation audit. This plan is designed for autonomous execution by Google Jules to restore system integrity and operational readiness.

## Phase 1: Emergency Operational Restoration
**Goal**: Resolve "show-stopper" bugs preventing system entry points from executing.

1.  **Rectify Live Trading Orchestration**:
    *   Locate the variable initialization for the model health monitoring component within the primary live trading execution script. 
    *   Ensure that the monitoring instance is correctly instantiated before being passed into the main trading loop. 
    *   Address any naming inconsistencies that lead to the current error where the monitor object is referenced but not defined in the local or global scope.

2.  **Fix Dataset Path Management**:
    *   Audit the data loading logic responsible for handling Parquet files. 
    *   Modify the logic to detect whether a specified data path is a single file or a directory. 
    *   Prevent the automated creation of shadow cache directories when the source is a standalone file, as this currently triggers file system errors.
    *   Implement a more resilient caching strategy that places metadata for single-file datasets into a dedicated system-level cache directory rather than attempting to nest it within the file path itself.

## Phase 2: Architectural Consistency and Cleanup
**Goal**: Eliminate design debt and redundant legacy components.

1.  **Module De-duplication**:
    *   Identify and remove legacy components that have been superseded by newer versions (e.g., old market regime detectors and early shadow storage implementations).
    *   Update all internal import references in the brain and execution modules to point exclusively to the current architectural standards (like the hierarchical regime detector).

2.  **Validation Path Restoration**:
    *   Investigate the missing module dependencies in the pre-training validation suite.
    *   Re-map imports to ensure that the validation scripts can access the latest temporal modeling entities.
    *   Ensure the validation suite is synchronized with the new project structure (domain-driven design).

## Phase 3: Quality, Reliability, and Safety
**Goal**: Address latent code quality issues and improve test visibility.

1.  **Static Analysis Remediation**:
    *   Prioritize fixing the indentation and shadowing errors in the shadow trade resolution logic, as these pose a risk of silent logical failures.
    *   Consolidated redundant imports identified by linter results to improve compilation speed and code clarity.

2.  **Closing the Testing Gap**:
    *   Develop targeted unit tests for the online learning and Continual Learning (EWC) components, which currently have low coverage.
    *   Verify the mathematical stability of the weight updating mechanism through simulation.
    *   Extend the integration test suite to include automated "dry runs" of the dashboard and API services.

## Phase 4: Performance and Observability
**Goal**: Optimize execution speed and monitoring depth.

1.  **Inference Latency Optimization**:
    *   Apply graph compilation techniques to the core neural network models to reduce the average inference latency beneath the 100ms threshold.
    *   Implement asynchronous checkpoint verification to prevent the ~1s startup delay from blocking the initial tick processing.

2.  **Monitoring Extensions**:
    *   Expand the real-time metrics to include per-layer inference timing to better diagnose future performance regressions.
    *   Ensure the risk management veto decisions are explicitly logged with the raw metrics that triggered them for post-mortem analysis.

---

## Best Practices Summary (Research-Based)
*   **Dataset Management**: When using Parquet, leverage columnar reading for sub-feature extraction. Always decouple the data manifest from the physical file path to support varying storage formats (local vs. cloud).
*   **Live Monitoring**: Use a specialized initialization pattern for system-wide observers. Observers should be registered in a global registry or a well-defined shared state to avoid scope-related initialization errors during high-pressure trading loops.
*   **Refactoring Cleanup**: Use "Incremental Deletion" for legacy code. First, mark modules as deprecated with warnings; once the test suite and all integration points confirm zero usage, physically remove the files to minimize project surface area.

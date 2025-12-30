# API_CONTRACT_ANALYSIS.md

## Critical Interfaces Analysis

### 1. `DerivOmniModel.forward`
- **Location**: `models/core.py`
- **Signature**: `(self, ticks: torch.Tensor, candles: torch.Tensor, vol_metrics: torch.Tensor, masks: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]`
- **Status**: ✅ Stable. Matches requirements for multi-input expert fusion.

### 2. `DecisionEngine.process_model_output`
- **Location**: `execution/decision.py`
- **Signature**: `(self, probs: dict[str, float], reconstruction_error: float, timestamp: datetime | None = None, market_data: dict[str, Any] | None = None, entry_price: float | None = None) -> list[TradeSignal]`
- **Status**: ✅ Stable. Correctly handles multi-task probabilities and volatility error.

### 3. `ExecutionPolicy.check_vetoes`
- **Location**: `execution/policy.py`
- **Signature**: `(self, reconstruction_error: float) -> tuple[bool, str]`
- **Status**: ✅ Unified. Veto logic has been correctly centralized here (Ref commit `14e61e0`).

### 4. `DerivDataset.__init__`
- **Location**: `data/dataset.py`
- **Signature**: `(self, data_source: Path, settings: Settings, mode: Literal["train", "eval"] = "train", lookahead_candles: int = 5)`
- **Deviation**: Requires the ROOT `data_source` directory to find partitions, but some scripts might passed subdirectories (e.g. `candles_60`).

## Breaking Changes Detected

- **Logging**: `utils.logging_setup.bootstrap_logging` no longer exists. All callers must switch to `config.logging_config`.
- **Regime Veto**: `DecisionEngine` no longer manages its own veto thresholds directly; it delegates to the `RegimeVeto` protocol/implementation.

## New APIs Introduced
- `core.interfaces.IExecutor`: Protocol for future execution engines.
- `core.interfaces.IStrategy`: Protocol for future strategy experts.
- `execution.sqlite_shadow_store`: New high-performance persistence layer for simulations.

## Backward Compatibility Issues
- Legacy models that do not support the `masks` argument in `forward` will fail if the dataset provides them.

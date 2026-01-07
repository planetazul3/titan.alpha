# x.titan Architecture: Single Source of Truth

> [!IMPORTANT]
> This document is the **definitive, canonical reference** for the x.titan trading system. It supersedes `unica.md` and all prior architectural documentation. Any deviation from this specification in the code must be treated as a bug or requires a formal update to this document.

## 1. Introduction and Goals

### 1.1 Vision
x.titan is a high-frequency, deep-learning-based algorithmic trading system designed for binary options markets (specifically Deriv.com). It aims to achieve consistent profitability by leveraging a **multi-expert neural network formulation** that separates temporal, spatial, and volatility analysis.

### 1.2 Core Design Principles
*   **Safety First (Swiss Cheese Model)**: Execution logic is decoupled from signal generation. Multiple independent safety layers (Regime Veto, Risk Caps, Circuit Breakers) must *all* align for a trade to execute.
*   **Micro-Modularity**: The codebase is composed of small, focused files (typically <200 lines) with single responsibilities.
*   **Canonical Data**: A single shared feature generation pipeline ensures that training data and real-time inference data are mathematically identical.
*   **Shadow Verification**: Every decision is logged to a persistent "Shadow Store", allowing for counterfactual analysis and safe validation of model updates.

## 2. Constraints

### 2.1 Technical Constraints
*   **Runtime Environment**: Python 3.12 (strict requirement).
*   **Resource Limits**:
    *   **RAM**: Unconstrained (System Dependent). Large datasets should still use memory-mapping for efficiency.
    *   **Concurrency**: `asyncio` for I/O bound tasks, avoiding blocking calls in the main loop.
*   **Dependencies**:
    *   **ML**: PyTorch (with CUDA/MPS acceleration).
    *   **Data**: NumPy/SciPy (Pandas used sparingly or not at all in hot paths).
    *   **Config**: Pydantic for strong typing.

### 2.2 Organizational & Regulatory Constraints
*   **Auditability**: All trade decisions must be traceable to specific model inputs and versions.
*   **Hardening**: No `NaN` propagation allowed in risk calculations (RC-8). Use `math.isfinite()`.
*   **API Usage**: Strict adherence to Deriv API rate limits to avoid account bans.

## 3. Context and Scope

The system operates as an autonomous agent interacting with the Deriv.com trading platform.

*   **Inputs**: Real-time tick stream (WebSocket), OHLC candles.
*   **Outputs**: Trade execution commands (Buy/Sell), Heartbeat metrics, Logs.
*   **Actors**:
    *   **Deriv API**: External platform for market data and execution.
    *   **Operator**: Manages lifecycle (Start/Stop) and monitors via Dashboard.

## 4. Solution Strategy

To meet the high-frequency and safety requirements, the architecture employs:
1.  **Asynchronous Event Loop**: For handling high-throughput WebSocket data without blocking.
2.  **DerivOmniModel**: A unified model architecture combining:
    *   **Temporal Expert**: BiLSTM + Attention (Trend detection).
    *   **Spatial Expert**: CNN Pyramid (Pattern/Shape detection).
    *   **Volatility Expert**: Autoencoder (Regime detection & anomaly scoring).
3.  **Strict Typing**: Extensive use of Python type hints and Pydantic validation to fail fast on configuration or data errors.

## 5. Building Block View

### 5.1 Level 1: High-Level Subsystems

| Module | Responsibility | Key Components |
| :--- | :--- | :--- |
| `config` | **Central Configuration**. Single source of truth for all tunable parameters. | `settings.py`, `constants.py` |
| `data` | **Data Pipeline**. Ingestion, normalization, and feature engineering. | `features.py`, `buffer.py`, `DerivDataset` |
| `models` | **inference Engine**. Neural network definitions and forward pass logic. | `core.py`, `temporal.py`, `spatial.py` |
| `execution` | **Decision & Safety**. Business logic, risk management, and API bridging. | `decision.py`, `safety.py`, `executor.py`, `position_sizer.py`, `backtest.py` |
| `training` | **Optimization**. Training loops and validation. | `train.py`, `trainer.py` |

### 5.2 Level 2: Component Details

#### 5.2.1 Data Subsystem (`data/`)
*   **`features.py`**: The **Canonical Feature Source**. Contains the `FeatureBuilder` class. It enforces that the exact same math (log returns, Z-scores) is applied in both `scripts/train.py` and `scripts/live.py`.
*   **`common/schema.py`**: Defines `FeatureSchema`, `CandleInputSchema`, and `FeatureOutputSchema`. Enforces strict shape and dtype validation.
*   **`validation/schemas.py`**: Defines `TickSchema` and `CandleSchema` using **Pandera**. Enforces strict statistical validation (e.g., High >= Low) on raw data inputs before processing.
*   **`buffer.py`**: `MarketDataBuffer` manages the sliding window of live data, ensuring the model always has a complete context window (e.g., last 200 candles) for inference.
*   **`normalizers.py`**: Pure NumPy implementations of mathematical transforms.

#### 5.2.2 Model Subsystem (`models/`)
*   **`core.py`**: `DerivOmniModel`. The orchestrator that calls the sub-experts and fuses their embeddings.
*   **`temporal.py`**: Captures time-dependencies. Input: Sequence of candles. Output: Temporal Embedding.
*   **`spatial.py`**: Captures geometric properties of the price curve. Input: Raw ticks/high-res sequence. Output: Spatial Embedding.
*   **`volatility.py`**: Autoencoder. Input: Volatility metrics. Output: Reconstruction Error (used for Regime Veto).

#### 5.2.3 Execution Subsystem (`execution/`)
*   **`decision/core.py`**: `DecisionEngine`. Central coordinator for trading decisions. Evaluates model probabilities against thresholds. Integrates `ProbabilityCalibrator` and `EnsembleStrategy` for robust signal generation.
*   **`decision_logic.py`**: Pure functional logic for processing signals and applying caution filters.
*   **`signal_adapter_service.py`**: **Adaptation Layer**. Centralizes conversion of TradeSignals to ExecutionRequests, handling duration resolution via `ContractParameterService` and position sizing.
*   **`contract_params.py`**: **Parameter Resolution**. Centralized logic for resolving contract durations and barriers, preventing training/execution mismatches.
*   **`idempotency_store.py`**: **Deduplication**. SQLite-based store preventing duplicate execution of the same signal ID.
*   **`position_sizer.py`**: **Dynamic Risk Management**. Calculates optimal stake using `KellyPositionSizer` or `TargetVolatilitySizer`, decoupled from signal logic.
*   **`regime/veto.py`**: `RegimeVeto`. Checks the Volatility Expert's reconstruction error. If Error > `REGIME_VETO_THRESHOLD`, the signal is vetoed immediately.
*   **`safety/core.py`**: `SafeTradeExecutor`. The final gatekeeper. Checks H1-H6 policies.
*   **`backtest.py`**: **Event-Driven Replay**. Replays historical data through the full pipeline (`Buffer` -> `Model` -> `Decision`) to ensure "What you test is what you fly".
*   **`sqlite_shadow_store.py`**: ACID-compliant storage for every decision made. Uses **Optimistic Concurrency Control (OCC)**.

#### 5.2.4 Calibration & Ensembling (`execution/`)
*   **`calibration.py`**: `ProbabilityCalibrator`. Maps raw model logits/probabilities to realized win rates using Isotonic Regression.
*   **`ensemble.py`**: `EnsembleStrategy`. Combines predictions from multiple model checkpoints (e.g., Voting, Weighted Average) to reduce variance.

## 6. Runtime View

### 6.1 Live Trading Pipeline
1.  **Ingest**: `DerivClient` receives a new Candle Close event.
2.  **Update**: `MarketDataBuffer` appends the candle and updates sliding windows.
3.  **Process**: `FeatureBuilder` takes the window and generates a Normalized Tensor.
4.  **Infer**: `DerivOmniModel` runs the forward pass. Returns raw probabilities.
5.  **Calibrate & Ensemble**: `ProbabilityCalibrator` maps raw outputs to win rates. `EnsembleStrategy` combines multiple model outputs.
6.  **Evaluate**: `DecisionEngine` checks thresholds.
    *   *IF* `reconstruction_error` > Limit: **VETO** (Unknown Regime).
    *   *ELSE IF* `probability` < Threshold: **IGNORE**.
    *   *ELSE*: Generate `TradeSignal`.
7.  **Safeguard**: `SafeTradeExecutor` runs H1-H6 policies.
    *   *IF* Pass: Execute Trade via API.
    *   *IF* Fail: Log rejection code.
8.  **Record**: `ShadowStore` records inputs, model outputs, and final decision.

## 7. Cross-cutting Concepts

### 7.1 Safety Policies (The "Hard Vetoes")
| Rule ID | Name | Trigger Condition | Action |
| :--- | :--- | :--- | :--- |
| **H1** | Daily Loss Limit | Cumulative P&L <= `-MAX_DAILY_LOSS` | Halt trading until reset (00:00 UTC) |
| **H2** | Stake Cap | Stake > `MAX_STAKE` | Reject trade |
| **H3** | Volatility Veto | Volatility Anomaly > `REGIME_VETO_THRESHOLD` (Percentile/StdDev) | Veto Signal |
| **H4** | Warmup Veto | Buffer candles < `WARMUP_PERIOD` | Reject Signal |
| **H5** | Regime Veto | Regime == `UNCERTAIN` (High AE Error) | Veto Signal |
| **H6** | Staleness Veto | Data Latency > `STALE_THRESHOLD` | Reject Signal (Data Quality) |

### 7.2 Persistence
*   **Safety State**: stored in `data_cache/safety_state.db`. Contains items like `daily_pnl` and `consecutive_losses`. Must persist across process restarts to prevent risk limit bypass.
*   **Shadow Store**: stored in `data_cache/shadow_trades.db`. Logging system for analysis.

## 8. Deployment View
*   **Production**: Deployed via Docker/Devcontainer.
*   **Configuration**: All environment-specific variables (API Keys, Limits) are injected via `.env` file, loaded by `config/settings.py`.
*   **Entry Points**:
    *   `scripts/live.py`: Main daemon.
    *   `scripts/train.py`: Training job.
    *   `scripts/backtest.py`: Historical simulation and validation.

## 9. Glossary
*   **Shadow Trade**: A simulated trade logged to the database used for validating model performance on live data without financial risk.
*   **Regime Veto**: A mechanism where the model self-identifies that market conditions are "out of distribution" compared to its training data, causing it to withdraw from trading.
*   **DerivOmniModel**: The specific name of the neural network architecture used in x.titan.

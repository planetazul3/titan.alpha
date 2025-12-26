# x.titan DerivOmniModel: Architectural Master Document

## 1. System Overview
The **DerivOmniModel** is a sophisticated, modular deep learning trading system designed for binary options trading on Deriv.com. It is architected to separate **signal generation** (ML inference) from **execution logic** (Safety/Decision Engine), ensuring that model probabilities are strictly governed by risk management rules before any trade is placed.

### High-Level Architecture
The system follows a strict data flow pipeline:
1.  **Ingestion**: Real-time WebSocket ticks and candles.
2.  **Processing**: Feature engineering and buffering.
3.  **Inference**: Multi-expert neural network (Temporal, Spatial, Volatility).
4.  **Decision**: Business logic application (Regime detection, Filters, Vetoes).
5.  **Execution**: Safety checks and API interaction.

### Core Design Principles
-   **Safety First**: Execution logic is the final authority. Models only yield probabilities.
-   **Modular Experts**: Specialized sub-models for different market aspects (Time, Space, Volatility).
-   **Shadow Verification**: All decisions are logged to a "Shadow Store" for continual learning and validation without risk.
-   **Canonical Data**: A single source of truth for feature engineering facilitates both training and inference.

## 2. Component Architecture

### `config/`
**Purpose**: Centralized, type-safe configuration.
-   `settings.py`: **Single Source of Truth** for configuration. Uses `pydantic` to load and validate environment variables.
-   `constants.py`: Immutable system constants (CONTRACT_TYPES, TIME_FRAMES).
-   `logging_config.py`: Structured JSON logging for production, colored for dev.

### `data/`
**Purpose**: Data ingestion, normalization, and feature engineering.
-   `ingestion/`: `DerivClient` and `DerivEventAdapter` for normalized tick/candle streaming.
-   `features.py`: **Canonical Feature Source**. `FeatureBuilder` class ensures training and inference use *identical* transformations.
-   `processor.py`: Internal preprocessors (log returns, Z-score, volatility metrics).
-   `buffer.py`: `MarketDataBuffer` manages sliding windows and candle close detection for live inference.
-   `dataset.py`: `DerivDataset` for efficient Parquet loading and lazy feature generation during training.

### `models/`
**Purpose**: Deep learning architecture.
-   `core.py`: `DerivOmniModel` - The unified "Brain" combining all experts.
-   `temporal.py` / `spatial.py` / `volatility.py`: Specialized expert networks.
-   `fusion.py`: Feature gating and fusion layers.
-   **Philosophy**: Model outputs *probabilities* and *embeddings* only. It does **not** make trade decisions.

### `execution/`
**Purpose**: Deterministic business logic and safety enforcement.
-   `decision.py`: `DecisionEngine` - The final authority on "To trade or not to trade".
-   `regime.py`: `RegimeVeto` - **First-Class Authority**. Can unconditionally veto trades based on autoencoder reconstruction error.
-   `safety.py`: `SafeTradeExecutor` - The "Safety Wrapper" implementing Kill Switch, Circuit Breaker, P&L Cap, and Rate Limits.
-   `sqlite_shadow_store.py`: **ACID-Compliant** storage for shadow trades and outcomes.
-   `executor.py`: Low-level Deriv API interaction (wrapped by SafeTradeExecutor).

### `training/`
**Purpose**: Model optimization loop.
-   `train.py`: Unified training implementation.
-   `trainer.py`: Abstraction for training loop, validation, and checkpointing.

### `observability/`
**Purpose**: Production monitoring and health checks.
-   `performance_tracker.py`: Real-time latency (p50/p95/p99) and memory monitoring.
-   `model_health.py`: Detects concept drift, accuracy degradation, and miscalibration.
-   `dashboard.py`: Aggregates system health for status reporting.

---

## 3. Data Flow & Interconnections

This section maps the lifecycle of data from the external API to a final trade execution.

### Live Trading Pipeline (`scripts/live.py`)

1.  **Ingestion Layer**:
    -   `DerivClient` connects to WebSocket.
    -   `DerivEventAdapter` normalizes raw JSON messages into `TickEvent` and `CandleEvent`.
    -   **Strict Boundary**: Downstream components never see raw API responses.

2.  **Buffering & State**:
    -   `MarketDataBuffer` receives events.
    -   Updates internal sliding windows (ticks queue, candles list).
    -   **Trigger**: On `CandleEvent` close -> checks `is_ready()` -> Triggers Inference.

3.  **Feature Engineering (Canonical)**:
    -   `DecisionEngine` calls `FeatureBuilder.build(buffer_snapshot)`.
    -   `FeatureBuilder` applies **identical** normalization/indicators as used in `DerivDataset`.
    -   **Output**: Normalized Tensors (`[Batch, Seq_Len, Features]`).

4.  **Inference**:
    -   `DerivOmniModel` performs forward pass.
    -   **Outputs**:
        -   `probabilities`: Dict of `UP/DOWN/TOUCH/etc.` probabilities.
        -   `reconstruction_error`: Scalar value from Volatility Expert (Audit Metric).

5.  **Decision & Regime Veto**:
    -   `DecisionEngine` receives model outputs.
    -   **Step 1: Regime Check**: `RegimeVeto` compares `reconstruction_error` vs `veto_threshold`.
        -   *If Vetoed*: Returns `[]` (Empty List). **Absolute Stop**.
    -   **Step 2: Signal Filtering**: Checks `probabilities` vs `confidence_threshold`.
    -   **Step 3: Shadow Logging**: Records decision to `SQLiteShadowStore` (regardless of execution).

6.  **Execution & Safety**:
    -   `SafeTradeExecutor` receives `TradeSignal`.
    -   **Safety Checks (Sequential)**:
        1.  **Kill Switch**: Is it active?
        2.  **Circuit Breaker**: Too many consecutive failures?
        3.  **Daily P&L**: Loss limit reached?
        4.  **Rate Limit**: Global/Per-symbol limits ok?
    -   **Action**: Calls `DerivTradeExecutor.execute()` -> API buy.

---

## 4. Key Module Deep Dives

### Execution Safety (`execution/safety.py`)
-   **Design**: Implements the "Swiss Cheese" model of safety. Multiple layers must potentially fail for a bad trade to pass.
-   **Storage**: Uses `SQLiteSafetyStateStore` for robust state persistence (crash recovery).
-   **Testing**: Validated by `tests/test_safety.py` covering all edge cases.

### Regime Detection (`execution/regime.py`)
-   **Philosophy**: "Uncertainty Awareness". The model knows when it doesn't know.
-   **Mechanism**: Uses an Autoencoder within the Volatility Expert. High reconstruction error -> Input data is unlike training distribution -> **Unknown Regime** -> **Veto**.

### Shadow Store (`execution/sqlite_shadow_store.py`)
-   **Evolution**: Replaced legacy JSON logger.
-   **Purpose**: Records EVERYTHING needed to replay a decision: `tick_window`, `candle_window`, `model_version`,probs.
-   **Benefit**: Allows for "Counterfactual Analysis" - *What would the new model have done on last week's data?*

---

## 5. Entry Points

-   `scripts/live.py`: **Production Daemon**. Orchestrates the entire pipeline above. Includes `CalibrationMonitor` for graceful degradation.
-   `scripts/train.py`: **Training CLI**. Uses `DerivDataset` and `Trainer` to optimize `DerivOmniModel`.
-   `scripts/download_data.py`: **Data Utility**. Fetches historical data for training.

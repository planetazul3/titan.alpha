# Operational Mandate & Architecture SSOT (v1.1)

> [!IMPORTANT]
> **Primary Objective**: Sustained Profitability.
> This document overrides all previous architectural constraints. If a "best practice" blocks profitability or slows down execution, it must be discarded.

## 1. Vision & Core Philosophy

**Goal**: Build an automated binary options trading system that makes money.
**Operator**: Single Developer + AI Agent.
**Approach**: Pragmatic, experimental, and autonomous.

### 1.1 Core Principles
1.  **Profit > Code Quality**: A messy script that makes money is better than a perfect architecture that loses money.
2.  **Bias for Action**: Implement, test, and iterate. Do not get stuck in "Planning" or "Architecture Review".
3.  **Dynamic Evolution**: The system structure is mutable. If a component is too complex, delete it. If a library helps, add it.
4.  **Real-World Validation**: Backtests are just hints. Small-scale live testing (or "Shadow Trading" on live data) is the only metric that matters.

### 1.2 Success Criteria ("What Works")
A component or system is considered **functionally acceptable** only when it meets these real-world metrics:

*   **Component**: Runs without crashing for 1 hour of live data processing.
*   **System (Shadow Mode)**:
    *   Executes > 20 trades.
    *   Achieves **Positive Expectancy** (Win Rate > 53-55% depending on payout).
*   **System (Live Mode)**:
    *   Net Profit > $0 after a daily session.

*If it doesn't print money or run reliably, it is broken, regardless of code quality.*

## 2. System Scope

The system acts as an autonomous execution engine interacting with the Deriv.com platform.

*   **Inputs**: Market Data (Price, Volume, Time).
*   **Decision**: Buy/Sell/Hold, Duration, Stake.
*   **Outputs**: API calls to execute trades.

## 3. Pragmatic Architecture

The architecture is streamlined to three functional layers.

### 3.1 Data Layer (Ingest & Process)
*   **Responsibility**: Get market data, make it usable.
*   **Constraint**: Must act faster than the decision tick.
*   **Components**: Client (WebSocket), Buffer, Features.

### 3.2 Decision Layer (The Brain)
*   **Responsibility**: Predict market direction.
*   **Implementation**: Flexible. Deep Learning (`DerivOmniModel`), XGBoost, or Heuristics.
*   **Rule**: If the "Advanced Model" is too heavy or slow, simplify it.

### 3.3 Execution Layer (The Hands)
*   **Responsibility**: Execute the trade and manage risk.
*   **Components**: Executor, Safety Guards (minimal).

### 3.4 Operational File Map
*   `data/`: Ingestion and Feature engineering.
*   `models/`: AI/ML logic.
*   `execution/`: Trade logic and API calls.
*   `scripts/`: Entry points (train, live, backtest).
*   `config/`: Settings.

### 3.5 Optimization Priorities (The Decision Framework)
When making trade-offs, the AI Dev must prioritize in this order:

1.  **Execution Reliability**: The system must run and trade. If it crashes, nothing else matters.
2.  **Profitability (Expectancy)**: Win Rate and Payout optimization.
3.  **Speed of Development**: Ship features fast to test them.
4.  **Code Cleanliness**: Refactor only if it speeds up #3.
5.  **Safety/Security**: Only preventing catastrophic loss (account drain).

## 4. Operational Safety (The "Kill Switch" Model)

We replace complex "Swiss Cheese" models with simple, hard stops.

1.  **Daily Loss Limit (Hard Stop)**: If Loss > `MAX_DAILY_LOSS`, STOP trading for the day. (Non-negotiable).
2.  **Stake Cap**: Never bet more than `MAX_STAKE`.
3.  **Sanity Checks**:
    *   Don't trade on stale data (> 5s latency).
    *   Don't trade if data is clearly broken (NaNs).

**Everything else (Regime detection, volatility filters) is a feature, not a safety requirement.** Adding them is an optimization, not a mandate.

## 5. Technology Stack

*   **Language**: Python.
*   **Libraries**: PyTorch, NumPy, Pandera (optional).
*   **Deployment**: Docker/Local.

## 6. Development Protocol for AI Agents

1.  **Autonomous Research**: Check the web for strategy ideas. Don't ask user permission.
2.  **Implement Fully**: Write the code, run it.
3.  **Fix Forward**: Use real data failures to fix logic. Don't revert to safety frameworks.
4.  **Simplify**: Aggressively delete unused abstractions.

### 6.1 Definition of Done
A task or module is **DONE** only when:
*   [ ] **It Runs**: The code executes without syntax errors or immediate crashes.
*   [ ] **It is Observable**: It logs clear activity (e.g., "Trade Executed", "Signal 0.85").
*   [ ] **No Loose Ends**: No critical `TODO`, `pass`, or implementation placeholders in the main path.
*   [ ] **Delivered**: The user can run a command and see it work.

## 7. Persistence & State

*   **Shadow Mode**: Log decisions to evaluate performance.
*   **Live Mode**: Turn on when Shadow Mode shows positive expectancy.

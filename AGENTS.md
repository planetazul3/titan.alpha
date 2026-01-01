# AGENTS.md

Configuration guide for the **Google Jules** agent in the **x.titan** project.

## 🧠 Project Context
x.titan is a high-frequency algorithmic trading system based on Deep Learning (PyTorch).
- **Core:** Python 3.12 (Upgraded Jan 2026)
- **ML Engine:** PyTorch with Temporal Fusion Transformer (`models/`, `training/`)
- **Execution:** Asyncio + Websockets (`execution/`, `api/`)
- **Data:** Scipy/Numpy for technical indicators (`data/`) — TA-Lib is available but legacy.
- **Domain:** Core domain models and interfaces (`core/`)

## 🏗️ Architecture Map
| Directory | Purpose | Important Notes |
| :--- | :--- | :--- |
| `/core` | Domain models and interfaces | Clean architecture base classes. |
| `/execution` | Real-time trading logic | **CRITICAL:** Any change here requires extra safety review. |
| `/models` | Neural network definitions | PyTorch models (DerivOmniModel with TFT). |
| `/training` | Training and validation loops | Use `scripts/train.py` as the entrypoint. |
| `/config` | Global configuration (Pydantic) | `settings.py` centralizes all config (~15 config sections). |
| `/data` | Data ingestion and features | Includes indicators, normalizers, and feature builders. |
| `/tools` | Validation and utility scripts | Checkpoint verification, import validation, benchmarks. |
| `/tests` | Automated tests (Pytest) | Tests are marked (slow, integration, requires_gpu). |
| `/observability` | Metrics and monitoring | Prometheus, health checks, tracing. |

## 🛠️ Development Commands
Use these commands to verify your work. Do not invent new commands.

### Testing
The full suite is slow. Use markers to be efficient:
- **Fast Tests (Unit):** `./venv/bin/pytest -m "not slow and not integration"`
- **Integration Tests:** `./venv/bin/pytest -m integration`
- **All Tests:** `./venv/bin/pytest` (Run only before submitting the final task)
- **Hardening Validation:** `./venv/bin/pytest tests/test_risk_nan_hardening.py`

### Code Quality (Linting)
The project is strict about typing and style.
- **Lint & Format:** `ruff check .`
- **Type Checking:** `mypy .` (Config is in `pyproject.toml`, ignores `python-deriv-api`)

### Validation Tools
- **Verify Checkpoint:** `python tools/verify_checkpoint.py --checkpoint checkpoints/best_model.pt`
- **Validate Imports:** `python tools/validation/validate_imports.py`

## ⚠️ Safety Rules (Safety Veto)
1. **Never** change `ENVIRONMENT` to "production" or `KILL_SWITCH_ENABLED` to false in default configuration files without explicit approval.
2. If you modify `execution/safety.py`, `execution/policy.py`, or `execution/adaptive_risk.py`, you must mandatorily run `pytest tests/test_execution.py` and `tests/test_adaptive_risk.py`.
3. **Numerical Stability:** Always validate numerical inputs in risk/execution modules using `math.isfinite()`. `NaN` values are strictly forbidden in P&L and Equity tracking (RC-8).
4. Do not upload real API Keys or tokens to the code. Use `os.getenv` and assume they will be in the environment.
5. The `core/` directory contains domain invariants - changes here affect the entire system.

## 📦 Dependencies
- If you need a new library, add it to `requirements.txt`.
- `TA-Lib` (system library) is pre-installed in the agent's environment via the setup script.
- **Custom Libs:** The custom `python-deriv-api` is installed from the fork `planetazul3/python-deriv-api`.

## ⚙️ Configuration Structure
Configuration is managed via Pydantic in `config/settings.py`. Key sections:
- `Trading` - Symbol, timeframe, stake, payout ratio, barriers
- `Thresholds` - Confidence thresholds for real vs shadow trades
- `Hyperparams` - Model architecture (TFT, embedding dims, dropout rates)
- `DataShapes` - Sequence lengths, feature dimensions, label thresholds
- `ExecutionSafety` - Rate limits, daily loss caps, circuit breakers
- `ShadowTrade` - Duration per contract type, staleness detection
- `Normalization` - Scaling factors for technical indicators
- `System` - Log retention, database paths

## 📜 Project Audit & Hardening Protocol
When performing audits or hardening tasks, strictly follow the [Project Audit Prompt](file:///.agent/prompts/project_audit_prompt.md).

### ⚡ Non-Negotiable Constraints (Audit Findings)
- **RAM Management:** The system operates under a strict **3.7GiB RAM limit**. Use memory-mapping for large datasets and avoid `torch.cat` or large list/tensor concatenations.
- **Numerical Safety (RC-8):** Always use `math.isfinite()` when updating risk metrics or recording P&L. `NaN` or `Inf` propagation is a critical failure.
- **Temporal Integrity:** Training must maintain a minimum gap of **200+ candles** between training and validation sets to prevent data leakage.
- **Fisher Information:** Always preserve OFFLINE knowledge during ONLINE updates by loading `fisher_state_dict`.

### 🔄 Structured Hardening Workflow
1.  **Inventory:** Discover all runnable targets and entry points.
2.  **Triage:** Trace every failure to its root cause before attempting a fix.
3.  **TDD Fix:** Write a failing test for every bug identified (e.g., `tests/test_risk_nan_hardening.py`).
4.  **Verification:** Citations from official docs are required for non-trivial architecture changes.

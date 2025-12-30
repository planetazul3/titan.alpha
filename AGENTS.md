# AGENTS.md

Configuration guide for the **Google Jules** agent in the **x.titan** project.

## 🧠 Project Context
x.titan is a high-frequency algorithmic trading system based on Deep Learning (PyTorch).
- **Core:** Python 3.10
- **ML Engine:** PyTorch with Temporal Fusion Transformer (`models/`, `training/`)
- **Execution:** Asyncio + Websockets (`execution/`, `api/`)
- **Data:** TA-Lib for technical indicators (`data/`)
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
- **Fast Tests (Unit):** `pytest -m "not slow and not integration"`
- **Integration Tests:** `pytest -m integration`
- **All Tests:** `pytest` (Run only before submitting the final task)

### Code Quality (Linting)
The project is strict about typing and style.
- **Lint & Format:** `ruff check .`
- **Type Checking:** `mypy .` (Config is in `pyproject.toml`, ignores `python-deriv-api`)

### Validation Tools
- **Verify Checkpoint:** `python tools/verify_checkpoint.py --checkpoint checkpoints/best_model.pt`
- **Validate Imports:** `python tools/validation/validate_imports.py`

## ⚠️ Safety Rules (Safety Veto)
1. **Never** change `ENVIRONMENT` to "production" or `KILL_SWITCH_ENABLED` to false in default configuration files without explicit approval.
2. If you modify `execution/safety.py` or `execution/policy.py`, you must mandatorily run `pytest tests/test_execution.py`.
3. Do not upload real API Keys or tokens to the code. Use `os.getenv` and assume they will be in the environment.
4. The `core/` directory contains domain invariants - changes here affect the entire system.

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

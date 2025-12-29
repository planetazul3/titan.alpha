# AGENTS.md

Configuration guide for the **Google Jules** agent in the **x.titan** project.

## 🧠 Project Context
x.titan is a high-frequency algorithmic trading system based on Deep Learning (PyTorch).
- **Core:** Python 3.10
- **ML Engine:** PyTorch (`models/`, `training/`)
- **Execution:** Asyncio + Websockets (`execution/`, `api/`)
- **Data:** TA-Lib for technical indicators (`data/`)

## 🏗️ Architecture Map
| Directory | Purpose | Important Notes |
| :--- | :--- | :--- |
| `/execution` | Real-time trading logic | **CRITICAL:** Any change here requires extra safety review. |
| `/models` | Neural network definitions | PyTorch models. |
| `/training` | Training and validation loops | Use `train.py` as the entrypoint. |
| `/config` | Global configuration (Pydantic) | `settings.py` centralizes all config. |
| `/tests` | Automated tests (Pytest) | Tests are marked (slow, integration, requires_gpu). |

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

## ⚠️ Safety Rules (Safety Veto)
1. **Never** change `TRADING_MODE` to "live" in default configuration files.
2. If you modify `execution/safety.py`, you must mandatorily run `tests/test_execution.py`.
3. Do not upload real API Keys to the code. Use `os.getenv` and assume they will be in the environment.

## 📦 Dependencies
- If you need a new library, add it to `requirements.txt`.
- Note that `TA-Lib` (system library) is already pre-installed in the agent's environment via the setup script.
- **Custom Libs:** The custom `python-deriv-api` is installed from the fork `planetazul3/python-deriv-api`.

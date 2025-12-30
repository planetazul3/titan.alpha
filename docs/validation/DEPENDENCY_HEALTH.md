# DEPENDENCY_HEALTH.md

## Core Dependencies Status

| Library | Required | Installed | Status |
|---------|----------|-----------|--------|
| `torch` | `>=2.0.0` | `2.6.0+cu124` | ✅ Healthy |
| `pandas`| `>=2.0.0` | `2.2.3` | ✅ Healthy |
| `numpy` | `>=1.24.0`| `2.0.2` | ✅ Healthy |
| `pydantic`| `>=2.0.0` | `2.10.4` | ✅ Healthy |
| `TA-Lib`| `>=0.4.0` | `0.4.32` | ✅ Healthy |

## ❌ MISSING DEPENDENCIES (Detected during tests)

1. **`freezegun`**: Required by `tests/test_m13_disk_management.py`. Not listed in `requirements.txt`.
2. **`pylint` / `flake8` / `bandit` / `radon`**: Static analysis tools requested for validation but not present in the environment.

## 🟠 VULNERABILITY & RISK ASSESSMENT

- **`python-deriv-api` (Local/Forked)**: The system relies on a local fork of `python-deriv-api`. While this allows for custom fixes, it creates a maintenance burden for keeping it in sync with the upstream repository.
- **Python Version**: System is running 3.12 (from `venv`), while some older scripts might have 3.10 artifacts (`.mypy_cache/3.10`). Compatibility appears okay but should be monitored.
- **`websockets>=14.0`**: Using a very recent version of websockets.

## System Requirements Audit
- **TA-Lib**: Binary dependency confirmed present.
- **SQLite3**: Python `sqlite3` module present, but `sqlite3` CLI tool is missing from the system (requires `apt install sqlite3`).
- **CUDA/GPU**: Not detected/available on this host. System is falling back to CPU for all ML operations.

## Recommendation
- Add `freezegun` and dev-tools (`pylint`, `bandit`) to `requirements-dev.txt`.
- Install `sqlite3` system utility for easier debugging.

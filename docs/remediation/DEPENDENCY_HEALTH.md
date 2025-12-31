# DEPENDENCY_HEALTH.md

## Outdated Packages
- `fastapi`: 0.127.0 -> 0.128.0
- `nvidia-cuda-*`: Various minor version updates available.
- `pip`: 24.0 -> 25.3
- `psutil`: 7.2.0 -> 7.2.1

## Missing Dependencies (in system path)
- `pylint`, `bandit`, `radon`, `torch`: Found in `venv` but missing from system-wide Python. This is correct for isolated development but requires careful `venv` usage in scripts.

## Conflict Resolution
- `pip check`: No broken dependencies or version conflicts detected in the current `venv`.
- `deriv-api`: Correctly installed from custom fork (`python-deriv-api`).

## Security Assessment (Bandit Snapshot)
- No critical CVEs reported by `pip check`.
- (Manual Note): WebSocket interactions should ensure TLS 1.3 for secure Deriv communication.

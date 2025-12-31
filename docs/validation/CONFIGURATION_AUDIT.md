# CONFIGURATION_AUDIT.md

## Env Variable Assessment
| Variable | Value/Status | Safety |
|----------|--------------|--------|
| `ENVIRONMENT` | `production` | ⚠️ HIGH RISK (for testing) |
| `TRADING__SYMBOL` | `R_100` | ✅ Normal |
| `DERIV_API_TOKEN` | [REDACTED] | ⚠️ ACTIVE REAL TOKEN DETECTED |
| `EXECUTION_SAFETY__MAX_DAILY_LOSS` | 100.0 | ✅ Safe |
| `EXECUTION_SAFETY__MAX_STAKE_PER_TRADE` | 10.0 | ✅ Safe |

## Config Consistency
- `config/settings.py` correctly maps to `@property` style access for Pydantic v2.
- No hardcoded API secrets found in source code (correctly moved to `.env`).

## Dangerous Defaults
- `ENVIRONMENT=production`: Should be changed to `staging` or `development` during validation to prevent accidental real trades, although `--test` and `--shadow-only` flags mitigate this.
- `EXECUTION_SAFETY__KILL_SWITCH_ENABLED=false`: Safe for testing, but should be verified before real production deployment.

## Undocumented Options
- `CALIBRATION__ERROR_THRESHOLD`: Usage in `CalibrationMonitor` is clear, but not explicitly documented in the top-level README.
- `SHADOW_TRADE__DURATION_MINUTES`: Defaults to 1; may need tuning for complex contract types.

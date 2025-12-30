# BEHAVIORAL_VALIDATION.md

## Safety Mechanism Verification

| Mechanism | Precedence | Status | Note |
|-----------|------------|--------|------|
| Circuit Breaker | 1 | ✅ functional | Reset confirmed |
| Daily Loss Limit | 2 | ✅ functional | Blocked at -1000.0 |
| Regime Veto | 5 | ✅ functional | Blocked at 0.8 error |

## Edge Case Assessment
- **Extreme Volatility**: Correctly blocked by Regime Veto.
- **System Drift**: Calibration veto observed as active in policy.
- **Recovery**: Circuit breaker reset behaves as expected.

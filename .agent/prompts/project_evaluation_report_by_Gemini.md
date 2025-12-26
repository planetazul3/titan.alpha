# X.TITAN PROJECT EVALUATION REPORT - CONSOLIDATED AI DEVELOPMENT GUIDE

**Project:** x.titan (DerivOmniModel Trading System)  
**Evaluation Date:** 2025-12-25  
**Purpose:** Comprehensive reference document for AI development assistants  

> [!IMPORTANT]
> This document consolidates all evaluation findings into a single, actionable reference. Issues are categorized by severity and domain. Each issue includes file locations, root causes, and detailed implementation steps.

---

## TABLE OF CONTENTS

1. [Master Issue Registry](#master-issue-registry)
2. [Critical Issues](#critical-issues)
3. [High-Priority Issues](#high-priority-issues)
4. [Medium-Priority Issues](#medium-priority-issues)
5. [Low-Priority Improvements](#low-priority-improvements)
6. [Architecture Validation Matrix](#architecture-validation-matrix)
7. [Technical Risk Assessment](#technical-risk-assessment)
8. [Consolidated Action Plan](#consolidated-action-plan)

---

## MASTER ISSUE REGISTRY

| ID | Status | Severity | Domain | Issue | Files Affected |
|:---|:-------|:---------|:-------|:------|:---------------|
| C01 | ✅ Fixed | Critical | Performance | Event Loop Blocking During Inference | `scripts/live.py` |
| C02 | ✅ Fixed | Critical | Data | Massive Data Leakage via Random Split | `scripts/train.py`, `scripts/evaluate.py` |
| C03 | ✅ Fixed | Critical | Data | Warmup Artifact (Indicator Blindness) | `data/dataset.py`, `data/features.py` |
| C04 | ✅ Fixed | Critical | Data | Feature Flickering (Non-Stationary Preprocessing) | `data/processor.py`, `scripts/live.py` |
| C05 | ✅ Fixed | Critical | Execution | Broken Barrier Contract Execution | `data/ingestion/client.py`, `execution/executor.py` |
| C06 | ✅ Fixed | Critical | Safety | Phantom Stake Safety Bypass | `scripts/live.py`, `execution/safety.py` |
| C07 | ✅ Fixed | Critical | Safety | Double Execution Risk on Network Timeout | `execution/safety.py` |
| C08 | ✅ Fixed | Critical | Resources | Zombie Subscription Memory Leak | `data/ingestion/client.py` |
| C09 | ✅ Fixed | Critical | Operations | Static Kill Switch (Unreachable Emergency Stop) | `scripts/live.py`, `execution/safety.py` |
| C10 | ✅ Fixed | Critical | Training | Optimizer State Amnesia | `scripts/online_train.py` |
| H01 | ✅ Fixed | High | Execution | Disconnected Position Sizing Logic | `execution/executor.py`, `execution/position_sizer.py` |
| H02 | ✅ Fixed | High | Data | NaN Propagation in Volatility Features | `data/processor.py` |
| H03 | ✅ Fixed | High | Data | Missing Attention Masking for Padded Sequences | `data/dataset.py`, `models/core.py`, `models/tft.py` |
| H04 | ✅ Fixed | High | Safety | Ephemeral Risk State (Memory Amnesia) | `execution/adaptive_risk.py` |
| H05 | ✅ Fixed | High | Safety | Non-Persistent Rate Limiting | `execution/safety.py`, `execution/safety_store.py` |
| H06 | ✅ Fixed | High | Data | Zero-Volatility Normalization Crash | `data/normalizers.py`, `data/dataset.py` |
| H07 | ✅ Fixed | High | Latency | Stale Data Processing in Live Loop | `scripts/live.py` |
| H08 | ✅ Fixed | High | Latency | Blocking Database I/O in Async Event Loop | `execution/sqlite_shadow_store.py` |
| H09 | ✅ Fixed | High | Latency | Synchronous SQLite Commits in Hot Path | `execution/safety_store.py` |
| H10 | ✅ Fixed | High | Security | Credential Leak in Logs | `config/logging_config.py` |
| H11 | ✅ Fixed | High | Data | Startup Data Gap | `scripts/live.py` |
| H12 | ✅ Fixed | High | Model | Broken Attention Implementation | `models/attention.py` |
| H13 | ✅ Fixed | High | Training | Loss Function Imbalance | `training/losses.py` |
| M01 | ✅ Fixed | Medium | Logic | Inference Cooldown Logic Flaw | `scripts/live.py` |
| M02 | ✅ Fixed | Medium | Safety | Disabled Slippage Protection | `data/ingestion/client.py` |
| M03 | ✅ Fixed | Medium | Data | Inaccurate Outcome Resolution for Barrier Trades | `execution/outcome_resolver.py` |
| M04 | ✅ Fixed | Medium | Scaling | Unscalable Dashboard Queries | `api/dashboard_server.py` |
| M05 | ✅ Fixed | Medium | Model | Improper Action Scaling in RL | `models/policy.py` |
| M06 | ✅ Fixed | Medium | Model | Underutilized TFT Capabilities | `models/tft.py`, `models/core.py` |
| M07 | ✅ Fixed | Medium | Model | Spatial Expert Pooling Flaw | `models/spatial.py` |
| M08 | ✅ Fixed | Medium | Model | Hardcoded Model Input Dimensions | `models/temporal.py` |
| M09 | ✅ Fixed | Medium | Config | Hardcoded Normalization Scaling | `data/processor.py` |
| M10 | ✅ Fixed | Medium | Config | Hardcoded Payout Ratios | `api/dashboard_server.py`, `observability/shadow_metrics.py` |
| M11 | ✅ Fixed | Medium | Resources | Dataset Memory Explosion | `data/dataset.py` |
| M12 | ✅ Fixed | Medium | Operations | Stale Model Weights in Live Inference | `scripts/live.py` |
| M13 | ✅ Fixed | Medium | Operations | Lack of Disk Space Management | `execution/sqlite_shadow_store.py` |
| M14 | ✅ Fixed | Medium | Operations | Single Point of Failure (DerivClient) | `data/ingestion/client.py` |
| M15 | ✅ Fixed | Medium | Logic | Indiscriminate Retry Logic (Bug Masking) | `execution/safety.py` |
| M16 | ✅ Fixed | Medium | Financial | Floating Point Precision Drift | `execution/real_trade_tracker.py` |
| M17 | ✅ Fixed | Medium | Security | Unsecured Dashboard API | `api/dashboard_server.py` |
| M18 | ✅ Fixed | Medium | Security | Insecure Dependency Retrieval | `scripts/install-talib.sh` |
| L01 | ✅ Fixed | Low | Config | Hardcoded "R_100" Dependency | Various |
| L02 | ✅ Fixed | Low | Code | Dead Code (AutoFeatures) | `data/auto_features.py` |
| L03 | ✅ Fixed | Low | Code | Unused Ensemble Capability | `models/ensemble.py` |
| L04 | ✅ Fixed | Low | Architecture | Missing Event-Driven Backtester | `execution/backtest.py` |
| L05 | ✅ Fixed | Low | Architecture | Non-Atomic State Management (Split Brain Risk) | `trading_state.db` |

---

## CRITICAL ISSUES

### C01: Event Loop Blocking During Inference
**File:** `scripts/live.py`  
**Function:** `run_inference`

**Problem:** CPU/GPU-bound operations (NumPy calculations, PyTorch forward passes) run synchronously on the asyncio event loop, blocking all I/O for 100-500ms+.

**Impact:** WebSocket timeouts, missed keep-alive pings, jitter in heartbeat logging, inaccurate trade timing.

**Solution:**
```python
# STEP 1: Extract synchronous logic into separate function
def _inference_task(feature_builder, model, buffer_snapshot, device):
    features = feature_builder.build(buffer_snapshot)
    tensor = torch.tensor(features, device=device)
    with torch.no_grad():
        return model(tensor)

# STEP 2: Offload to thread pool
async def run_inference(...):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _inference_task, feature_builder, model, buffer.snapshot(), device)
    # ... process result
```

---

### C02: Massive Data Leakage via Random Split
**Files:** `scripts/train.py`, `scripts/evaluate.py`

**Problem:** Using `torch.utils.data.random_split` on sliding window time-series data causes 99% overlap between train/val samples.

**Impact:** Near-perfect validation metrics (95%+) that fail completely in production (0% generalization).

**Solution:**
```python
# Replace random_split with temporal split
total_len = len(dataset)
split_idx = int(total_len * 0.8)
purge_gap = sequence_length  # Prevent overlap at boundary

train_indices = list(range(0, split_idx - purge_gap))
val_indices = list(range(split_idx, total_len))

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
```

---

### ✅ C03: Warmup Artifact (Indicator Blindness)
**Files:** `data/dataset.py`, `data/features.py`

**Problem:** Technical indicators are calculated on isolated window slices. The first N points of every sample are invalid (NaN/default values) because indicators need warmup periods.

**Impact:** ~28% of input data is garbage. Model learns that sequence beginnings always have "flat" indicators.

**Solution:**
```python
# STEP 1: In DerivDataset, slice extended window
lookback = 50  # For indicator warmup
extended_slice = data[idx - sequence_length - lookback : idx]

# STEP 2: Calculate indicators on extended window
features = preprocessor.process(extended_slice)

# STEP 3: Trim to sequence_length AFTER calculation
final_features = features[-sequence_length:]
```

---

### ✅ C04: Feature Flickering (Non-Stationary Preprocessing)
**Files:** `data/processor.py`, `scripts/live.py`

**Problem:** Normalization statistics (mean, std) are recalculated for every sliding window. Historical values change on each inference cycle.

**Impact:** Neural network sees unstable, morphing history that introduces artificial noise.

**Solution:**
```python
# Option 1: Incremental statistics (Welford's algorithm)
class IncrementalNormalizer:
    def __init__(self):
        self.mean = 0.0
        self.M2 = 0.0
        self.count = 0
    
    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.M2 += delta * (x - self.mean)
    
    @property
    def std(self):
        return np.sqrt(self.M2 / self.count) if self.count > 1 else 1.0

# Option 2: Fixed lookback window (e.g., 24h rolling stats, updated hourly)
```

---

### C05: Broken Barrier Contract Execution
**Files:** `data/ingestion/client.py`, `execution/executor.py`

**Problem:** The `buy` method omits the `barrier` parameter required for TOUCH, NO_TOUCH, and RANGE contracts.

**Impact:** API errors or execution with random barriers that don't match strategy intent.

**Solution:**
```python
# In data/ingestion/client.py
async def buy(self, amount: float, contract_type: str, duration: int, 
              symbol: str, barrier: float | None = None, barrier2: float | None = None):
    proposal_req = {
        "proposal": 1,
        "amount": amount,
        "contract_type": contract_type,
        "duration": duration,
        "symbol": symbol,
    }
    if barrier is not None:
        proposal_req["barrier"] = str(barrier)
    if barrier2 is not None:
        proposal_req["barrier2"] = str(barrier2)
    # ... rest of method

# In execution/executor.py
barrier = current_price * (1 + self.config.touch_barrier_percent)
await self.client.buy(..., barrier=barrier)
```

---

### C06: Phantom Stake Safety Bypass
**Files:** `scripts/live.py`, `execution/safety.py`

**Problem:** `SafeTradeExecutor` is instantiated without a `stake_resolver`. It calculates stake as 0.0, bypassing the max stake safety check.

**Impact:** Max Stake safety guard is completely bypassed. Trades can exceed safety limits.

**Solution:**
```python
# In scripts/live.py
def stake_resolver(signal: TradeSignal) -> float:
    return sizer.suggest_stake_for_signal(signal)

executor = SafeTradeExecutor(
    raw_executor, 
    safety_config, 
    state_file=safety_state_file,
    stake_resolver=stake_resolver  # ADD THIS
)
```

---

### C07: Double Execution Risk on Network Timeout
**File:** `execution/safety.py`

**Problem:** Retry logic doesn't verify if the original request executed before retrying.

**Impact:** Duplicate positions double the risk exposure.

**Solution:**
```python
async def _execute_with_retry(self, signal: TradeSignal) -> ExecutionResult:
    for attempt in range(max_retries):
        try:
            return await self._inner_executor.execute(signal)
        except (ConnectionError, TimeoutError) as e:
            # BEFORE RETRY: Check if trade already executed
            open_contracts = await self.client.get_open_contracts()
            recent_match = self._find_matching_contract(open_contracts, signal, last_n_seconds=30)
            if recent_match:
                logger.warning("Found existing contract from timed-out request, adopting")
                return ExecutionResult(contract_id=recent_match.contract_id, ...)
            # ... proceed with retry
```

---

### C08: Zombie Subscription Memory Leak
**File:** `data/ingestion/client.py`

**Problem:** On reconnection, old RxPY subscriptions are never disposed. They accumulate and push duplicate data.

**Impact:** Memory leak + duplicate data corruption in buffer.

**Solution:**
```python
async def stream_ticks(self, symbol: str) -> AsyncIterator[Tick]:
    while True:
        source = await self.api.subscribe({"ticks": symbol})
        subscription = None
        try:
            subscription = source.subscribe(lambda x: queue.put_nowait(x))
            while True:
                # ... process ticks
                if is_stale:
                    break  # Will trigger finally block
        finally:
            if subscription:
                subscription.dispose()  # CRITICAL: Clean up before re-subscribe
```

---

### C09: Static Kill Switch (Unreachable Emergency Stop)
**Files:** `scripts/live.py`, `execution/safety.py`

**Problem:** Kill switch is in-memory only. Dashboard runs in separate process and cannot trigger it.

**Impact:** No remote emergency stop capability. Operator must SSH and kill process.

**Solution:**
```python
# STEP 1: Add to SQLiteSafetyStateStore
def get_kill_switch_state(self) -> bool:
    return self.get_metric("kill_switch_enabled", default=False)

def set_kill_switch_state(self, enabled: bool) -> None:
    self.set_metric("kill_switch_enabled", enabled)

# STEP 2: In SafeTradeExecutor.execute, check DB state
if self._kill_switch or await self._check_db_kill_switch():
    return ExecutionResult(blocked=True, reason="Kill switch active")

# STEP 3: Add dashboard endpoint to toggle kill switch
@app.post("/api/kill-switch")
async def toggle_kill_switch(enabled: bool, api_key: str = Depends(verify_api_key)):
    safety_store.set_kill_switch_state(enabled)
```

---

### C10: Optimizer State Amnesia
**File:** `scripts/online_train.py`

**Problem:** Optimizer is re-initialized from scratch on every run, losing momentum/gradient history.

**Impact:** Loss spikes and unstable adaptation that can degrade the model.

**Solution:**
```python
# Load optimizer state if available
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model_state_dict"])

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
if "optimizer_state_dict" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# Save optimizer state at end
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),  # ADD THIS
}, output_path)
```

---

## HIGH-PRIORITY ISSUES

### ✅ H01: Disconnected Position Sizing Logic
**Files:** `execution/executor.py`, `execution/position_sizer.py`

**Problem:** `DerivTradeExecutor` doesn't pass account balance to `PositionSizer`. Kelly Criterion falls back to fixed stake.

**Solution:** Fetch balance before sizing: `balance = await self.client.get_balance()` then pass to `suggest_stake_for_signal(signal, account_balance=balance)`.

---

### ✅ H02: NaN Propagation in Volatility Features
**File:** `data/processor.py`

**Problem:** `np.clip` is called before `np.nan_to_num`, propagating NaNs. Division by zero when `price_level == 0`.

**Solution:**
```python
# Move nan_to_num BEFORE clip
metrics = np.nan_to_num(metrics, nan=0.0, posinf=1.0, neginf=0.0)
metrics = np.clip(metrics, 0.0, 1.0)

# Check for zero division
if price_level == 0:
    return np.zeros(self.output_dim)
```

---

### ✅ H03: Missing Attention Masking for Padded Sequences
**Files:** `data/dataset.py`, `models/core.py`, `models/tft.py`

**Problem:** Padded tensors are passed to attention layers without masks. Model treats padding as valid data.

**Solution:** Generate mask in `FeatureBuilder.build`, pass through `DerivOmniModel.forward`, apply in attention layers (set masked positions to `-inf` before softmax).

---

### ✅ H04: Ephemeral Risk State (Memory Amnesia)
**File:** `execution/adaptive_risk.py`

**Problem:** `AdaptiveRiskManager` state resets on restart. A bot in "Conservative" mode becomes "Normal" after restart.

**Solution:** Add `save_state`/`load_state` methods that persist `PerformanceTracker._returns` to SQLite. Call on startup and after each trade.

---

### ✅ H05: Non-Persistent Rate Limiting
**Files:** `execution/safety.py`, `execution/safety_store.py`

**Problem:** Rate limit history lives in memory. Crash-restart loop can bypass limits.

**Solution:** Persist trade timestamps to SQLite. Load on startup. Or implement 60-second trading pause after restart.

---

### ✅ H06: Zero-Volatility Normalization Crash
**Files:** `data/normalizers.py`, `data/dataset.py`

**Problem:** `log_returns` crashes on non-positive values. Zero padding causes `log(0)`.

**Solution:** Use edge padding (copy first valid price) instead of zero padding. Add guard: `if np.any(values <= 0): values = np.maximum(values, 1e-8)`.

---

### ✅ H07: Stale Data Processing in Live Loop
**File:** `scripts/live.py`

**Problem:** Sequential processing accumulates lag. After slow inference, next message is already stale.

**Solution:** Add staleness check:
```python
latency = (datetime.now(timezone.utc) - candle_event.timestamp).total_seconds()
if latency > 2.0:
    logger.warning(f"Skipping stale candle ({latency:.1f}s old)")
    continue
```

---

### ✅ H08: Blocking Database I/O in Async Event Loop
**File:** `execution/sqlite_shadow_store.py`

**Problem:** Standard `sqlite3` calls block the asyncio loop.
**Impact:** Severe latency spikes during heavy read/write (e.g., shadow trade resolution).
**Fix:** Wrap all DB operations in `loop.run_in_executor`.

---

### ✅ H09: Synchronous SQLite Commits in Hot Path
**File:** `execution/safety_store.py`

**Problem:** Default synchronous SQLite commits can block the event loop, especially in `safety_store.py`'s hot path.
**Impact:** Increased latency and potential for event loop starvation during critical safety checks.
**Fix:** Use `PRAGMA synchronous = NORMAL` in WAL mode for faster, non-blocking commits.

---

### H10: Credential Leak in Logs
**File:** `config/logging_config.py`

**Problem:** Debug logging exposes API tokens.

**Solution:** Add log filter:
```python
class TokenSanitizer(logging.Filter):
    def filter(self, record):
        record.msg = re.sub(r'"authorize":\s*"[^"]+"', '"authorize": "***"', str(record.msg))
        return True
```

---

### H11: Startup Data Gap
**File:** `scripts/live.py`

**Problem:** Gap between historical fetch and stream subscription loses ticks.

**Solution:** Subscribe first (buffer to temp queue), fetch history, then stitch (discard duplicates).

---

### H12: Broken Attention Implementation
**File:** `models/attention.py`

**Problem:** `self.U` is defined but never used in forward pass.

**Solution:** Either remove unused parameter or fix the formula: `attn_score = self.V(torch.tanh(self.W(query) + self.U(encoder_outputs)))`.

---

### H13: Loss Function Imbalance
**File:** `training/losses.py`

**Problem:** Reconstruction loss (~0.01) with 0.1 weight is negligible vs classification loss (~0.7). Autoencoder barely learns.

**Solution:** Increase `reconstruction_weight` to 10.0-50.0, or implement dynamic loss weighting (GradNorm).

---

## MEDIUM-PRIORITY ISSUES

### M01: Inference Cooldown Logic Flaw
**File:** `scripts/live.py`

**Problem:** Wall-clock cooldown of 60s for 60s candles skips valid inference when processing takes any time.

**Solution:** Track last processed candle epoch instead of wall-clock time.

---

### M02: Disabled Slippage Protection
**File:** `data/ingestion/client.py`

**Problem:** `price = amount + 100` effectively disables slippage protection.

**Solution:** Calculate `limit_price = proposal_price * (1 + tolerance)` where tolerance defaults to 0.02 (2%).

---

### M03: Inaccurate Outcome Resolution for Barrier Trades
**File:** `execution/outcome_resolver.py`

**Problem:** Barriers are recalculated from config instead of using actual execution values.

**Solution:** Store `barrier_high` and `barrier_low` in `ShadowTradeRecord`. Use stored values in resolution.

---

### M04-M17: See Issue Registry for Implementation Details
Detailed solutions available in original sections. Key themes:
- **Config centralization:** Payout ratios, scaling factors, input dimensions → `config/settings.py`
- **Resource management:** Lazy loading, connection pooling, disk retention policies
- **Model improvements:** TFT context features, proper RL action scaling, sequence-aware spatial pooling

---

## LOW-PRIORITY IMPROVEMENTS

| ID | Issue | Recommendation |
|:---|:------|:---------------|
| L01 | Hardcoded "R_100" | Audit for string literals, use `settings.trading.symbol` |
| L02 | Dead Code | Delete `data/auto_features.py` or integrate into pipeline |
| L03 | Unused Ensemble | Integrate `ModelRegistry` + `EnsemblePredictor` in live.py |
| L04 | Missing Backtester | Create `execution/backtest.py` with `MockDerivClient` |
| L05 | Split Brain Risk | Consolidate all SQLite tables into single `trading_state.db` |

---

## ARCHITECTURE VALIDATION MATRIX

| Requirement | Status | Notes |
|:------------|:-------|:------|
| Modular Architecture | ✅ Compliant | Clear separation: `models/`, `execution/`, `data/` |
| Safety Mechanisms | ⚠️ Partial | "Swiss Cheese" model present, but persistence gaps exist |
| Shadow Mode | ✅ Compliant | Full context capture in SQLite |
| Inference Latency | ❌ At Risk | Event loop blocking under load |
| Data Integrity | ⚠️ Partial | Feature pipeline has warmup/flickering issues |
| Model Architecture | ⚠️ Partial | Attention masking missing, unused capabilities |
| Online Learning | ⚠️ Partial | No hot-reload, manual trigger required |
| Financial Accuracy | ⚠️ Partial | Float precision, position sizing disconnected |

---

## TECHNICAL RISK ASSESSMENT

| Risk Category | Severity | Primary Issues |
|:--------------|:---------|:---------------|
| **Operational** | 🔴 High | Event loop blocking, subscription leaks, no remote kill switch |
| **Financial** | 🔴 High | Stake bypass, double execution, disconnected position sizing |
| **Data Integrity** | 🔴 High | Data leakage, warmup artifacts, feature flickering |
| **Resource Exhaustion** | 🟠 Medium | Memory leaks, unbounded DB growth, dataset OOM |
| **Model Stability** | 🟠 Medium | NaN propagation, loss imbalance, broken attention |
| **Security** | 🟠 Medium | No API auth, credential logging, HTTP dependencies |

---

## CONSOLIDATED ACTION PLAN

> [!CAUTION]
> Address issues in priority order. Critical issues should be resolved before production deployment.

### Phase 1: Critical Stability (Immediate)
1. ✅ **C01** - Offload inference to thread pool
2. ✅ **C02** - Implement temporal train/val split
3. ✅ **C08** - Fix subscription dispose logic
4. ✅ **C06** - Inject stake resolver

### Phase 2: Critical Execution (Before Live Trading)
5. ✅ **C05** - Add barrier parameters to buy method
6. ✅ **C07** - Add idempotency checks for retries
7. ✅ **C09** - Implement DB-backed kill switch
8. ✅ **C10** - Persist optimizer state

### Phase 3: High Priority (First Week)
9. ✅ **H01** - Connect position sizer to balance
10. ✅ **H02** - Fix NaN handling order
11. ✅ **H04 + H05** - Persist adaptive risk + rate limit state
12. ✅ **H07** - Add staleness checks in live loop
13. ✅ **H08 + H09** - Async DB operations

### Phase 4: Data Pipeline (First Month)
14. ✅ **C03** - Implement lookback buffering
15. ✅ **C04** - Stabilize normalization
16. ✅ **H03** - Add attention masking
17. **H11** - Subscribe-then-fetch startup

### Phase 5: Ongoing Improvements
18. **M01-M17** - Configuration centralization, security hardening
19. **H12 + H13** - Model architecture fixes
20. **L01-L05** - Code cleanup and architecture enhancements

---

## QUICK REFERENCE: FILE → ISSUES MAPPING

```
scripts/live.py          → C01, C06, H07, H11, M01, M12
data/ingestion/client.py → C05, C08, M02
execution/safety.py      → C06, C07, H05, M15
execution/executor.py    → C05, H01
data/dataset.py          → C02, C03, H03, H06, M11
data/processor.py        → C04, H02, M09
data/features.py         → C03, H03
models/attention.py      → H12
training/losses.py       → H13
scripts/train.py         → C02
scripts/online_train.py  → C10
execution/adaptive_risk.py → H04
execution/safety_store.py  → H05, H09
execution/sqlite_shadow_store.py → H08
config/settings.py       → M08, M09, M10 (centralization target)
api/dashboard_server.py  → C09, M04, M17
```

---

**Document Version:** 1.0.0-consolidated  
**Last Updated:** 2025-12-26  
**For AI Development Assistants:** Use this document as the authoritative reference for issue identification and remediation. Each issue ID is unique and can be referenced in commits/PRs.
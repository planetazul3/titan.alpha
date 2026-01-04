
import asyncio
import time
import logging
from execution.policy import ExecutionPolicy, VetoPrecedence
from execution.safety import SafeTradeExecutor, ExecutionSafetyConfig
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_async_policy():
    print("\n--- Testing Async Policy Integration ---")
    
    # 1. Initialize Policy
    policy = ExecutionPolicy()
    
    # 2. Register a slow async veto (simulating DB latency)
    async def slow_db_check():
        await asyncio.sleep(0.1) # Simulate 100ms latency
        return False # No veto
        
    policy.register_veto(
        level=VetoPrecedence.RATE_LIMIT,
        check_fn=lambda: False, # Sync fallback
        async_check_fn=slow_db_check,
        reason="Slow DB Check"
    )
    
    # 3. Register a blocking sync veto (to show difference)
    def blocking_check():
        time.sleep(0.1) # Blocks loop
        return False
        
    policy.register_veto(
        level=VetoPrecedence.CONFIDENCE,
        check_fn=blocking_check,
        reason="Blocking Check"
    )
    
    print("Registered vetoes.")
    
    # 4. Measure Sync Check (Should block 100ms + 100ms if we used the blocking version of async check)
    # But wait, we only registered async_check_fn for the first one. 
    # For sync check_vetoes, it uses check_fn.
    
    t0 = time.time()
    policy.check_vetoes()
    t_sync = time.time() - t0
    print(f"Sync check_vetoes took: {t_sync:.4f}s (Expected ~0.1s due to blocking_check only)")
    
    # 5. Measure Async Check
    # Should run blocking_check (0.1s) AND slow_db_check (0.1s)
    # If slow_db_check is truly async, it shouldn't block the loop, but here verify it works.
    
    t0 = time.time()
    await policy.async_check_vetoes()
    t_async = time.time() - t0
    print(f"Async check_vetoes took: {t_async:.4f}s")
    
    # 6. Verify Correctness
    # Register a VETOING async check
    async def fast_veto():
        return True
        
    policy.register_veto(
        level=VetoPrecedence.KILL_SWITCH,
        check_fn=lambda: False,
        async_check_fn=fast_veto,
        reason="Async Kill Switch"
    )
    
    veto = await policy.async_check_vetoes()
    if veto and veto.reason == "Async Kill Switch":
        print("SUCCESS: Async veto triggered correctly.")
    else:
        print(f"FAILURE: Async veto did not trigger or wrong reason: {veto}")

    # 7. Mock DecisionEngine usage
    print("\n--- Simulating Decision Engine Loop ---")
    start_time = time.time()
    
    # Simulate 5 concurrent decisions
    tasks = [policy.async_check_vetoes() for _ in range(5)]
    await asyncio.gather(*tasks)
    
    duration = time.time() - start_time
    print(f"5 Concurrent checks took: {duration:.4f}s")
    
if __name__ == "__main__":
    asyncio.run(test_async_policy())

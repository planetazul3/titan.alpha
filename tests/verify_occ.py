
import asyncio
import logging
import threading
import sqlite3
import time
from pathlib import Path
from execution.shadow_store import ShadowTradeRecord
from execution.sqlite_shadow_store import SQLiteShadowStore, OptimisticLockError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ConcurrencyTest")

DB_PATH = Path("data_cache/test_concurrency.db")
if DB_PATH.exists():
    DB_PATH.unlink()

def run_test():
    print(f"--- Testing Optimistic Concurrency Control (OCC) ---")
    store = SQLiteShadowStore(DB_PATH)
    
    # 1. Create a trade
    record = ShadowTradeRecord.create(
        contract_type="RISE_FALL",
        direction="CALL",
        probability=0.8,
        entry_price=100.0,
        reconstruction_error=0.1,
        regime_state="TRUSTED"
    )
    store.append(record)
    print(f"[Init] Created trade {record.trade_id} (Version 0)")
    
    # 2. Simulate Race Condition
    # We will try to update the SAME record object from two threads implicitly
    # Since ShadowTradeRecord is immutable, we use the same instance which holds version 0
    
    # Thread 1: Resolves as WIN
    def update_win():
        try:
            print("[Thread 1] Attempting update (WIN)...")
            store.update_outcome(record, outcome=True, exit_price=101.0)
            print("[Thread 1] Update SUCCESS")
        except OptimisticLockError:
            print("[Thread 1] Update FAILED (Optimistic Lock)")
        except Exception as e:
            print(f"[Thread 1] Error: {e}")

    # Thread 2: Resolves as LOSS
    def update_loss():
        try:
            print("[Thread 2] Attempting update (LOSS)...")
            time.sleep(0.1) # Slight delay to force sequence T1 -> T2
            store.update_outcome(record, outcome=False, exit_price=99.0)
            print("[Thread 2] Update SUCCESS")
        except OptimisticLockError:
            print("[Thread 2] Update FAILED (Optimistic Lock) - AS EXPECTED")
        except Exception as e:
            print(f"[Thread 2] Error: {e}")
            
    # Run sequentially first to verify logic, then we could try parallel but python threads are GIL limited
    # Actually, proper test is:
    # 1. Update (v0 -> v1)
    # 2. Try update again with v0 record -> Should Fail
    
    print("\n[Sequence Test]")
    
    # First update
    success = store.update_outcome(record, outcome=True, exit_price=101.0)
    print(f"First Update: {'Success' if success else 'Fail'}")
    
    # Fetch updated record to check version
    updated_record = store.get_by_id(record.trade_id)
    print(f"DB Version after 1st update: {updated_record.version_number}")
    
    # Second update using OLD record (v0)
    print("Attempting second update with STALE record (v0)...")
    try:
        store.update_outcome(record, outcome=False, exit_price=99.0)
        print("CRITICAL FAILURE: Stale update succeeded!")
    except OptimisticLockError:
        print("SUCCESS: Stale update was rejected with OptimisticLockError.")
        
    print("\n--- Test Complete ---")

if __name__ == "__main__":
    run_test()

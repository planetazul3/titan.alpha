import pytest
import asyncio
import sqlite3
from pathlib import Path
from execution.idempotency_store import SQLiteIdempotencyStore

@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_idempotency.db"

@pytest.fixture
def store(db_path):
    return SQLiteIdempotencyStore(db_path)

@pytest.mark.asyncio
async def test_atomic_reservation_success(store):
    signal_id = "sig_1"
    is_new, contract_id = await store.check_and_reserve_async(signal_id, "AAPL")
    
    assert is_new is True
    assert contract_id is None
    
    # Check that it is now "PENDING"
    contract_id = await store.get_contract_id_async(signal_id)
    assert contract_id == "PENDING"

@pytest.mark.asyncio
async def test_atomic_reservation_race_condition(store):
    signal_id = "sig_race"
    
    # First reservation succeeds
    is_new, cid = await store.check_and_reserve_async(signal_id, "AAPL")
    assert is_new
    
    # Second reservation fails and returns "PENDING"
    is_new_2, cid_2 = await store.check_and_reserve_async(signal_id, "AAPL")
    assert not is_new_2
    assert cid_2 == "PENDING"

@pytest.mark.asyncio
async def test_update_contract_id(store):
    signal_id = "sig_update"
    await store.check_and_reserve_async(signal_id, "AAPL")
    
    await store.update_contract_id_async(signal_id, "REAL_CONTRACT_ID")
    
    contract_id = await store.get_contract_id_async(signal_id)
    assert contract_id == "REAL_CONTRACT_ID"
    
@pytest.mark.asyncio
async def test_delete_record(store):
    signal_id = "sig_delete"
    await store.check_and_reserve_async(signal_id, "AAPL")
    
    await store.delete_record_async(signal_id)
    # exists() is now internal _exists(), but better to rely on public API? 
    # Public API doesn't have exist_async, but check_and_reserve uses insertion.
    # We can check internal state for test verification or add exist_async.
    # For now, accessing internal _exists for assertion is acceptable in unit test?
    # Or add exist_async. User said "all must be compliant with async".
    # Let's rely on get_contract_id_async returning None.
    
    cid = await store.get_contract_id_async(signal_id)
    assert cid is None

import os
import sys
import numpy as np
import pytest
from datetime import datetime
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.getcwd())

from config.settings import load_settings, Settings
from execution.signals import TradeSignal, SIGNAL_TYPES
from execution.common.types import ExecutionRequest
from execution.executor_adapter import SignalAdapter
from data.common.schema import CandleInputSchema
import pandas as pd
import pandera as pa

def test_verify_c001_signal_adapter():
    print("\n--- Verifying C-001 (Signal Adapter) ---")
    settings = load_settings()
    adapter = SignalAdapter(settings)
    
    signal = TradeSignal(
        signal_type=SIGNAL_TYPES.REAL_TRADE,
        contract_type="RISE_FALL",
        direction="CALL",
        probability=0.85,
        timestamp=datetime.now(),
        symbol="R_100",
        signal_id="test_signal"
    )
    
    req = adapter.to_execution_request(signal)
    
    assert isinstance(req, ExecutionRequest)
    assert req.contract_type == "CALL"
    assert req.stake > 0
    assert req.duration > 0
    print("✅ SignalAdapter converted TradeSignal to ExecutionRequest successfully.")

def test_verify_c002_schema_float32():
    print("\n--- Verifying C-002 (Schema float32) ---")
    # create float32 dataframe
    data = {
        "open": np.array([100.0, 101.0], dtype=np.float32),
        "high": np.array([102.0, 103.0], dtype=np.float32),
        "low": np.array([99.0, 100.0], dtype=np.float32),
        "close": np.array([101.0, 102.0], dtype=np.float32),
        "volume": np.array([10.0, 10.0], dtype=np.float32),
        "timestamp": np.array([1600000000.0, 1600000060.0], dtype=np.float64) # Timestamp usually kept high precision
    }
    df = pd.DataFrame(data)
    
    try:
        CandleInputSchema.validate(df)
        print("✅ Schema accepted float32 data.")
    except Exception as e:
        print(f"❌ Schema validation failed: {e}")
        raise

def test_verify_i002_test_isolation():
    print("\n--- Verifying I-002 (Test Isolation) ---")
    # Mock environment
    os.environ["PYTEST_CURRENT_TEST"] = "True"
    
    try:
        settings = load_settings()
        print(f"Loaded environment: {settings.environment}")
        assert settings.is_test_mode
        # Check if derived from .env.test (DERIV_API_TOKEN should be the test one)
        token = settings.deriv_api_token.get_secret_value()
        if token == "test_token_do_not_use_in_prod":
            print("✅ Loaded .env.test correctly.")
        else:
            print(f"❌ Loaded token: {token} (Expected test_token...)")
    finally:
        del os.environ["PYTEST_CURRENT_TEST"]

def test_verify_c004_decision_engine():
    print("\n--- Verifying C-004 (Decision Engine Refactor) ---")
    settings = load_settings()
    # Mock dependencies
    storage = MagicMock()
    
    from execution.decision.core import DecisionEngine
    engine = DecisionEngine(settings, shadow_store=storage)
    
    # Check if sub-components are initialized
    assert hasattr(engine, "metrics")
    assert hasattr(engine, "safety_sync")
    assert hasattr(engine, "processor")
    
    print("✅ DecisionEngine initialized with micro-modules.")

if __name__ == "__main__":
    try:
        test_verify_c001_signal_adapter()
        test_verify_c002_schema_float32()
        test_verify_i002_test_isolation()
        test_verify_c004_decision_engine()
        print("\n✨ ALL VERIFICATIONS PASSED")
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        sys.exit(1)

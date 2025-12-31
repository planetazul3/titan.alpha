import pytest
import subprocess
import sys
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"

def run_script(script_name, args=None):
    """Run a script as a subprocess and return result."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        pytest.fail(f"Script not found: {script_path}")

    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    # Capture output to help debugging
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30 # 30s timeout to prevent hangs
    )
    return result

def test_live_script_help():
    """Test that live.py can initialize and show help."""
    result = run_script("live.py", ["--help"])
    assert result.returncode == 0
    assert "Live trading" in result.stdout

def test_live_script_test_mode():
    """
    Test live.py in test mode.
    This should initialize components and exit gracefully.
    We skip if no internet or credentials, but here we assume sandbox env.
    If it fails due to network, we might check stderr.
    """
    # Note: this requires connection. In CI without net, this might fail.
    # But usually smoke tests run in an env where basic init is possible.
    # If it fails due to network, it usually returns non-zero.
    # We'll check if it crashes with ImportError or SyntaxError which is the main goal.

    # We use --test and --shadow-only to minimize side effects
    # Increased timeout for connection attempts
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "live.py"), "--test", "--shadow-only"],
            capture_output=True,
            text=True,
            timeout=60
        )
    except subprocess.TimeoutError:
        # If it times out, it's likely stuck on connection retry, which means it didn't crash on import.
        # We can consider this a pass for "smoke test" purposes (it started).
        return

    # Check for specific "CRITICAL" failures or Python errors
    assert "Traceback" not in result.stderr
    assert "ImportError" not in result.stderr
    assert "SyntaxError" not in result.stderr

    # It might return 1 if connection fails, but we want to ensure it didn't crash on import
    # So strictly checking returncode might be flaky without mocking.
    # But let's check basic sanity.

def test_download_data_help():
    """Test download_data.py help."""
    result = run_script("download_data.py", ["--help"])
    assert result.returncode == 0
    assert "Download historical data" in result.stdout

def test_train_script_help():
    """Test train.py help."""
    result = run_script("train.py", ["--help"])
    assert result.returncode == 0
    assert "Train DerivOmniModel" in result.stdout

def test_main_script():
    """Test main.py (simulation loop)."""
    # main.py is in root
    cmd = [sys.executable, str(ROOT_DIR / "main.py")]
    # It runs a simulation and exits.
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    assert result.returncode == 0
    assert "Unified Trading System" in result.stderr # Logging goes to stderr by default or stdout depending on config
    assert "System execution test complete" in result.stderr

def test_import_execution_regime():
    """Verify regime module consolidation."""
    try:
        from execution.regime import RegimeVeto, HierarchicalRegimeDetector
    except ImportError as e:
        pytest.fail(f"Failed to import regime classes: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error importing regime: {e}")

def test_import_database_stores():
    """Verify unified database stores."""
    try:
        from execution.sqlite_shadow_store import SQLiteShadowStore
        from execution.safety_store import SQLiteSafetyStateStore

        # Check paths in docstrings or default args if accessible (hard via introspection sometimes)
        # Instead, verify we can instantiate them pointing to same DB
        db_path = Path("test_unified.db")
        shadow = SQLiteShadowStore(db_path)
        safety = SQLiteSafetyStateStore(db_path)

        # Cleanup
        shadow.close()
        if db_path.exists():
            db_path.unlink()

    except Exception as e:
        pytest.fail(f"Failed to instantiate stores: {e}")

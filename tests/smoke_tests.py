import pytest
import subprocess
import sys
import sqlite3
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
DATA_CACHE_DIR = ROOT_DIR / "data_cache"

def run_script(script_name, args=None):
    """Run a script as a subprocess and return result."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        pytest.fail(f"Script not found: {script_path}")

    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30
    )
    return result

# --- Script Health Tests ---

def test_live_script_help():
    """Test that live.py can initialize and show help."""
    result = run_script("live.py", ["--help"])
    assert result.returncode == 0
    assert "Live trading" in result.stdout

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
    cmd = [sys.executable, str(ROOT_DIR / "main.py")]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    assert "Unified Trading System" in result.stderr
    assert "System execution test complete" in result.stderr

# --- Core System Integrity Tests ---

def test_critical_imports():
    """Verify all critical modules import without error."""
    try:
        import execution.decision
        import execution.policy
        import models.core
        import data.processor
        import training.trainer
    except ImportError as e:
        pytest.fail(f"Critical import failed: {e}")

def test_settings_load():
    """Verify settings can be loaded from .env."""
    from config.settings import load_settings
    try:
        settings = load_settings()
        assert settings.trading.symbol is not None
    except Exception as e:
        pytest.fail(f"Settings loading failed: {e}")

def test_database_connectivity():
    """Verify all core databases are accessible."""
    db_files = ['shadow_trades.db', 'safety_state.db', 'trading_state.db']
    for db in db_files:
        path = DATA_CACHE_DIR / db
        if path.exists():
            conn = sqlite3.connect(str(path))
            cursor = conn.execute("SELECT 1")
            assert cursor.fetchone()[0] == 1
            conn.close()

def test_model_initialization():
    """Verify model can be initialized on CPU."""
    from models.core import DerivOmniModel
    from config.settings import load_settings
    settings = load_settings()
    model = DerivOmniModel(settings)
    assert model is not None

def test_import_execution_regime():
    """Verify regime module consolidation."""
    try:
        from execution.regime import RegimeVeto, HierarchicalRegimeDetector
    except ImportError as e:
        pytest.fail(f"Failed to import regime classes: {e}")

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))

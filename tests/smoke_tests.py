"""
Smoke tests for script entry points.
Ensures that scripts can initialize without crashing.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"

def run_script(script_name: str, args: list[str]) -> subprocess.CompletedProcess:
    """Run a script with arguments and return result."""
    script_path = SCRIPTS_DIR / script_name
    cmd = [sys.executable, str(script_path)] + args

    # Ensure python-deriv-api is in PYTHONPATH
    python_path = str(SCRIPTS_DIR.parent)
    deriv_api_path = SCRIPTS_DIR.parent / "python-deriv-api"
    if deriv_api_path.exists():
        python_path += f":{deriv_api_path}"

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": python_path}
    )

@pytest.mark.smoke
def test_live_py_help():
    """Test that scripts/live.py can run --help."""
    result = run_script("live.py", ["--help"])
    assert result.returncode == 0
    assert "Live trading" in result.stdout

@pytest.mark.smoke
def test_live_py_test_mode():
    """Test that scripts/live.py initializes in test mode."""
    # This might require network or credentials, so we expect it to at least start
    # and fail gracefully or succeed.
    # We use --test flag which should just verify connection.
    result = run_script("live.py", ["--test"])

    # It might fail if no .env or no network, but we check output for startup logs
    # or specific error messages, avoiding NameError/ImportError.

    if result.returncode != 0:
        # If it fails, make sure it's not a syntax/import error
        assert "NameError" not in result.stderr
        assert "ImportError" not in result.stderr
        assert "ModuleNotFoundError" not in result.stderr
    else:
        assert "Test mode - connection verified" in result.stderr or \
               "Test mode - connection verified" in result.stdout

@pytest.mark.smoke
def test_train_py_help():
    """Test that scripts/train.py can run --help."""
    result = run_script("train.py", ["--help"])
    assert result.returncode == 0
    assert "usage:" in result.stdout

@pytest.mark.smoke
def test_download_data_py_help():
    """Test that scripts/download_data.py can run --help."""
    result = run_script("download_data.py", ["--help"])
    assert result.returncode == 0
    assert "Download historical data" in result.stdout

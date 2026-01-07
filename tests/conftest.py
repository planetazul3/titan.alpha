import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path for module resolution
# This ensures pytest can import project modules without PYTHONPATH
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Mock talib if not present to allow tests to run on environments without C-libs
try:
    import talib
except ImportError:
    msg = "TA-Lib not found. Mocking for structural verification."
    print(msg)
    
    mock_talib = MagicMock()
    # Common functions used in indicators
    mock_talib.RSI = MagicMock(return_value=[50.0] * 100)
    mock_talib.BBANDS = MagicMock(return_value=([100.0]*100, [100.0]*100, [100.0]*100))
    mock_talib.EMA = MagicMock(return_value=[100.0] * 100)
    mock_talib.ATR = MagicMock(return_value=[1.0] * 100)
    
    sys.modules["talib"] = mock_talib
    
# Mock deriv_api
try:
    import deriv_api
except ImportError:
    print("deriv_api not found. Mocking.")
    mock_deriv = MagicMock()
    mock_deriv.DerivAPI = MagicMock
    mock_deriv.APIError = Exception
    sys.modules["deriv_api"] = mock_deriv
    
# Mock pandera
try:
    import pandera
except ImportError:
    print("pandera not found. Mocking.")
    mock_pandera = MagicMock()
    mock_pandera.typing = MagicMock()
    sys.modules["pandera"] = mock_pandera
    sys.modules["pandera.typing"] = mock_pandera.typing
    
# Mock pandas
try:
    import pandas
except ImportError:
    print("pandas not found. Mocking.")
    sys.modules["pandas"] = MagicMock()


import pytest
from config.settings import load_settings


@pytest.fixture
def test_settings():
    """Shared Settings fixture using load_settings() for proper test isolation.
    
    This fixture correctly uses .env.test and handles frozen model constraints.
    For tests needing custom values, use settings.model_copy(update={...}).
    """
    return load_settings()
